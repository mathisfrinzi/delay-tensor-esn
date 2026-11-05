"""Partie 6 : protocole commun 6.0 (rapport/2_regle.md), implemente une fois.

Reservoir : x(t) = tanh(W_in [u(t),1] + sum_{tau in S} W_tau x(t-tau)), leak=1,
etat nul avant t=0. W_in : 2xN iid U[-0.5,0.5]. Chaque W_tau : NxN gaussienne
iid, divisee par son rayon spectral (eigvals), fois rho/sqrt(K), K=|S|.
UNE graine numpy par seed, tirages dans l'ordre : W_in puis W_tau par tau
croissant. Budget N = round(sqrt(400k/K)) ; ESN = S={1}, N=632.

Readout : ridge float64 sur [x,1], lambda in {1e-8,1e-6,1e-4,1e-2} choisi sur
les 20 % finaux du bloc d'entrainement, puis reentraine sur tout le bloc.
NRMSE = RMSE / std(cible) sur le test.

NARMA-x : u iid U[0;0.5] ; y(t) = 0.2 y(t-1) + 0.004 y(t-1) sum_{i<=x} y(t-i)
+ 1.5 u(t-x) u(t-1) + 0.001 pour t >= x, y=0 avant. Divergence (|y|>1e3 ou
NaN) -> regenerer u avec graine+1000, note dans le JSON.

Splits (protocole std) : T=5200, washout max(200, 3*max(S)), entrainement
[washout, T-1000), test [T-1000, T).
"""
import numpy as np, json, time
from pathlib import Path

OUT = Path(__import__("os").environ.get("P6_OUT",
    str(Path(__file__).resolve().parents[1] / "results")))
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = [0, 1, 2, 3, 4]
BUDGET = 400_000
LAMBDAS = [1e-8, 1e-6, 1e-4, 1e-2]
TIME_BUDGET = float(__import__("os").environ.get("TEMPS_MAX", "205"))


def make_reservoir(N, taus, rho, seed):
    """W_in puis W_tau par tau croissant, une seule graine (protocole 6.0)."""
    rng = np.random.default_rng(seed)
    Win = rng.uniform(-0.5, 0.5, size=(2, N)).astype(np.float32)
    taus = sorted(taus)
    K = len(taus)
    W = np.empty((K, N, N), np.float32)
    for i, tau in enumerate(taus):
        M = rng.normal(0, 1, (N, N)).astype(np.float32)
        r = max(np.abs(np.linalg.eigvals(M.astype(np.float64))).max(), 1e-12)
        W[i] = M * ((rho / np.sqrt(K)) / r)
    return Win, W


def make_reservoir_rank(N, taus, rho, seed, r):
    """Variante rang faible : W_tau = A B^T (rang r), rescale a rho/sqrt(K)
    (construction de src/cp_rank.py, tirages dans l'ordre 6.0)."""
    rng = np.random.default_rng(seed)
    Win = rng.uniform(-0.5, 0.5, size=(2, N)).astype(np.float32)
    taus = sorted(taus)
    K = len(taus)
    W = np.empty((K, N, N), np.float32)
    for i, tau in enumerate(taus):
        A = rng.normal(0, 1, (N, r)).astype(np.float32)
        B = rng.normal(0, 1, (N, r)).astype(np.float32)
        M = (A @ B.T).astype(np.float32)
        rad = max(np.abs(np.linalg.eigvals(M.astype(np.float64))).max(), 1e-12)
        W[i] = M * ((rho / np.sqrt(K)) / rad)
    return Win, W


def run_states(u, taus, Win, W):
    """Retourne F (T, N) ; x(t) a vu u(t)."""
    T = len(u)
    N = Win.shape[1]
    taus = sorted(taus)
    tmax = max(taus)
    sel = np.array(taus) - 1
    hist = np.zeros((tmax, N), np.float32)
    x_ = np.zeros(N, np.float32)
    F = np.empty((T, N), np.float32)
    ub = np.stack([u, np.ones_like(u)], 1).astype(np.float32)
    WT = [np.ascontiguousarray(W[i]) for i in range(len(taus))]
    for t in range(T):
        acc = ub[t] @ Win
        for i, s in enumerate(sel):
            acc += WT[i] @ hist[s]
        x_ = np.tanh(acc)
        F[t] = x_
        hist = np.concatenate([x_[None], hist[:-1]], axis=0)
    return F


def augment(F):
    return np.hstack([F, np.ones((len(F), 1), np.float64)])


def ridge_select(Ftr, ztr, Fte):
    """Choix de lambda sur les 20 % finaux de Ftr, reentrainement complet.
    Retourne (pred_test, lambda, nrmse_validation, A_full). Ftr/Fte : deja
    augmentes, float64. 14.2 : nrmse_validation = erreur des 20 % finaux
    (fit sur les 80 % premiers) au lambda choisi ; c'est le critere de
    selection du rho, le test n'est jamais un critere."""
    n = len(Ftr)
    m = int(0.8 * n)
    A_main = Ftr[:m].T @ Ftr[:m]
    A_full = Ftr.T @ Ftr
    d = A_full.shape[0]
    eye = np.eye(d)
    zm = ztr[:m]
    best = (np.inf, None)
    for lam in LAMBDAS:
        Wo = np.linalg.solve(A_main + lam * eye, Ftr[:m].T @ zm)
        v = float(np.sqrt(np.mean((Ftr[m:] @ Wo - ztr[m:]) ** 2)))
        if v < best[0]:
            best = (v, lam)
    lam = best[1]
    val = float(best[0] / np.std(ztr[m:]))
    Wo = np.linalg.solve(A_full + lam * eye, Ftr.T @ ztr)
    pred = Fte @ Wo
    return pred, lam, val, A_full


def nrmse_of(pred, zte):
    return float(np.sqrt(np.mean((pred - zte) ** 2)) / np.std(zte))


def narma_x(seed, x, T=5200, variant="std"):
    """variantes 6.6 : 'std', 'A' (u(t).u(t-x)), 'B' (u(t-2).u(t-x)),
    'C' (entree lissee, moyenne glissante 3 pas, meme moyenne/std que U[0;0.5]).
    Retourne (u, y, generations_de_divergence)."""
    gen = 0
    while True:
        rng = np.random.default_rng(seed + 1000 * gen)
        w = rng.uniform(0, 0.5, size=T)
        if variant == "C":
            s = np.empty_like(w)
            s[0] = w[0]; s[1] = (w[0] + w[1]) / 2
            s[2:] = (w[:-2] + w[1:-1] + w[2:]) / 3
            u = (s - s.mean()) / s.std() * (0.5 / np.sqrt(12)) + 0.25
        else:
            u = w
        y = np.zeros(T, np.float64)
        run = 0.0
        ok = True
        for t in range(x, T):
            run += y[t - 1] - (y[t - x - 1] if t >= x + 1 else 0.0)
            if variant in ("std", "C"):
                g = 1.5 * u[t - x] * u[t - 1]
            elif variant == "A":
                g = 1.5 * u[t] * u[t - x]
            elif variant == "B":
                g = 1.5 * u[t - 2] * u[t - x]
            else:
                raise ValueError(variant)
            y[t] = 0.2 * y[t - 1] + 0.004 * y[t - 1] * run + g + 0.001
            if not np.isfinite(y[t]) or abs(y[t]) > 1e3:
                ok = False
                break
        if ok:
            return u.astype(np.float32), y.astype(np.float64), gen
        gen += 1


def eval_narma(taus, rho, seed, x=30, variant="std", T=5200, rank=None):
    """Un run complet du protocole std. Retourne dict(nrmse, lam, gen, N)."""
    N = int(round((BUDGET / len(taus)) ** 0.5))
    if rank is None:
        Win, W = make_reservoir(N, taus, rho, seed)
    else:
        Win, W = make_reservoir_rank(N, taus, rho, seed, rank)
    u, y, gen = narma_x(seed, x, T, variant)
    F = run_states(u, taus, Win, W)
    wash = max(200, 3 * max(taus))
    i_te = T - 1000
    Ftr = augment(F[wash:i_te]); ztr = y[wash:i_te]
    Fte = augment(F[i_te:]); zte = y[i_te:]
    pred, lam, _, _ = ridge_select(Ftr, ztr, Fte)
    return {"nrmse": nrmse_of(pred, zte), "lam": lam, "gen": gen, "N": N}


def semigroup(taus, limit):
    """(attenable[0..limit], nb_sauts_min[0..limit]) ; inf = inatteignable."""
    att = np.zeros(limit + 1, bool); att[0] = True
    h = np.full(limit + 1, np.inf); h[0] = 0
    for L in range(1, limit + 1):
        best = np.inf
        for tau in taus:
            if L >= tau and att[L - tau] and h[L - tau] + 1 < best:
                best = h[L - tau] + 1
                att[L] = True
        h[L] = best
    return att, h


def renewal_e(taus, rho, Lmax):
    """e(L) du champ moyen 5.2 : e(0)=1, e(L) = sum_{tau<=L} g_tau e(L-tau)."""
    K = len(taus)
    g = rho * rho / K
    e = np.zeros(Lmax + 1)
    e[0] = 1.0
    for L in range(1, Lmax + 1):
        s = 0.0
        for tau in taus:
            if tau <= L:
                s += g * e[L - tau]
        e[L] = s
    return e


def save(results, name):
    json.dump(results, open(OUT / f"results_p6_{name}.json", "w"), indent=1)
