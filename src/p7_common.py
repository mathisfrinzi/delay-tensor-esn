"""Partie 7 : socle commun aux protocoles 7.1 a 7.5 (rapport/4_comparaisons.md).

Protocole commun = 6.0 (rapport/2_regle.md), sauf mention contraire dans la
Partie 7 :

  - reservoir : x(t) = tanh(W_in @ [u(t), 1] + sum_{tau in S} W_tau @ x(t-tau)),
    leak = 1, etat nul avant t = 0, tampon circulaire de profondeur max(S) ;
  - W_in : (d, N) i.i.d. U[-0.5, 0.5] (canal 0 = u, canal 1 = biais 1) ;
  - chaque W_tau : N x N, gaussienne i.i.d. N(0,1) puis rescalee au rayon
    spectral EXACT rho/sqrt(K), K = |S| (calcul par eigvals) ; une graine
    numpy par seed, tirages dans l'ordre : W_in, puis les W_tau par tau
    croissant ;
  - budget : N = round(sqrt(400 000 / K)) ; l'ESN standard est S = {1},
    N = 632 (632^2 = 400 000) ;
  - readout : ridge float64 sur [x(t), 1], lambda choisi dans
    {1e-8, 1e-6, 1e-4, 1e-2} sur les 20 % finaux du bloc d'entrainement,
    puis reentraine sur tout le bloc d'entrainement ;
  - NRMSE = RMSE / std(cible) sur le bloc de test.

Taches et splits (6.0 etendu aux donnees reelles, memes conventions que
esn_ref.py / ngrc.py / cp_rank.py) :

  - narma30 : T = 5200, washout max(200, 3*max(S)), test = 1000 derniers pas ;
  - silverbox / wh : 30000 points, washout max(500, 3*max(S)), test = 10000
    derniers pas ; u normalise par std(u), y par std(y[:100000]) ;
  - mnist lignes : 5000 train / 1000 test, ridge selectionne sur les 20 %
    finaux du train (metrique = accuracy), meme pipeline que esn_ref.py ;
  - mc : profil MC_k (k = 1..40) de 6.1 : u i.i.d. U[0, 0.5], T = 12000,
    washout 1000, train 8000, test 3000.

Chaque script 7.x checkpointe dans OUT (par defaut
results/brut/tensor-suite/part7, surcharge P7_OUT),
au format {"cells": ..., "agg": ..., "meta": ...} comme les JSON existants.

P7_SMOKE=1 : grille reduite (T, budget, seeds) pour verifier que le code
tourne : ne JAMAIS rapporter ces resultats (meta["smoke"] = true).
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from chemins import BRUT  # noqa: E402
import gzip
import json
import os
import time
from pathlib import Path

import numpy as np

SMOKE = os.environ.get("P7_SMOKE") == "1"
OUT = Path(os.environ.get("P7_OUT", str(BRUT / "tensor-suite/part7")))
OUT.mkdir(parents=True, exist_ok=True)

BUDGET = 40_000 if SMOKE else 400_000
RIDGES = (1e-8, 1e-6, 1e-4, 1e-2)
SEEDS5 = [0, 1] if SMOKE else [0, 1, 2, 3, 4]
TIME_BUDGET = float(os.environ.get("P7_TIME", "240"))


# donnees
def narma_x(x, seed, T=None):
    """NARMA-x famille du projet (6.0) : u i.i.d. U[0 ; 0,5] ;
    y(t) = a y(t-1) + b y(t-1) sum_{i=1..x} y(t-i) + g u(t-x) u(t-1) + d,
    (a,b,g,d) = (0.3,0.05,1.5,0.1) si x = 10 sinon (0.2,0.004,1.5,0.001),
    y = 0 avant t = x. Divergence (|y| > 1e3 ou NaN) -> regeneration avec la
    graine + 1000, flag retourne. Renvoie (U=[u,1] (T,2), y, regenere)."""
    T = T or (1600 if SMOKE else 5200)
    regenerated = False
    for attempt in range(3):
        rng = np.random.default_rng(seed + 1000 * attempt)
        u = rng.uniform(0, 0.5, size=T)
        y = np.zeros(T, np.float32)
        a, b, g, d = (0.3, 0.05, 1.5, 0.1) if x == 10 else (0.2, 0.004, 1.5, 0.001)
        for t in range(x, T):
            y[t] = a * y[t - 1] + b * y[t - 1] * y[t - x:t].sum() + g * u[t - x] * u[t - 1] + d
        if np.isfinite(y).all() and np.abs(y).max() <= 1e3:
            break
        regenerated = True
    else:
        raise RuntimeError(f"NARMA-{x} diverge meme apres regeneration (seed {seed})")
    U = np.stack([u, np.ones_like(u)], 1).astype(np.float32)
    return U, y, regenerated


def narma30(seed, T=None):
    return narma_x(30, seed, T)


def load_phys(which):
    """Silverbox / Wiener-Hammerstein, meme chargement que esn_ref.py :
    30000 points, u/std(u), y/std(y[:100000]). Renvoie (U=[u,1], y)."""
    from scipy.io import loadmat

    if which == "silverbox":
        d = loadmat("/tmp/datasets/silverbox/SilverboxFiles/SNLS80mV.mat")
        u = d["V1"].ravel().astype(np.float64)
        y = d["V2"].ravel().astype(np.float64)
    elif which == "wh":
        d = loadmat("/tmp/datasets/WH.mat")
        u = d["uBenchMark"].ravel().astype(np.float64)
        y = d["yBenchMark"].ravel().astype(np.float64)
    else:
        raise ValueError(which)
    s = np.std(y[:100000])
    T = 2000 if SMOKE else 30000
    U = u[:T]
    Y = y[:T] / s
    return (np.stack([U / np.std(U), np.ones_like(U)], 1).astype(np.float32),
            Y.astype(np.float32))


def get_mnist(n_train=5000, n_test=1000):
    """MNIST depuis /tmp/mnist (telecharge si absent, comme suite_newtasks.py).
    Renvoie (Xtr (n,784) float32/255, ytr, Xte, yte)."""
    import urllib.request

    BASE = "/tmp/mnist"
    files = ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
             "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]
    os.makedirs(BASE, exist_ok=True)
    for f in files:
        p = os.path.join(BASE, f)
        if not os.path.exists(p):
            urllib.request.urlretrieve(f"https://ossci-datasets.s3.amazonaws.com/mnist/{f}", p)

    def imgs(fn):
        with gzip.open(os.path.join(BASE, fn), "rb") as f:
            dd = np.frombuffer(f.read(), np.uint8, offset=16)
        return dd.reshape(-1, 784).astype(np.float32) / 255.0

    def labs(fn):
        with gzip.open(os.path.join(BASE, fn), "rb") as f:
            return np.frombuffer(f.read(), np.uint8, offset=8).astype(np.int64)

    n_train = 200 if SMOKE else n_train
    n_test = 100 if SMOKE else n_test
    return imgs(files[0])[:n_train], labs(files[1])[:n_train], imgs(files[2])[:n_test], labs(files[3])[:n_test]


def mc_input(seed, T=None):
    """Entree du profil MC (6.1) : u i.i.d. U[0 ; 0,5]. Renvoie (U=[u,1], u)."""
    T = T or (4000 if SMOKE else 12000)
    rng = np.random.default_rng(seed + 5000)
    u = rng.uniform(0, 0.5, size=T).astype(np.float32)
    return np.stack([u, np.ones_like(u)], 1).astype(np.float32), u


# reservoir
KINDS_STRUCT = ("perm", "ident", "diag")


def slice_matrix(kind, rng, N):
    """Tranche W_tau AVANT rescale, rayon spectral exact 1 :
      - dense : gaussienne i.i.d. N(0,1), rescalee par eigvals ;
      - perm  : matrice de permutation aleatoire (racines de l'unite) ;
      - ident : I ;
      - diag  : diagonale de ±1 aleatoires.
    Le rayon spectral de perm/ident/diag est exactement 1 : pas de calcul."""
    if kind == "dense":
        M = rng.normal(0, 1, (N, N))
        r = max(np.abs(np.linalg.eigvals(M)).max(), 1e-12)
        return M / r
    if kind == "perm":
        return np.eye(N)[rng.permutation(N)]
    if kind == "ident":
        return np.eye(N)
    if kind == "diag":
        return np.diag(rng.choice((-1.0, 1.0), size=N))
    raise ValueError(f"kind inconnu : {kind}")


def make_reservoir(seed, taus, rho, N=None, d=2, kinds=None):
    """Ordre des tirages (6.0) : W_in (d,N) puis les tranches par tau croissant.
    kinds : dict tau -> kind ('dense' par defaut). Chaque tranche est
    rescalee au rayon spectral exact rho/sqrt(K).
    Renvoie (Win, W (K,N,N), taus_tries, N)."""
    taus = sorted(taus)
    K = len(taus)
    N = N or int(round((BUDGET / K) ** 0.5))
    rng = np.random.default_rng(seed)
    Win = rng.uniform(-0.5, 0.5, size=(d, N)).astype(np.float32)
    W = np.empty((K, N, N), np.float32)
    scale = rho / np.sqrt(K)
    for i, tau in enumerate(taus):
        kind = (kinds or {}).get(tau, "dense")
        W[i] = (slice_matrix(kind, rng, N) * scale).astype(np.float32)
    return Win, W, taus, N


def make_cp_reservoir(seed, taus, rho, r, N=None, d=2):
    """Tranges CP de rang r (2.7) : W_tau = A_tau @ B_tau.T, rescalees au meme
    rayon effectif rho/sqrt(K). Ordre des tirages : W_in, puis (A, B) par tau
    croissant. Renvoie (Win, W (K,N,N), taus, N)."""
    taus = sorted(taus)
    K = len(taus)
    N = N or int(round((BUDGET / K) ** 0.5))
    rng = np.random.default_rng(seed)
    Win = rng.uniform(-0.5, 0.5, size=(d, N)).astype(np.float32)
    W = np.empty((K, N, N), np.float32)
    scale = rho / np.sqrt(K)
    for i in range(K):
        A = rng.normal(0, 1, (N, r)).astype(np.float32)
        B = rng.normal(0, 1, (N, r)).astype(np.float32)
        M = (A @ B.T).astype(np.float64)
        rad = max(np.abs(np.linalg.eigvals(M)).max(), 1e-12)
        W[i] = (M * (scale / rad)).astype(np.float32)
    return Win, W, taus, N


def _fast_handlers(W):
    """Detection des tranches structurees (ident / diag ±1 / permutation) a
    partir des matrices : renvoie un handler par tranche, None si dense.
    Handler = ("diag", v (N,)) pour diagonale*echelle (inclut l'identite) ou
    ("perm", s, idx (N,)) pour s*P. Detection exacte : les tranches construites
    par slice_matrix ont des zeros exacts hors structure ; une gaussienne
    dense a N^2 nonzeros. C'est le chemin rapide du "budget honnete" de 7.1 :
    les tranches structurees coutent ~0 a appliquer."""
    K = len(W)
    N = W.shape[1]
    handlers = []
    for i in range(K):
        M = W[i]
        nz = M != 0
        if nz.sum() == N and (nz.sum(axis=1) == 1).all():
            idx = nz.argmax(axis=1)
            vals = M[np.arange(N), idx]
            if (nz.sum(axis=0) == 1).all() and np.all(vals == vals[0]):
                handlers.append(("perm", float(vals[0]), idx))
                continue
            handlers.append(("diag", np.diag(M).astype(np.float32)))
            continue
        handlers.append(None)
    return handlers


def _apply_slices(hs, W, sel, fast, dense_idx):
    """Somme des contributions W_k @ x(t-tau_k) ; hs = hist[sel] (K, ...).
    Chemin rapide pour les tranches structurees, einsum pour les denses."""
    if fast is None:
        return np.einsum("kij,kj->i", W, hs)
    drive = np.zeros(hs.shape[1], np.float32)
    if len(dense_idx):
        drive += np.einsum("kij,kj->i", W[dense_idx], hs[dense_idx])
    for i, h in enumerate(fast):
        if h is None:
            continue
        if h[0] == "diag":
            drive += h[1] * hs[i]
        else:  # perm
            drive += h[1] * hs[i][h[2]]
    return drive


def run_states(U, Win, W, taus, leak=None):
    """Simulation du reservoir (6.0) : U (T,d) canal biais inclus, etat nul
    avant t = 0, tampon circulaire hist[tau-1] = x(t-tau).
    leak = None -> leak 1 (protocole 6.0) ; sinon scalaire ou vecteur a_i
    (DTS / fuite) : x = (1-a) x(t-1) + a tanh(...). Renvoie F (T,N) float32."""
    T, d = U.shape
    N = Win.shape[1]
    taus = sorted(taus)
    tmax = max(taus)
    sel = np.asarray(taus) - 1
    fast = _fast_handlers(W)
    if fast is not None and not any(h is not None for h in fast):
        fast = None
    dense_idx = np.array([i for i, h in enumerate(fast) if h is None],
                         dtype=int) if fast is not None else None
    hist = np.zeros((tmax, N), np.float32)
    F = np.empty((T, N), np.float32)
    zero_row = np.zeros((1, N), np.float32)
    for t in range(T):
        drive = U[t] @ Win + _apply_slices(hist[sel], W, sel, fast, dense_idx)
        z = np.tanh(drive)
        if leak is None:
            x_new = z
        else:
            x_new = (1.0 - leak) * hist[0] + leak * z
        F[t] = x_new
        hist = np.concatenate([x_new[None] if hasattr(x_new, "shape") else zero_row,
                               hist[:-1]], axis=0)
    return F


def states_mnist(X, Win, W, taus, chunk=500):
    """MNIST ligne par ligne : 28 pas de 28 pixels, Win (28, N) sans canal
    biais (comme esn_ref.py), etat et historique remis a zero a chaque image.
    Renvoie F (B, N)."""
    B = len(X)
    N = Win.shape[1]
    taus = sorted(taus)
    tmax = max(taus)
    sel = np.asarray(taus) - 1
    fast = _fast_handlers(W)
    if fast is not None and not any(h is not None for h in fast):
        fast = None
    dense_idx = np.array([i for i, h in enumerate(fast) if h is None],
                         dtype=int) if fast is not None else None
    out = np.empty((B, N), np.float32)
    for lo in range(0, B, chunk):
        Bk = X[lo:lo + chunk].reshape(-1, 28, 28)
        nb = len(Bk)
        hist = np.zeros((tmax, nb, N), np.float32)
        x = np.zeros((nb, N), np.float32)
        for t in range(28):
            hs = hist[sel]
            if fast is None:
                rec = np.tensordot(hs, W, axes=([0, 2], [0, 2]))
            else:
                rec = np.zeros((nb, N), np.float32)
                if len(dense_idx):
                    rec += np.tensordot(hs[dense_idx], W[dense_idx],
                                        axes=([0, 2], [0, 2]))
                for i, h in enumerate(fast):
                    if h is None:
                        continue
                    if h[0] == "diag":
                        rec += h[1][None, :] * hs[i]
                    else:  # perm
                        rec += h[1] * hs[i][:, h[2]]
            drive = Bk[:, t, :] @ Win + rec
            x = np.tanh(drive)
            hist = np.concatenate([x[None], hist[:-1]], axis=0)
        out[lo:lo + nb] = x
    return out


# readout
def _ridge_solve(Xb, yb, lam):
    A = Xb.T @ Xb + lam * np.eye(Xb.shape[1])
    return np.linalg.solve(A, Xb.T @ yb)


def _bias(F):
    return np.hstack([F, np.ones((len(F), 1), np.float32)]).astype(np.float64)


def ridge_readout(F, y, wash, tr_end, te0, te1, ridges=RIDGES):
    """Readout 6.0 : lambda choisi sur les 20 % finaux du bloc d'entrainement,
    reentrainement sur tout le bloc, NRMSE sur [te0, te1).
    Renvoie (nrmse_test, lam_choisi, nrmse_validation) — 14.2 : la validation
    est le critere de selection du rho, le test n'est jamais un critere."""
    Fb = _bias(F)
    yv = np.asarray(y, np.float64)
    v0 = wash + int(0.8 * (tr_end - wash))
    best_lam, best_val = ridges[0], np.inf
    for lam in ridges:
        w = _ridge_solve(Fb[wash:v0], yv[wash:v0], lam)
        resid = Fb[v0:tr_end] @ w - yv[v0:tr_end]
        val = float(np.sqrt(np.mean(resid ** 2)) / np.std(yv[v0:tr_end]))
        if val < best_val:
            best_val, best_lam = val, lam
    w = _ridge_solve(Fb[wash:tr_end], yv[wash:tr_end], best_lam)
    pred = Fb[te0:te1] @ w
    nrmse = float(np.sqrt(np.mean((pred - yv[te0:te1]) ** 2)) / np.std(yv[te0:te1]))
    # 14.2 : best_val = NRMSE de validation au lambda choisi (fit 80 %)
    return nrmse, best_lam, best_val


def ridge_readout_given_lam(F, y, wash, tr_end, te0, te1, lam):
    """Fit ridge (lam fixe) sur [wash, tr_end), NRMSE sur le test : pour la
    recherche aleatoire de 7.4 ou le ridge est un parametre echantillonne."""
    Fb = _bias(F)
    yv = np.asarray(y, np.float64)
    w = _ridge_solve(Fb[wash:tr_end], yv[wash:tr_end], lam)
    pred = Fb[te0:te1] @ w
    return float(np.sqrt(np.mean((pred - yv[te0:te1]) ** 2)) / np.std(yv[te0:te1]))


def mnist_readout(Ftr, ytr, Fte, yte, ridge_grid=RIDGES):
    """MNIST : ridge choisi sur les 20 % finaux du train (accuracy), puis
    reentrainement sur tout le train. Renvoie (acc_test, lam)."""
    Ftrb, Fteb = _bias(Ftr), _bias(Fte)
    v0 = int(0.8 * len(Ftrb))
    Tt = np.zeros((len(ytr), 10))
    Tt[np.arange(len(ytr)), ytr] = 1.0
    best_lam, best_acc = ridge_grid[0], -1.0
    for lam in ridge_grid:
        A = Ftrb[:v0].T @ Ftrb[:v0] + lam * np.eye(Ftrb.shape[1])
        Wout = np.linalg.solve(A, Ftrb[:v0].T @ Tt[:v0])
        acc = float((np.argmax(Ftrb[v0:] @ Wout, axis=1) == ytr[v0:]).mean())
        if acc > best_acc:
            best_acc, best_lam = acc, lam
    A = Ftrb.T @ Ftrb + best_lam * np.eye(Ftrb.shape[1])
    Wout = np.linalg.solve(A, Ftrb.T @ Tt)
    acc = float((np.argmax(Fteb @ Wout, axis=1) == yte).mean())
    # 14.2 : best_acc = accuracy de validation au lambda choisi (fit 80 %)
    return acc, best_lam, best_acc


def mc_profile(F, u, kmax=40, wash=None, tr_end=None, with_val=False):
    """Profil MC_k de 6.1 : cible u(t-k), MC_k = Pearson^2 sur le test.
    Splits 6.1 : T = 12000, washout 1000, train [1000, 9000), test 3000
    (reduits en mode smoke). Renvoie {k: MC_k} ; avec with_val=True (14.2),
    renvoie (test, validation) — le profil de validation (fit 80 %) sert au
    choix du rho, jamais le test."""
    T = len(u)
    wash = wash or (200 if SMOKE else 1000)
    ntr = 800 if SMOKE else 8000
    tr_end = tr_end or (wash + ntr)
    Fb = _bias(F)
    uv = np.asarray(u, np.float64)
    v0 = wash + int(0.8 * (tr_end - wash))
    out = {}
    out_val = {}
    for k in range(1, kmax + 1):
        z = np.zeros(T, np.float64)
        z[k:] = uv[:-k]
        best_lam, best_val = RIDGES[0], np.inf
        for lam in RIDGES:
            w = _ridge_solve(Fb[wash:v0], z[wash:v0], lam)
            resid = Fb[v0:tr_end] @ w - z[v0:tr_end]
            val = float(np.sqrt(np.mean(resid ** 2)) / np.std(z[v0:tr_end]))
            if val < best_val:
                best_val, best_lam = val, lam
        if with_val:
            wv = _ridge_solve(Fb[wash:v0], z[wash:v0], best_lam)
            pv = Fb[v0:tr_end] @ wv
            if np.std(pv) < 1e-12:
                out_val[str(k)] = 0.0
            else:
                rv = np.corrcoef(pv, z[v0:tr_end])[0, 1]
                out_val[str(k)] = float(max(rv * rv, 0.0))
        w = _ridge_solve(Fb[wash:tr_end], z[wash:tr_end], best_lam)
        pred = Fb[tr_end:] @ w
        if np.std(pred) < 1e-12:
            out[str(k)] = 0.0
        else:
            r = np.corrcoef(pred, z[tr_end:])[0, 1]
            out[str(k)] = float(max(r * r, 0.0))
    return (out, out_val) if with_val else out


# splits
def seq_splits(taus):
    """Splits 6.0 : NARMA T = 5200, washout max(200, 3 max(S)), test 1000
    derniers ; phys T = 30000, washout max(500, 3 max(S)), test 10000 derniers.
    Renvoie dict(T, wash, tr_end, te0, te1)."""
    ms = max(taus)
    if SMOKE:
        T, nte, wash = 1600, 300, max(200, 3 * ms)
    else:
        T, nte, wash = 5200, 1000, max(200, 3 * ms)
    return {"T": T, "wash": wash, "tr_end": T - nte, "te0": T - nte, "te1": T}


def phys_splits(taus):
    ms = max(taus)
    T, nte = 30000, (2000 if SMOKE else 10000)
    return {"T": T, "wash": max(500, 3 * ms), "tr_end": T - nte, "te0": T - nte, "te1": T}


def task_splits(task, taus):
    return seq_splits(taus) if task.startswith("narma") else phys_splits(taus)


# checkpoint
def load_json(path):
    if path.exists():
        return json.load(open(path, encoding="utf-8-sig"))
    return {"cells": {}, "agg": {}, "meta": {}}


def save_json(results, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(path, "w"), indent=1)


def cell_add(results, key, seed, value, val=None, **cellmeta):
    """value : metrique de test. val (14.2) : metrique de validation, critere
    de choix du rho ; les cellules sans val sont recalees par cell_missing."""
    cell = results["cells"].setdefault(key, {"seeds": {}})
    cell["seeds"][str(seed)] = value
    if val is not None:
        cell.setdefault("seeds_val", {})[str(seed)] = val
    cell.update(cellmeta)
    return cell


def cell_missing(results, key, seeds):
    # 14.2 : une seed n'est complete que si elle a aussi sa metrique de
    # validation (les vieilles cellules sans val sont recalees)
    cell = results["cells"].get(key, {"seeds": {}})
    return [s for s in seeds
            if str(s) not in cell["seeds"]
            or str(s) not in cell.get("seeds_val", {})]


def aggregate(results, higher_is_better=False):
    """Recalcule agg : moyenne / std / n par cellule (sur les seeds presentes)."""
    agg = {}
    for key, cell in results["cells"].items():
        if not isinstance(cell, dict) or "seeds" not in cell:
            continue
        vals = list(cell["seeds"].values())
        if not vals:
            continue
        scalars = [v for v in vals if isinstance(v, (int, float))]
        entry = {"n": len(vals)}
        if scalars:
            arr = np.asarray(scalars, np.float64)
            entry["mean"] = float(arr.mean())
            entry["std"] = float(arr.std())
            entry["min"] = float(arr.min())
            entry["max"] = float(arr.max())
        # 14.2 : stats de validation (critere de selection du rho)
        vals_v = [v for v in cell.get("seeds_val", {}).values()
                  if isinstance(v, (int, float))]
        if vals_v:
            arrv = np.asarray(vals_v, np.float64)
            entry["mean_val"] = float(arrv.mean())
            entry["std_val"] = float(arrv.std())
        agg[key] = entry
    results["agg"] = agg
    return agg


# stats (7.4)
def wilcoxon_paired(a, b):
    """Test de Wilcoxon apparie (a et b = memes seeds). Renvoie p ou None."""
    from scipy.stats import wilcoxon

    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if np.allclose(a, b):
        return 1.0
    try:
        res = wilcoxon(a, b)
        return float(res.pvalue)
    except ValueError:
        return None


def bootstrap_ci(x, stat=np.mean, n=10000, alpha=0.05, seed=0):
    """IC bootstrap a 95 % sur stat(x) : resamples indices avec remise."""
    x = np.asarray(x, np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n, len(x)))
    samples = stat(x[idx], axis=1) if stat is not np.mean else x[idx].mean(axis=1)
    lo, hi = np.percentile(samples, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def meta_init(results, name, **params):
    results.setdefault("meta", {})
    results["meta"]["script"] = name
    results["meta"]["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    results["meta"]["smoke"] = SMOKE
    results["meta"].update(params)
