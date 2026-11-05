"""Partie 2 : structure des tau. Variantes = (ensembles tau, sparsite par tau, profil s(tau)).
Taches : narma30, silverbox (tronque), wh (tronque), mnist (ligne par ligne, 5k).

Protocole (correction 14.2 du rapport, 2026-10-29) : le meilleur rho par seed est
choisi sur un segment de VALIDATION (NVAL[task] pas avant le test ; pour mnist les
1000 dernieres images du train), jamais sur le test ; le readout final est reajuste
sur train+validation. Les cellules marquees "proto": "rho_sur_validation" dans le
JSON sont reprises telles quelles, les autres sont recalculees."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from chemins import BRUT  # noqa: E402
import numpy as np, json, sys, os, gzip
from pathlib import Path


OUT = (BRUT / "tensor-suite/part2")
OUT.mkdir(parents=True, exist_ok=True)
RJ = OUT / "results_part2.json"
results = json.load(open(RJ)) if RJ.exists() else {}

# noyau tenseur generalise
def slices(N, taus, rho, seed, p_by_tau=None, s_by_tau=None):
    """Tire les tranches W_tau avec sparsite p(tau) et profil d'echelle s(tau).
    Chaque tranche : gaussien -> masque creux -> rayon spectral rho/sqrt(K) -> * s(tau)."""
    rng = np.random.default_rng(seed)
    K = len(taus)
    W = []
    for i, tau in enumerate(taus):
        M = rng.normal(0, 1, (N, N)).astype(np.float32)
        p = 1.0 if p_by_tau is None else p_by_tau(tau)
        if p < 1.0:
            mask = rng.random((N, N)) < p
            M *= mask
        r = max(np.abs(np.linalg.eigvals(M.astype(np.float64))).max(), 1e-12)
        M *= (rho / np.sqrt(K)) / r
        if s_by_tau is not None:
            M = M * s_by_tau(tau)
        W.append(M.astype(np.float32))
    return W

def states_seq(u, N, taus, rho, seed, p_by_tau=None, s_by_tau=None, Win=None):
    """u: (T, d). -> (T, N)."""
    rng = np.random.default_rng(seed + 7 if Win is None else 0)
    T, d = u.shape
    if Win is None:
        Win = rng.uniform(-0.5, 0.5, size=(d, N)).astype(np.float32)
    W = slices(N, taus, rho, seed, p_by_tau, s_by_tau)
    tmax = max(taus)
    x = np.zeros((N,), np.float32)
    hist = np.zeros((tmax, N), np.float32)
    F = np.empty((T, N), np.float32)
    for t in range(T):
        drive = u[t] @ Win
        for i, tau in enumerate(taus):
            drive = drive + hist[tau - 1] @ W[i]
        x = np.tanh(drive)
        hist = np.concatenate([x[None], hist[:-1]], axis=0)
        F[t] = x
    return F

def states_batch(Xb, N, taus, rho, seed, p_by_tau=None, s_by_tau=None, chunk=4000):
    """Xb: (B, 28, 28) -> (B, N)."""
    rng = np.random.default_rng(seed)
    Win = (rng.uniform(-0.5, 0.5, size=(28, N))).astype(np.float32)
    W = slices(N, taus, rho, seed + 7, p_by_tau, s_by_tau)
    tmax = max(taus)
    out = np.empty((len(Xb), N), np.float32)
    for lo in range(0, len(Xb), chunk):
        B = Xb[lo:lo+chunk]
        x = np.zeros((len(B), N), np.float32)
        hist = np.zeros((tmax, len(B), N), np.float32)
        for t in range(28):
            drive = B[:, t, :] @ Win
            for i, tau in enumerate(taus):
                drive = drive + hist[tau-1] @ W[i]
            x = np.tanh(drive)
            hist = np.concatenate([x[None], hist[:-1]], axis=0)
        out[lo:lo+len(B)] = x
    return out

def nrmse_score(F, y, wash, ntest, ridge=1e-8):
    Fb = np.hstack([F, np.ones((len(F), 1), np.float32)]).astype(np.float64)
    y = np.asarray(y, np.float64)
    Ftr, ytr = Fb[wash:-ntest], y[wash:-ntest]
    Fte, yte = Fb[-ntest:], y[-ntest:]
    A = Ftr.T @ Ftr + ridge * np.eye(Ftr.shape[1])
    Wout = np.linalg.solve(A, Ftr.T @ ytr)
    return float(np.sqrt(np.mean(((Fte @ Wout) - yte) ** 2)) / np.std(yte))

def classify_acc(Ftr, ytr, Fte, yte, ridge=1e-6):
    Ftr = np.hstack([Ftr, np.ones((len(Ftr), 1), np.float32)]).astype(np.float64)
    Fte = np.hstack([Fte, np.ones((len(Fte), 1), np.float32)]).astype(np.float64)
    T = np.zeros((len(ytr), 10)); T[np.arange(len(ytr)), ytr] = 1.0
    A = Ftr.T @ Ftr + ridge * np.eye(Ftr.shape[1])
    Wout = np.linalg.solve(A, Ftr.T @ T)
    return float((np.argmax(Fte @ Wout, axis=1) == yte).mean())

# 14.2 : segments de validation (choix de rho, jamais sur le test)
NVAL = {"narma30": 1000, "silverbox": 2000, "wh": 2000}
NVAL_MNIST = 1000

def nrmse_val_test(F, y, wash, ntest, nval, ridge=1e-8):
    """Choix de rho sur la validation : readout sur le train seul -> erreur de
    validation ; readout final reajuste sur train+validation -> erreur de test."""
    Fb = np.hstack([F, np.ones((len(F), 1), np.float32)]).astype(np.float64)
    y = np.asarray(y, np.float64)
    Ftr, ytr = Fb[wash:-ntest-nval], y[wash:-ntest-nval]
    Fva, yva = Fb[-ntest-nval:-ntest], y[-ntest-nval:-ntest]
    A = Ftr.T @ Ftr + ridge * np.eye(Ftr.shape[1])
    Wout = np.linalg.solve(A, Ftr.T @ ytr)
    val = float(np.sqrt(np.mean((Fva @ Wout - yva) ** 2)) / np.std(yva))
    Ftv, ytv = Fb[wash:-ntest], y[wash:-ntest]
    A2 = Ftv.T @ Ftv + ridge * np.eye(Ftv.shape[1])
    Wout2 = np.linalg.solve(A2, Ftv.T @ ytv)
    Fte, yte = Fb[-ntest:], y[-ntest:]
    test = float(np.sqrt(np.mean((Fte @ Wout2 - yte) ** 2)) / np.std(yte))
    return val, test

# donnees
def narma(n, seed, T=5200):
    rng = np.random.default_rng(seed)
    u = rng.uniform(0, 0.5, size=T)
    y = np.zeros(T, np.float32)
    a, b, g, d = (0.3, 0.05, 1.5, 0.1) if n == 10 else (0.2, 0.004, 1.5, 0.001)
    for t in range(n, T):
        y[t] = a * y[t-1] + b * y[t-1] * y[t-n:t].sum() + g * u[t-n] * u[t-1] + d
    return np.stack([u, np.ones_like(u)], 1).astype(np.float32), y

def _load_phys(which):
    from scipy.io import loadmat
    if which == "silverbox":
        d = loadmat("/tmp/datasets/silverbox/SilverboxFiles/SNLS80mV.mat")
        u = d["V1"].ravel().astype(np.float64); y = d["V2"].ravel().astype(np.float64)
    else:
        d = loadmat("/tmp/datasets/WH.mat")
        u = d["uBenchMark"].ravel().astype(np.float64); y = d["yBenchMark"].ravel().astype(np.float64)
    n_train, n_test = 20000, 10000
    U = u[:n_train + n_test]; Y = y[:n_train + n_test] / np.std(y[:100000])
    uu = np.stack([U / np.std(U), np.ones_like(U)], 1).astype(np.float32)
    return uu, Y / np.std(y[:100000])

def mnist_data(n_train=5000, n_test=1000):
    import urllib.request
    BASE = "/tmp/mnist"
    os.makedirs(BASE, exist_ok=True)
    files = ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
             "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]
    for f in files:
        p = os.path.join(BASE, f)
        if not os.path.exists(p):
            urllib.request.urlretrieve(f"https://ossci-datasets.s3.amazonaws.com/mnist/{f}", p)
    def imgs(fn):
        with gzip.open(os.path.join(BASE, fn), "rb") as f:
            d = np.frombuffer(f.read(), np.uint8, offset=16)
        return d.reshape(-1, 28, 28).astype(np.float32) / 255.0
    def labs(fn):
        with gzip.open(os.path.join(BASE, fn), "rb") as f:
            return np.frombuffer(f.read(), np.uint8, offset=8).astype(np.int64)
    return imgs(files[0])[:n_train], labs(files[1])[:n_train], imgs(files[2])[:n_test], labs(files[3])[:n_test]

# variantes
VARIANTS = {
    # ensembles de tau (famille A)
    "tau_1_10":  dict(taus=list(range(1, 11)), note="baseline uniforme"),
    "tau_1_5":   dict(taus=list(range(1, 6))),
    "tau_1_2":   dict(taus=[1, 2]),
    "tau_10_19": dict(taus=list(range(10, 20)), note="fenetre tardive"),
    "tau_1_20":  dict(taus=list(range(1, 21)), note="20 tranches"),
    "tau_log16": dict(taus=[1, 2, 4, 8, 16]),
    "tau_5_10_15_20": dict(taus=[5, 10, 15, 20]),
    # sparsite par tau (famille B) : base tau=1..10
    "sparse_p02":   dict(taus=list(range(1, 11)), p=lambda t: 0.2, note="20% partout"),
    "dense_court_creux_long": dict(taus=list(range(1, 11)), p=lambda t: 1.0 if t <= 3 else 0.1),
    "creux_court_dense_long": dict(taus=list(range(1, 11)), p=lambda t: 0.1 if t <= 3 else 1.0),
    # profils s(tau) (famille D) : base tau=1..10
    "s_exp_t03": dict(taus=list(range(1, 11)), s=lambda t: float(np.exp(-t / 3.0))),
    "s_exp_t05": dict(taus=list(range(1, 11)), s=lambda t: float(np.exp(-t / 5.0))),
    "s_pow_a1":  dict(taus=list(range(1, 11)), s=lambda t: 1.0 / t),
}

def run_variant(task, vkey, v):
    # 14.2 : le meilleur rho par seed est choisi sur la validation, jamais sur le
    # test ; les cellules marquees "proto" sont reprises, les autres recalculees.
    if task in results and vkey in results[task] and \
       results[task][vkey].get("proto") == "rho_sur_validation": return None
    taus = v["taus"]; p = v.get("p"); s = v.get("s")
    scores = []
    if task == "mnist":
        Xtr, ytr, Xte, yte = mnist_data()
        for seed in range(3):
            best_val, best_test = -1.0, 0.0
            for rho in (0.9, 0.99):
                Ftr = states_batch(Xtr, 200, taus, rho, seed, p, s)
                Fte = states_batch(Xte, 200, taus, rho, seed, p, s)
                va = classify_acc(Ftr[:-NVAL_MNIST], ytr[:-NVAL_MNIST],
                                  Ftr[-NVAL_MNIST:], ytr[-NVAL_MNIST:])
                if va > best_val:
                    best_val = va
                    best_test = classify_acc(Ftr, ytr, Fte, yte)
            scores.append(best_test)
        scores = np.array(scores)
        return {"mean": float(scores.mean()), "std": float(scores.std()),
                "per_seed": scores.tolist(), "metric": "acc",
                "proto": "rho_sur_validation"}
    # taches temporelles
    if task == "narma30":
        gen = lambda sd: narma(30, sd)
        wash, ntest, radii, seeds = 200, 2000, (0.5, 0.9, 0.99), range(3)
    else:
        gen = lambda sd: _load_phys(task)
        wash, ntest, radii, seeds = 500, 10000, (0.5, 0.9), range(2)
    nval = NVAL[task]
    for seed in seeds:
        u, y = gen(seed)
        best_val, best_test = np.inf, np.inf
        per_rho = {}
        for rho in radii:
            F = states_seq(u, 200, taus, rho, seed, p, s)[:len(y)]
            val, test = nrmse_val_test(F, y, wash, ntest, nval)
            per_rho[str(rho)] = {"val": val, "test": test}
            if val < best_val:
                best_val, best_test = val, test
        scores.append(best_test)
    scores = np.array(scores)
    return {"mean": float(scores.mean()), "std": float(scores.std()),
            "per_seed": scores.tolist(), "metric": "nrmse", "note": v.get("note", ""),
            "per_rho": per_rho, "proto": "rho_sur_validation"}

def main():
    task = sys.argv[1]; only = sys.argv[2:] or list(VARIANTS)
    if task not in results: results[task] = {}
    for vkey in only:
        if vkey not in VARIANTS: print("unknown", vkey); continue
        r = run_variant(task, vkey, VARIANTS[vkey])
        if r is None:
            print(f"  {vkey:22s} [skip]"); continue
        results[task][vkey] = r
        json.dump(results, open(RJ, "w"), indent=2, ensure_ascii=False)
        m = r["mean"]
        print(f"  {vkey:22s} = {m:.4f} +/- {r['std']:.4f}", flush=True)

if __name__ == "__main__":
    main()
