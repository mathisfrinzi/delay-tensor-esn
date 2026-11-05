"""Partie 2.7 : factorisation CP / rang faible par tranche sur les ensembles
elus du pruning (2.2) : chaque W_tau = A_tau @ B_tau.T (rang r), rescale au
rayon effectif rho/sqrt(K) : protocole d'evaluation EXACTEMENT celui de
eval_pruned.py (N=sqrt(400k/K), splits standards), meilleur rho par seed CHOISI
SUR LA VALIDATION (NVAL[task] pas avant le test ; mnist : 1000 dernieres images du
train ; correction 14.2 du 2026-10-29), readout final reajuste sur train+validation.
Les seeds stockees en dict ("test"/"per_rho") sont reprises, les anciennes recalculees.
Rangs : r in {N/4, N/2, N} : a r=N/2 la tranche n'a plus que 2Nr = N^2/2
connexions : la moitie du budget recurrent. Question : les gains du pruning
({1,10} x7,6 sur NARMA, {4,7} record WH) survivent-ils a la factorisation ?
Reference plein rang gaussien = results_prune_final.json.
Checkpoint : results_cprank.json."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from chemins import BRUT  # noqa: E402
import numpy as np, json, time, sys, gzip, os
from pathlib import Path
from scipy.io import loadmat

OUT = (BRUT / "tensor-suite/part2")
RJ = OUT / "results_cprank.json"
results = json.load(open(RJ)) if RJ.exists() else {}
BUDGET, SEEDS = 400_000, [0, 1, 2]

ELECTED = {
    "narma30":   [1, 10],
    "silverbox": [2, 3, 4, 5, 7, 8, 10],
    "wh":        [4, 7],
    "mnist":     [2, 5, 7, 10],
}

def narma30(seed, T=5200):
    rng = np.random.default_rng(seed)
    u = rng.uniform(0, 0.5, size=T)
    y = np.zeros(T, np.float32)
    a, b, g, d = 0.2, 0.004, 1.5, 0.001
    n = 30
    for t in range(n, T):
        y[t] = a * y[t-1] + b * y[t-1] * y[t-n:t].sum() + g * u[t-n] * u[t-1] + d
    return np.stack([u, np.ones_like(u)], 1).astype(np.float32), y

def load_phys(which):
    if which == "silverbox":
        d = loadmat("/tmp/datasets/silverbox/SilverboxFiles/SNLS80mV.mat")
        u = d["V1"].ravel().astype(np.float64); y = d["V2"].ravel().astype(np.float64)
    else:
        d = loadmat("/tmp/datasets/WH.mat")
        u = d["uBenchMark"].ravel().astype(np.float64); y = d["yBenchMark"].ravel().astype(np.float64)
    s = np.std(y[:100000])
    U = u[:30000]; Y = y[:30000] / s
    return np.stack([U / np.std(U), np.ones_like(U)], 1).astype(np.float32), Y

def get_mnist():
    BASE = "/tmp/mnist"
    files = ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
             "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]
    def imgs(fn):
        with gzip.open(os.path.join(BASE, fn), "rb") as f:
            dd = np.frombuffer(f.read(), np.uint8, offset=16)
        return dd.reshape(-1, 28, 28).astype(np.float32) / 255.0
    def labs(fn):
        with gzip.open(os.path.join(BASE, fn), "rb") as f:
            return np.frombuffer(f.read(), np.uint8, offset=8).astype(np.int64)
    return imgs(files[0])[:5000], labs(files[1])[:5000], imgs(files[2])[:1000], labs(files[3])[:1000]

def make_W_rank(N, taus, rho, seed, r):
    """Tranches W_tau = A B^T de rang r, rayon effectif rho/sqrt(K)."""
    rng = np.random.default_rng(seed + 7)
    W = np.empty((len(taus), N, N), np.float32)
    for i in range(len(taus)):
        A = rng.normal(0, 1, (N, r)).astype(np.float32)
        B = rng.normal(0, 1, (N, r)).astype(np.float32)
        M = (A @ B.T).astype(np.float32)
        rad = max(np.abs(np.linalg.eigvals(M.astype(np.float64))).max(), 1e-12)
        W[i] = M * ((rho / np.sqrt(len(taus))) / rad)
    return W

def states_seq(u, N, taus, W, seed):
    rng = np.random.default_rng(seed)
    T, d = u.shape
    Win = rng.uniform(-0.5, 0.5, size=(d, N)).astype(np.float32)
    tmax = max(taus)
    hist = np.zeros((tmax, N), np.float32)
    x_ = np.zeros(N, np.float32)
    F = np.empty((T, N), np.float32)
    sel = np.array(taus) - 1
    for t in range(T):
        x_ = np.tanh(u[t] @ Win + np.einsum("kij,kj->i", W, hist[sel]))
        hist = np.concatenate([x_[None], hist[:-1]], axis=0)
        F[t] = x_
    return F

def states_mnist(Xb, N, taus, W, seed, chunk=1000):
    rng = np.random.default_rng(seed)
    Win = rng.uniform(-0.5, 0.5, size=(28, N)).astype(np.float32)
    tmax = max(taus)
    sel = np.array(taus) - 1
    out = np.empty((len(Xb), N), np.float32)
    for lo in range(0, len(Xb), chunk):
        B = Xb[lo:lo+chunk]
        x = np.zeros((len(B), N), np.float32)
        hist = np.zeros((tmax, len(B), N), np.float32)
        for t in range(28):
            drive = B[:, t, :] @ Win + np.tensordot(hist[sel], W, axes=([0, 2], [0, 2]))
            x = np.tanh(drive)
            hist = np.concatenate([x[None], hist[:-1]], axis=0)
        out[lo:lo+len(B)] = x
    return out

RADII = {"narma30": [0.5, 0.9, 0.99], "silverbox": [0.5, 0.9], "wh": [0.5, 0.9], "mnist": [0.9, 0.99]}
PROTO = {"narma30": (200, 2000), "silverbox": (500, 10000), "wh": (500, 10000)}
NVAL = {"narma30": 1000, "silverbox": 2000, "wh": 2000}   # validation : choix de rho (14.2)
NVAL_MNIST = 1000

task = sys.argv[1]
taus = ELECTED[task]
N = int(round((BUDGET / len(taus)) ** 0.5))
RANKS = sorted({max(2, N // 4), N // 2, N})
MAXIM = task == "mnist"
t0 = time.time()

if task == "mnist":
    Xtr, ytr, Xte, yte = get_mnist()
else:
    u, y = (narma30(0) if task == "narma30" else load_phys(task))
    wash, ntest = PROTO[task]

cell = results.setdefault(task, {"taus": taus, "N": N, "ranks": {}})
for r in RANKS:
    rc = cell["ranks"].setdefault(str(r), {"seeds": {}})
    for s in SEEDS:
        if str(s) in rc["seeds"] and isinstance(rc["seeds"][str(s)], dict):
            continue  # deja calculee sous le protocole corrige (14.2)
        per_rho = {}
        best_rho = None
        if task == "mnist":
            best_val = -1.0
            for rho in RADII[task]:
                W = make_W_rank(N, taus, rho, s, r)
                Ftr = states_mnist(Xtr, N, taus, W, s)
                Ftrb = np.hstack([Ftr, np.ones((len(Ftr), 1), np.float32)]).astype(np.float64)
                nfit = len(ytr) - NVAL_MNIST
                Tt = np.zeros((nfit, 10)); Tt[np.arange(nfit), ytr[:-NVAL_MNIST]] = 1.0
                A = Ftrb[:-NVAL_MNIST].T @ Ftrb[:-NVAL_MNIST] + 1e-6 * np.eye(Ftrb.shape[1])
                Wout = np.linalg.solve(A, Ftrb[:-NVAL_MNIST].T @ Tt)
                va = float((np.argmax(Ftrb[-NVAL_MNIST:] @ Wout, axis=1) == ytr[-NVAL_MNIST:]).mean())
                per_rho[str(rho)] = {"val": va}
                if va > best_val:
                    best_val, best_rho = va, rho
            # readout final reajuste sur tout le train avec le rho elu, test officiel
            W = make_W_rank(N, taus, best_rho, s, r)
            Ftr = states_mnist(Xtr, N, taus, W, s)
            Fte = states_mnist(Xte, N, taus, W, s)
            Ftrb = np.hstack([Ftr, np.ones((len(Ftr), 1), np.float32)]).astype(np.float64)
            Fteb = np.hstack([Fte, np.ones((len(Fte), 1), np.float32)]).astype(np.float64)
            Tt = np.zeros((len(ytr), 10)); Tt[np.arange(len(ytr)), ytr] = 1.0
            A = Ftrb.T @ Ftrb + 1e-6 * np.eye(Ftrb.shape[1])
            Wout = np.linalg.solve(A, Ftrb.T @ Tt)
            best = float((np.argmax(Fteb @ Wout, axis=1) == yte).mean())
        else:
            us, ys = (narma30(s) if task == "narma30" else (u, y))
            nval = NVAL[task]
            best_val = np.inf
            for rho in RADII[task]:
                W = make_W_rank(N, taus, rho, s, r)
                F = states_seq(us, N, taus, W, s)
                Fb = np.hstack([F, np.ones((len(F), 1), np.float32)]).astype(np.float64)
                yv = np.asarray(ys, np.float64)
                Ftr_, ytr_ = Fb[wash:-ntest-nval], yv[wash:-ntest-nval]
                Fva_, yva_ = Fb[-ntest-nval:-ntest], yv[-ntest-nval:-ntest]
                A = Ftr_.T @ Ftr_ + 1e-8 * np.eye(Ftr_.shape[1])
                Wout = np.linalg.solve(A, Ftr_.T @ ytr_)
                va = float(np.sqrt(np.mean((Fva_ @ Wout - yva_) ** 2)) / np.std(yva_))
                per_rho[str(rho)] = {"val": va}
                if va < best_val:
                    best_val, best_rho = va, rho
            # readout final reajuste sur train+validation avec le rho elu, puis test
            W = make_W_rank(N, taus, best_rho, s, r)
            F = states_seq(us, N, taus, W, s)
            Fb = np.hstack([F, np.ones((len(F), 1), np.float32)]).astype(np.float64)
            yv = np.asarray(ys, np.float64)
            Ftv_, ytv_ = Fb[wash:-ntest], yv[wash:-ntest]
            Fte_, yte_ = Fb[-ntest:], yv[-ntest:]
            A = Ftv_.T @ Ftv_ + 1e-8 * np.eye(Ftv_.shape[1])
            Wout = np.linalg.solve(A, Ftv_.T @ ytv_)
            best = float(np.sqrt(np.mean((Fte_ @ Wout - yte_) ** 2)) / np.std(yte_))
        rc["seeds"][str(s)] = {"rho": best_rho, "test": best, "per_rho": per_rho}
        json.dump(results, open(RJ, "w"), indent=1)
        print(f"{task} r={r} seed{s}: test={best:.4f} ({time.time()-t0:.0f}s)", flush=True)

for r in RANKS:
    v = [cell["ranks"][str(r)]["seeds"][str(s)]["test"] if isinstance(cell["ranks"][str(r)]["seeds"][str(s)], dict)
         else cell["ranks"][str(r)]["seeds"][str(s)] for s in SEEDS]
    print(f"{task} r={r}/{N}: mean={np.mean(v):.4f} +/- {np.std(v):.4f}")
print(f"{task} 2.7 CP-rang terminé")
