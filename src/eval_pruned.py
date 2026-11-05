"""Evaluation finale (protocole standard) des ensembles elus par le pruning 2.2 :
narma30 {1,10} / silverbox {2,3,4,5,7,8,10} / wh {4,7} / mnist {2,5,7,10}, budget egal 400k.
Protocole standard : narma30 wash 200 / test 2000, rho {0.5,0.9,0.99} ; silverbox/wh wash 500 /
test 10000, rho {0.5,0.9} ; mnist 5k/1k, rho {0.9,0.99}. 3 seeds, meilleur rho par seed.
Protocole corrige (14.2, 2026-10-29) : le meilleur rho par seed est choisi sur un
segment de VALIDATION (NVAL[task] pas avant le test ; mnist : 1000 dernieres images
du train), jamais sur le test ; readout final reajuste sur train+validation. Les
seeds stockees en dict ("test"/"per_rho") sont reprises, les anciennes recalculees.
Checkpoint : part2/results_prune_final.json."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from chemins import BRUT  # noqa: E402
import numpy as np, json, time, sys, gzip, os
from pathlib import Path
from scipy.io import loadmat

OUT = (BRUT / "tensor-suite/part2")
RJ = OUT / "results_prune_final.json"
results = json.load(open(RJ)) if RJ.exists() else {"cells": {}}
BUDGET, SEEDS = 400_000, [0, 1, 2]
TIME_BUDGET = float(__import__("os").environ.get("TEMPS_MAX", "215"))

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
    return imgs(files[0])[:5000], labs(files[1])[:5000], imgs(files[2])[:1000], labs(files[3])[:1000]

def states_seq(u, N, taus, rho, seed):
    rng = np.random.default_rng(seed + 7)
    T, d = u.shape
    Win = rng.uniform(-0.5, 0.5, size=(d, N)).astype(np.float32)
    W = np.empty((len(taus), N, N), np.float32)
    for i in range(len(taus)):
        M = rng.normal(0, 1, (N, N)).astype(np.float32)
        r = max(np.abs(np.linalg.eigvals(M.astype(np.float64))).max(), 1e-12)
        W[i] = M * ((rho / np.sqrt(len(taus))) / r)
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

def states_mnist(Xb, N, taus, rho, seed, chunk=1000):
    rng = np.random.default_rng(seed)
    Win = rng.uniform(-0.5, 0.5, size=(28, N)).astype(np.float32)
    W = np.empty((len(taus), N, N), np.float32)
    for i in range(len(taus)):
        M = rng.normal(0, 1, (N, N)).astype(np.float32)
        r = max(np.abs(np.linalg.eigvals(M.astype(np.float64))).max(), 1e-12)
        W[i] = M * ((rho / np.sqrt(len(taus))) / r)
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
if task == "mnist":
    Xtr, ytr, Xte, yte = get_mnist()
t0 = time.time()
cell = results["cells"].setdefault(task, {"taus": taus, "N": N, "seeds": {}})
for s in SEEDS:
    if str(s) in cell["seeds"] and isinstance(cell["seeds"][str(s)], dict):
        continue  # deja calculee sous le protocole corrige (14.2)
    per_rho = {}
    best_rho = None
    if task == "mnist":
        best_val = -1.0
        for rho in RADII[task]:
            Ftr = states_mnist(Xtr, N, taus, rho, s)
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
        Ftr = states_mnist(Xtr, N, taus, best_rho, s)
        Fte = states_mnist(Xte, N, taus, best_rho, s)
        Ftrb = np.hstack([Ftr, np.ones((len(Ftr), 1), np.float32)]).astype(np.float64)
        Fteb = np.hstack([Fte, np.ones((len(Fte), 1), np.float32)]).astype(np.float64)
        Tt = np.zeros((len(ytr), 10)); Tt[np.arange(len(ytr)), ytr] = 1.0
        A = Ftrb.T @ Ftrb + 1e-6 * np.eye(Ftrb.shape[1])
        Wout = np.linalg.solve(A, Ftrb.T @ Tt)
        best = float((np.argmax(Fteb @ Wout, axis=1) == yte).mean())
    else:
        u, y = narma30(s) if task == "narma30" else load_phys(task)
        wash, ntest = PROTO[task]
        nval = NVAL[task]
        best_val = np.inf
        for rho in RADII[task]:
            F = states_seq(u, N, taus, rho, s)
            Fb = np.hstack([F, np.ones((len(F), 1), np.float32)]).astype(np.float64)
            yv = np.asarray(y, np.float64)
            Ftr, ytr2 = Fb[wash:-ntest-nval], yv[wash:-ntest-nval]
            Fva, yva2 = Fb[-ntest-nval:-ntest], yv[-ntest-nval:-ntest]
            A = Ftr.T @ Ftr + 1e-8 * np.eye(Ftr.shape[1])
            Wout = np.linalg.solve(A, Ftr.T @ ytr2)
            va = float(np.sqrt(np.mean((Fva @ Wout - yva2) ** 2)) / np.std(yva2))
            per_rho[str(rho)] = {"val": va}
            if va < best_val:
                best_val, best_rho = va, rho
        # readout final reajuste sur train+validation avec le rho elu, puis test
        F = states_seq(u, N, taus, best_rho, s)
        Fb = np.hstack([F, np.ones((len(F), 1), np.float32)]).astype(np.float64)
        yv = np.asarray(y, np.float64)
        Ftv, ytv2 = Fb[wash:-ntest], yv[wash:-ntest]
        Fte, yte2 = Fb[-ntest:], yv[-ntest:]
        A = Ftv.T @ Ftv + 1e-8 * np.eye(Ftv.shape[1])
        Wout = np.linalg.solve(A, Ftv.T @ ytv2)
        best = float(np.sqrt(np.mean((Fte @ Wout - yte2) ** 2)) / np.std(yte2))
    cell["seeds"][str(s)] = {"rho": best_rho, "test": best, "per_rho": per_rho}
    print(f"{task} {taus} N={N} seed={s} {best:.4f} ({time.time()-t0:.0f}s)", flush=True)
    results["cells"][task] = cell
    json.dump(results, open(RJ, "w"), indent=1)

ndone = sum(1 for s in SEEDS if isinstance(cell["seeds"].get(str(s)), dict))
print("COMPLETE" if ndone == len(SEEDS) else "pas encore")
