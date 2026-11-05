"""1.3quater (b) : replique du sweep 1.3ter sur MNIST ligne par ligne (protocole partie 2 :
5k train / 1k test, meilleur rho de {0.9, 0.99} par seed, 3 seeds, budget constant N^2*K=400k).
Ensembles tau = {k0..10}, k0=1..10. Sur MNIST seule l'etendue spatiale compte (cf 2.6 :
tau=10..19 tenait 82,2 %) -> retirer tau=1 ne devrait presque rien couter, contrairement a
NARMA. Checkpoint : part2b/results_mnist_slices.json - relancer jusqu'a COMPLETE.
Protocole corrige (14.2, 2026-10-29) : le meilleur rho par seed est choisi sur les 1000
dernieres images du train (validation), jamais sur le test ; readout final reajuste sur
tout le train. Les seeds stockees en dict ("test"/"per_rho") sont reprises, les anciennes
(floats, choix sur le test) recalculees."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from chemins import BRUT  # noqa: E402
import numpy as np, json, time, gzip, os
from pathlib import Path

OUT = (BRUT / "tensor-suite/part2b")
RJ = OUT / "results_mnist_slices.json"
results = json.load(open(RJ)) if RJ.exists() else {"cells": {}}

BUDGET = 400_000
SEEDS = [0, 1, 2]
RADII = (0.9, 0.99)
CONFIGS = {f"k{k0}": list(range(k0, 11)) for k0 in range(1, 11)}
NTRAIN, NTEST = 5000, 1000
NVAL_MNIST = 1000  # dernieres images du train : le rho y est choisi (14.2)
TIME_BUDGET = 220

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
    return imgs(files[0])[:NTRAIN], labs(files[1])[:NTRAIN], imgs(files[2])[:NTEST], labs(files[3])[:NTEST]

def states_batch(Xb, N, taus, rho, seed, chunk=1000):
    """Xb: (B, 28, 28) -> (B, N). Lecture ligne par ligne, 28 pas."""
    rng = np.random.default_rng(seed)
    Win = rng.uniform(-0.5, 0.5, size=(28, N)).astype(np.float32)
    K = len(taus)
    W = np.empty((K, N, N), np.float32)
    for i in range(K):
        M = rng.normal(0, 1, (N, N)).astype(np.float32)
        r = max(np.abs(np.linalg.eigvals(M.astype(np.float64))).max(), 1e-12)
        W[i] = M * ((rho / np.sqrt(K)) / r)
    tmax = max(taus)
    sel = np.array(taus) - 1
    out = np.empty((len(Xb), N), np.float32)
    for lo in range(0, len(Xb), chunk):
        B = Xb[lo:lo+chunk]
        x = np.zeros((len(B), N), np.float32)
        hist = np.zeros((tmax, len(B), N), np.float32)
        for t in range(28):
            drive = B[:, t, :] @ Win
            # (K,B,N) x (K,N,N) -> (B,N)
            drive = drive + np.tensordot(hist[sel], W, axes=([0, 2], [0, 2]))
            x = np.tanh(drive)
            hist = np.concatenate([x[None], hist[:-1]], axis=0)
        out[lo:lo+len(B)] = x
    return out

def classify_acc(Ftr, ytr, Fte, yte, ridge=1e-6):
    Ftr = np.hstack([Ftr, np.ones((len(Ftr), 1), np.float32)]).astype(np.float64)
    Fte = np.hstack([Fte, np.ones((len(Fte), 1), np.float32)]).astype(np.float64)
    T = np.zeros((len(ytr), 10)); T[np.arange(len(ytr)), ytr] = 1.0
    A = Ftr.T @ Ftr + ridge * np.eye(Ftr.shape[1])
    Wout = np.linalg.solve(A, Ftr.T @ T)
    return float((np.argmax(Fte @ Wout, axis=1) == yte).mean())

Xtr, ytr, Xte, yte = get_mnist()
t0 = time.time()
stop = False
for name, taus in CONFIGS.items():
    if stop: break
    key = f"mnist_{name}"
    cell = results["cells"].setdefault(key, {"config": name, "taus": taus, "seeds": {}})
    for s in SEEDS:
        if str(s) in cell["seeds"] and isinstance(cell["seeds"][str(s)], dict):
            continue  # deja calculee sous le protocole corrige (14.2)
        if time.time() - t0 > TIME_BUDGET:
            stop = True; break
        N = int(round((BUDGET / len(taus)) ** 0.5))
        best_val, best = -1.0, 0.0
        per_rho = {}
        for rho in RADII:
            Ftr = states_batch(Xtr, N, taus, rho, s)
            Fte = states_batch(Xte, N, taus, rho, s)
            # 14.2 : rho choisi sur les NVAL_MNIST dernieres images du train
            nfit = NTRAIN - NVAL_MNIST
            Ftrb = np.hstack([Ftr, np.ones((len(Ftr), 1), np.float32)]).astype(np.float64)
            Tt = np.zeros((nfit, 10)); Tt[np.arange(nfit), ytr[:-NVAL_MNIST]] = 1.0
            A = Ftrb[:-NVAL_MNIST].T @ Ftrb[:-NVAL_MNIST] + 1e-6 * np.eye(Ftrb.shape[1])
            Wout = np.linalg.solve(A, Ftrb[:-NVAL_MNIST].T @ Tt)
            va = float((np.argmax(Ftrb[-NVAL_MNIST:] @ Wout, axis=1) == ytr[-NVAL_MNIST:]).mean())
            per_rho[str(rho)] = {"val": va}
            if va > best_val:
                best_val = va
                best = classify_acc(Ftr, ytr, Fte, yte)
        cell["seeds"][str(s)] = {"test": best, "per_rho": per_rho}
        cell["N"] = N
        print(f"mnist {name}(K={len(taus)}) seed={s} acc={best:.4f} ({time.time()-t0:.0f}s)", flush=True)
        results["cells"][key] = cell
        json.dump(results, open(RJ, "w"), indent=1)

remaining = sum(1 for name in CONFIGS
               if any(str(s) not in results["cells"].get(f"mnist_{name}", {"seeds": {}})["seeds"]
                      or not isinstance(results["cells"].get(f"mnist_{name}", {"seeds": {}})["seeds"][str(s)], dict)
                      for s in SEEDS))
print(f"restants : {remaining}")
if remaining == 0:
    print("COMPLETE")
    agg = {}
    for key, c in results["cells"].items():
        v = np.array([sv["test"] if isinstance(sv, dict) else sv for sv in c["seeds"].values()])
        agg[key] = dict(config=c["config"], taus=c["taus"], K=len(c["taus"]), N=c["N"],
                       mean=float(v.mean()), std=float(v.std()), n=len(v))
    results["agg"] = agg
    json.dump(results, open(RJ, "w"), indent=1)
    print("agregation ecrite")
