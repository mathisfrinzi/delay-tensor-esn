"""Addendum : lignes supplementaires de la famille NARMA-30 (params uniformes 0.2/0.004/1.5/0.001).
x=10f (meme famille), x=35, x=40. Meme protocole budget 400k, 3 seeds."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from chemins import BRUT  # noqa: E402
import numpy as np, json, time
from pathlib import Path

OUT = (BRUT / "tensor-suite/part2b")
RJ = OUT / "results_narma_sweep.json"
results = json.load(open(RJ))
BUDGET, RHO, T, WASH, NTEST = 400_000, 0.9, 5200, 200, 1000
SEEDS = [0, 1, 2]
KS = list(range(1, 11))
EXTRA = {"x10f": 10, "x35": 35, "x40": 40}  # tous params famille

def narma_fam(n, seed, T=T):
    rng = np.random.default_rng(seed)
    u = rng.uniform(0, 0.5, size=T)
    y = np.zeros(T, np.float32)
    a, b, g, d = 0.2, 0.004, 1.5, 0.001
    for t in range(n, T):
        y[t] = a * y[t-1] + b * y[t-1] * y[t-n:t].sum() + g * u[t-n] * u[t-1] + d
    return np.stack([u, np.ones_like(u)], 1).astype(np.float32), y

def run_one(x, K, seed):
    N = int(round((BUDGET / K) ** 0.5))
    rng = np.random.default_rng(seed)
    Win = rng.uniform(-0.5, 0.5, size=(2, N)).astype(np.float32)
    W = np.empty((K, N, N), np.float32)
    for i in range(K):
        M = rng.normal(0, 1, (N, N)).astype(np.float32)
        r = max(np.abs(np.linalg.eigvals(M.astype(np.float64))).max(), 1e-12)
        W[i] = M * ((RHO / np.sqrt(K)) / r)
    u, y = narma_fam(x, seed + 100)
    hist = np.zeros((K, N), np.float32)
    x_ = np.zeros(N, np.float32)
    F = np.empty((T, N), np.float32)
    for t in range(T):
        x_ = np.tanh(u[t] @ Win + np.einsum("kij,kj->i", W, hist))
        hist = np.concatenate([x_[None], hist[:-1]], axis=0)
        F[t] = x_
    Fb = np.hstack([F, np.ones((T, 1), np.float32)]).astype(np.float64)
    yv = y.astype(np.float64)
    Ftr, ytr = Fb[WASH:-NTEST], yv[WASH:-NTEST]
    Fte, yte = Fb[-NTEST:], yv[-NTEST:]
    A = Ftr.T @ Ftr + 1e-8 * np.eye(Ftr.shape[1])
    Wout = np.linalg.solve(A, Ftr.T @ ytr)
    return float(np.sqrt(np.mean((Fte @ Wout - yte) ** 2)) / np.std(yte)), N

t0 = time.time()
for name, x in EXTRA.items():
    for K in KS:
        key = f"{name}_K{K}"
        cell = results["cells"].setdefault(key, {"x": x, "K": K, "seeds": {}, "name": name})
        for s in SEEDS:
            if str(s) in cell["seeds"]:
                continue
            nrmse, N = run_one(x, K, s)
            cell["seeds"][str(s)] = nrmse
            cell["N"] = N
            print(f"{name} K={K} seed={s} NRMSE={nrmse:.4f} ({time.time()-t0:.0f}s)", flush=True)
        results["cells"][key] = cell
        json.dump(results, open(RJ, "w"), indent=1)

# reagreger
agg = results.get("agg", {})
for key, c in results["cells"].items():
    v = np.array(list(c["seeds"].values()))
    kk = c["K"]
    k2 = f"{c.get('name','x%d'%c['x'])}_K{kk}"
    agg[k2] = dict(x=c["x"], K=kk, N=c["N"], name=c.get("name", f"x{c['x']}"),
                   mean=float(v.mean()), std=float(v.std()), n=len(v))
results["agg"] = agg
json.dump(results, open(RJ, "w"), indent=1)
print("COMPLETE2")
