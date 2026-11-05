"""1.3ter : quelles couches comptent ? Sweep des ensembles tau = {k0..10}, k0=1..10.
Budget constant N^2*K = 400k (N=sqrt(400k/K)). Si seul le NOMBRE de couches comptait,
les configurations de meme cardinal (ex. {2..10} vs {1..9}) seraient egales ; si les
couches precoces comptent, retirer tau=1..k0 doit couter plus que retirer des tardives.
Checkpoint : part2b/results_narma_slices.json - relancer jusqu'a COMPLETE."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from chemins import BRUT  # noqa: E402
import numpy as np, json, time
from pathlib import Path

OUT = (BRUT / "tensor-suite/part2b")
RJ = OUT / "results_narma_slices.json"
results = json.load(open(RJ)) if RJ.exists() else {"cells": {}}

BUDGET, RHO = 400_000, 0.9
SEEDS = [0, 1, 2]
ORDERS = [5, 10, 15, 20, 25, 30, 35, 40]
CONFIGS = {f"k{k0}": list(range(k0, 11)) for k0 in range(1, 11)}  # {1..10}, {2..10}, ..., {10}
T, WASH, NTEST = 5200, 200, 1000
TIME_BUDGET = 230

def narma_fam(n, seed, T=T):
    rng = np.random.default_rng(seed)
    u = rng.uniform(0, 0.5, size=T)
    y = np.zeros(T, np.float32)
    a, b, g, d = 0.2, 0.004, 1.5, 0.001
    for t in range(n, T):
        y[t] = a * y[t-1] + b * y[t-1] * y[t-n:t].sum() + g * u[t-n] * u[t-1] + d
    return np.stack([u, np.ones_like(u)], 1).astype(np.float32), y

def run_one(x, taus, seed):
    K = len(taus)
    N = int(round((BUDGET / K) ** 0.5))
    rng = np.random.default_rng(seed)
    Win = rng.uniform(-0.5, 0.5, size=(2, N)).astype(np.float32)
    W = np.empty((K, N, N), np.float32)
    for i in range(K):
        M = rng.normal(0, 1, (N, N)).astype(np.float32)
        r = max(np.abs(np.linalg.eigvals(M.astype(np.float64))).max(), 1e-12)
        W[i] = M * ((RHO / np.sqrt(K)) / r)
    u, y = narma_fam(x, seed + 100)
    hist = np.zeros((max(taus), N), np.float32)
    x_ = np.zeros(N, np.float32)
    F = np.empty((T, N), np.float32)
    sel = np.array(taus) - 1
    for t in range(T):
        x_ = np.tanh(u[t] @ Win + np.einsum("kij,kj->i", W, hist[sel]))
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
stop = False
for name, taus in CONFIGS.items():
    if stop: break
    for x in ORDERS:
        if stop: break
        key = f"{name}_x{x}"
        cell = results["cells"].setdefault(key, {"config": name, "taus": taus, "x": x, "seeds": {}})
        for s in SEEDS:
            if str(s) in cell["seeds"]:
                continue
            if time.time() - t0 > TIME_BUDGET:
                stop = True; break
            nrmse, N = run_one(x, taus, s)
            cell["seeds"][str(s)] = nrmse
            cell["N"] = N
            print(f"{name}(K={len(taus)}) x={x} seed={s} NRMSE={nrmse:.4f} ({time.time()-t0:.0f}s)", flush=True)
        results["cells"][key] = cell
        json.dump(results, open(RJ, "w"), indent=1)

remaining = sum(1 for name in CONFIGS for x in ORDERS
                if any(str(s) not in results["cells"].get(f"{name}_x{x}", {"seeds": {}})["seeds"] for s in SEEDS))
print(f"restants : {remaining}")
if remaining == 0:
    print("COMPLETE")
    agg = {}
    for key, c in results["cells"].items():
        v = np.array(list(c["seeds"].values()))
        agg[key] = dict(config=c["config"], taus=c["taus"], K=len(c["taus"]), x=c["x"], N=c["N"],
                        mean=float(v.mean()), std=float(v.std()), n=len(v))
    results["agg"] = agg
    json.dump(results, open(RJ, "w"), indent=1)
    print("agregation ecrite")
