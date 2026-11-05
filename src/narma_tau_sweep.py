"""Preuve de la partie 1.3 : NARMA-x en fonction de l'ordre x et de la profondeur tau.
Ensembles tau = {1..K}, K=1..10, budget de connexions CONSTANT N^2*K = 400 000
(K=1 -> ESN standard N=632, ... K=10 -> tenseur N=200 x 10 tranches, la config Partie 1).
Prediction de la loi 1.3 : plus x est grand, plus il faut de tranches (grand K) ;
pour x petit, K=1 (ESN) doit suffire ou gagner.
Checkpoint : results_narma_sweep.json, re-executer jusqu'a "COMPLETE".
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from chemins import BRUT  # noqa: E402
import numpy as np, json, time
from pathlib import Path

OUT = (BRUT / "tensor-suite/part2b")
OUT.mkdir(parents=True, exist_ok=True)
RJ = OUT / "results_narma_sweep.json"
results = json.load(open(RJ)) if RJ.exists() else {"cells": {}}

BUDGET = 400_000
RHO = 0.9
SEEDS = [0, 1, 2]
ORDERS = [5, 10, 15, 20, 25, 30]
KS = list(range(1, 11))
T, WASH, NTEST = 5200, 200, 1000
TIME_BUDGET = 240  # s, redemarre proprement avant le timeout bash

def narma(n, seed, T=T):
    rng = np.random.default_rng(seed)
    u = rng.uniform(0, 0.5, size=T)
    y = np.zeros(T, np.float32)
    a, b, g, d = (0.3, 0.05, 1.5, 0.1) if n == 10 else (0.2, 0.004, 1.5, 0.001)
    for t in range(n, T):
        y[t] = a * y[t-1] + b * y[t-1] * y[t-n:t].sum() + g * u[t-n] * u[t-1] + d
    return np.stack([u, np.ones_like(u)], 1).astype(np.float32), y

def run_one(x, K, seed):
    taus = list(range(1, K + 1))
    N = int(round((BUDGET / K) ** 0.5))
    rng = np.random.default_rng(seed)
    Win = rng.uniform(-0.5, 0.5, size=(2, N)).astype(np.float32)
    # tranches : gaussien -> rayon spectral rho/sqrt(K)
    W = np.empty((K, N, N), np.float32)
    for i in range(K):
        M = rng.normal(0, 1, (N, N)).astype(np.float32)
        r = max(np.abs(np.linalg.eigvals(M.astype(np.float64))).max(), 1e-12)
        W[i] = M * ((RHO / np.sqrt(K)) / r)
    u, y = narma(x, seed + 100)
    hist = np.zeros((K, N), np.float32)
    x_ = np.zeros(N, np.float32)
    F = np.empty((T, N), np.float32)
    for t in range(T):
        drive = u[t] @ Win + np.einsum("kij,kj->i", W, hist)
        x_ = np.tanh(drive)
        hist = np.concatenate([x_[None], hist[:-1]], axis=0)
        F[t] = x_
    Fb = np.hstack([F, np.ones((T, 1), np.float32)]).astype(np.float64)
    yv = y.astype(np.float64)
    Ftr, ytr = Fb[WASH:-NTEST], yv[WASH:-NTEST]
    Fte, yte = Fb[-NTEST:], yv[-NTEST:]
    A = Ftr.T @ Ftr + 1e-8 * np.eye(Ftr.shape[1])
    Wout = np.linalg.solve(A, Ftr.T @ ytr)
    nrmse = float(np.sqrt(np.mean((Fte @ Wout - yte) ** 2)) / np.std(yte))
    return nrmse, N

t0 = time.time()
done, todo = 0, 0
for x in ORDERS:
    for K in KS:
        key = f"x{x}_K{K}"
        cell = results["cells"].get(key, {"x": x, "K": K, "seeds": {}})
        missing = [s for s in SEEDS if str(s) not in cell["seeds"]]
        if not missing:
            done += 1
            continue
        for s in missing:
            if time.time() - t0 > TIME_BUDGET:
                break
            nrmse, N = run_one(x, K, s)
            cell["seeds"][str(s)] = nrmse
            cell["N"] = N
            todo += 1
            print(f"x={x} K={K} N={N} seed={s} NRMSE={nrmse:.4f} ({time.time()-t0:.0f}s)", flush=True)
        results["cells"][key] = cell
        json.dump(results, open(RJ, "w"), indent=1)
        if time.time() - t0 > TIME_BUDGET:
            print("TIME BUDGET - checkpoint sauvegarde, relancer pour continuer")
            break
    else:
        continue
    break

# resume check
remaining = sum(1 for x in ORDERS for K in KS
               if any(str(s) not in results["cells"].get(f"x{x}_K{K}", {"seeds": {}})["seeds"] for s in SEEDS))
print(f"cellules restantes (seed-manquants) : {remaining}")
if remaining == 0:
    print("COMPLETE")
    agg = {}
    for x in ORDERS:
        for K in KS:
            c = results["cells"][f"x{x}_K{K}"]
            v = np.array(list(c["seeds"].values()))
            agg[f"x{x}_K{K}"] = dict(x=x, K=K, N=c["N"], mean=float(v.mean()),
                                     std=float(v.std()), n=len(v))
    results["agg"] = agg
    json.dump(results, open(RJ, "w"), indent=1)
    print("agregation ecrite")
