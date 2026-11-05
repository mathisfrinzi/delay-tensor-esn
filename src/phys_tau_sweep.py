"""1.3quater (a) : replique du sweep 1.3ter sur les systemes physiques (Silverbox, WH).
Ensembles tau = {k0..10}, k0=1..10, budget constant N^2*K = 400k, meilleur rho de {0.5, 0.9}
par seed (protocole partie 2), 3 seeds, train 20k / wash 500 / test 10k (meme decoupe que la
campagne 2.6 -> directement comparable). Prediction tiree de 2.4/2.6 : sur ces taches l'alpha
appris prunait tau=1 (alpha_1=0) -> retirer tau=1 ne doit PAS faire s'effondrer la performance,
contrairement a NARMA. Checkpoint : part2b/results_phys_slices.json - relancer jusqu'a COMPLETE.
Protocole corrige (14.2, 2026-10-29) : le meilleur rho par seed est choisi sur un segment
de VALIDATION (NVAL = 2000 pas avant le test), jamais sur le test ; readout final reajuste
sur train+validation. Les seeds stockees en dict ("test"/"per_rho") sont reprises, les
anciennes (floats, choix sur le test) recalculees."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from chemins import BRUT  # noqa: E402
import numpy as np, json, time, sys
from pathlib import Path
from scipy.io import loadmat

OUT = (BRUT / "tensor-suite/part2b")
RJ = OUT / "results_phys_slices.json"
results = json.load(open(RJ)) if RJ.exists() else {"cells": {}}

BUDGET = 400_000
SEEDS = [0, 1, 2]
RADII = (0.5, 0.9)
CONFIGS = {f"k{k0}": list(range(k0, 11)) for k0 in range(1, 11)}  # {1..10}, {2..10}, ..., {10}
WASH, NTEST, NTRAIN = 500, 10000, 20000
NVAL = 2000  # segment de validation avant le test : le rho y est choisi (14.2)
TIME_BUDGET = 220

def load_phys(which):
    if which == "silverbox":
        d = loadmat("/tmp/datasets/silverbox/SilverboxFiles/SNLS80mV.mat")
        u = d["V1"].ravel().astype(np.float64); y = d["V2"].ravel().astype(np.float64)
    else:
        d = loadmat("/tmp/datasets/WH.mat")
        u = d["uBenchMark"].ravel().astype(np.float64); y = d["yBenchMark"].ravel().astype(np.float64)
    s = np.std(y[:100000])
    U = u[:NTRAIN + NTEST]; Y = y[:NTRAIN + NTEST] / s
    uu = np.stack([U / np.std(U), np.ones_like(U)], 1).astype(np.float32)
    return uu, Y

def run_one(u, y, taus, rho, seed):
    K = len(taus)
    N = int(round((BUDGET / K) ** 0.5))
    rng = np.random.default_rng(seed)
    Win = rng.uniform(-0.5, 0.5, size=(2, N)).astype(np.float32)
    W = np.empty((K, N, N), np.float32)
    for i in range(K):
        M = rng.normal(0, 1, (N, N)).astype(np.float32)
        r = max(np.abs(np.linalg.eigvals(M.astype(np.float64))).max(), 1e-12)
        W[i] = M * ((rho / np.sqrt(K)) / r)
    T = len(u); tmax = max(taus)
    hist = np.zeros((tmax, N), np.float32)
    x_ = np.zeros(N, np.float32)
    F = np.empty((T, N), np.float32)
    sel = np.array(taus) - 1
    for t in range(T):
        x_ = np.tanh(u[t] @ Win + np.einsum("kij,kj->i", W, hist[sel]))
        hist = np.concatenate([x_[None], hist[:-1]], axis=0)
        F[t] = x_
    Fb = np.hstack([F, np.ones((T, 1), np.float32)]).astype(np.float64)
    yv = np.asarray(y, np.float64)
    # 14.2 : renvoie (validation, test) ; le readout du test est reajuste sur
    # train+validation, apres le choix de rho sur la validation.
    Ftr, ytr = Fb[WASH:-NTEST-NVAL], yv[WASH:-NTEST-NVAL]
    Fva, yva = Fb[-NTEST-NVAL:-NTEST], yv[-NTEST-NVAL:-NTEST]
    Fte, yte = Fb[-NTEST:], yv[-NTEST:]
    A = Ftr.T @ Ftr + 1e-8 * np.eye(Ftr.shape[1])
    Wout = np.linalg.solve(A, Ftr.T @ ytr)
    val = float(np.sqrt(np.mean((Fva @ Wout - yva) ** 2)) / np.std(yva))
    Ftv, ytv = Fb[WASH:-NTEST], yv[WASH:-NTEST]
    A2 = Ftv.T @ Ftv + 1e-8 * np.eye(Ftv.shape[1])
    Wout2 = np.linalg.solve(A2, Ftv.T @ ytv)
    return val, float(np.sqrt(np.mean((Fte @ Wout2 - yte) ** 2)) / np.std(yte)), N

task = sys.argv[1]
u, y = load_phys(task)
t0 = time.time()
stop = False
for name, taus in CONFIGS.items():
    if stop: break
    key = f"{task}_{name}"
    cell = results["cells"].setdefault(key, {"task": task, "config": name, "taus": taus, "seeds": {}})
    for s in SEEDS:
        if str(s) in cell["seeds"] and isinstance(cell["seeds"][str(s)], dict):
            continue  # deja calculee sous le protocole corrige (14.2)
        if time.time() - t0 > TIME_BUDGET:
            stop = True; break
        best_val, best, N = np.inf, np.inf, None
        per_rho = {}
        for rho in RADII:
            va, te, N = run_one(u, y, taus, rho, s)
            per_rho[str(rho)] = {"val": va, "test": te}
            if va < best_val:
                best_val, best = va, te
        cell["seeds"][str(s)] = {"test": best, "per_rho": per_rho}
        cell["N"] = N
        print(f"{task} {name}(K={len(taus)}) seed={s} NRMSE={best:.4f} ({time.time()-t0:.0f}s)", flush=True)
        results["cells"][key] = cell
        json.dump(results, open(RJ, "w"), indent=1)

remaining = sum(1 for t2 in ("silverbox", "wh") for name in CONFIGS
               if any(str(s) not in results["cells"].get(f"{t2}_{name}", {"seeds": {}})["seeds"]
                      or not isinstance(results["cells"].get(f"{t2}_{name}", {"seeds": {}})["seeds"][str(s)], dict)
                      for s in SEEDS))
print(f"restants : {remaining}")
if remaining == 0:
    print("COMPLETE")
    agg = {}
    for key, c in results["cells"].items():
        v = np.array([sv["test"] if isinstance(sv, dict) else sv for sv in c["seeds"].values()])
        agg[key] = dict(task=c["task"], config=c["config"], taus=c["taus"], K=len(c["taus"]),
                       N=c["N"], mean=float(v.mean()), std=float(v.std()), n=len(v))
    results["agg"] = agg
    json.dump(results, open(RJ, "w"), indent=1)
    print("agregation ecrite")
