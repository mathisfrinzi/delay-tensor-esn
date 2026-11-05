"""Carte de Henon : prediction a horizon 1 et 5. Suite Partie 1."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from chemins import BRUT  # noqa: E402
import numpy as np, json, sys
from suite_part1 import (states_all, nrmse_score, nrmse_val_test, NVAL, CONFIGS, RADII,
                         SEEDS, WASH, NTEST, results, RJ)

def henon(seed, T=5200, burn=2000):
    rng = np.random.default_rng(seed)
    X = np.zeros((T + burn, 2))
    x, y = rng.uniform(-0.2, 0.2, 2)
    for t in range(burn):  # converge vers l'attracteur (float64, garde)
        x, y = 1 - 1.4 * x * x + y, 0.3 * x
        if abs(x) > 10: x, y = rng.uniform(-0.2, 0.2, 2)
    traj = np.empty((T + 1, 2))
    for t in range(T + 1):
        traj[t] = (x, y)
        x, y = 1 - 1.4 * x * x + y, 0.3 * x
    u = np.hstack([traj[:-1], np.ones((len(traj)-1, 1), np.float32)]).astype(np.float32)
    return u, traj[1:].astype(np.float32)

def run_h(key, H):
    # 14.2 : meilleur rho par seed choisi sur la validation, jamais sur le test.
    if key in results and all(results[key].get(c, {}).get("proto") == "rho_sur_validation"
                              for c, *_ in CONFIGS):
        print(f"[skip] {key}"); return
    res = {}
    for cid, name, N, taus in CONFIGS:
        bests = []
        for seed in range(SEEDS):
            u, Y = henon(seed)
            y = Y[H:]
            best_val, best_test = np.inf, np.inf
            for rho in RADII:
                F = states_all(u, N, rho, taus, seed)[:len(y)]
                val, test = nrmse_val_test(F, y, WASH, NTEST, NVAL)
                if val < best_val:
                    best_val, best_test = val, test
            bests.append(best_test)
        res[cid] = {"name": name, "mean": float(np.mean(bests)), "std": float(np.std(bests)),
                    "per_seed": [float(b) for b in bests], "proto": "rho_sur_validation"}
        print(f"  {name:35s} {res[cid]['mean']:.4f} +/- {res[cid]['std']:.4f}", flush=True)
    results[key] = res
    json.dump(results, open(RJ, "w"), indent=2, ensure_ascii=False)

print("== Henon horizon 1 =="); run_h("henon_h1", 1)
print("== Henon horizon 5 =="); run_h("henon_h5", 5)
