"""MG horizon h=10 et Lorenz, puis MC. Reutilise suite_part1."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from chemins import BRUT  # noqa: E402
import numpy as np, json, sys
from suite_part1 import (states_all, nrmse_score, nrmse_val_test, NVAL, mackey_glass,
                         lorenz, memory_capacity, mc_score, CONFIGS, RADII, SEEDS,
                         WASH, NTEST, results, RJ)

def run_task(key, gen, scorer, ntest, wash):
    # 14.2 : meilleur rho par seed choisi sur la validation, jamais sur le test.
    if key in results and all(results[key].get(c, {}).get("proto") == "rho_sur_validation"
                              for c, *_ in CONFIGS):
        print(f"[skip] {key}"); return
    res = {}
    for cid, name, N, taus in CONFIGS:
        bests = []
        for seed in range(SEEDS):
            u, y = gen(seed)
            best_val, best_test = np.inf, np.inf
            for rho in RADII:
                F = states_all(u, N, rho, taus, seed)[:len(y)]
                val, test = nrmse_val_test(F, y, wash, ntest, NVAL)
                if val < best_val:
                    best_val, best_test = val, test
            bests.append(best_test)
        res[cid] = {"name": name, "mean": float(np.mean(bests)), "std": float(np.std(bests)),
                    "per_seed": [float(b) for b in bests], "proto": "rho_sur_validation"}
        print(f"  {name:35s} {res[cid]['mean']:.4f} +/- {res[cid]['std']:.4f}", flush=True)
    results[key] = res
    json.dump(results, open(RJ, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

H = 10
print(f"== Mackey-Glass tau=17, horizon {H} ==")
run_task("mg17_h10", lambda s: (lambda uy: (uy[0], np.asarray(uy[1])[H:]))(mackey_glass(17, s)), nrmse_score, NTEST, WASH)
