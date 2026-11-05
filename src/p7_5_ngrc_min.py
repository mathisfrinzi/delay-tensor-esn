"""Partie 7.5 : Le NG-RC minimal : NARMA-30 est-elle trop facile ?
(rapport/4_comparaisons.md)

Predicteur lineaire-en-parametres SANS reservoir, features :
  [1, u(t-d) pour d dans D, tous les produits deux a deux des u(t-d)] :
  ridge. Trois ensembles de retards :
  - narma30_min    : D = {0, 1, 30} (les deux facteurs du produit NARMA,
                     ~10 features) ;
  - narma30_elected: D = {0} U <{1,10}> inter [1,30] = {0..30} ;
  - wh_elected     : D = {0} U <{4,7}> inter [1,30] (elu WH de 2.2) ;
  - silverbox_elected : D = {0} U <[2..17]> inter [1,30] = tous sauf 1.
<S> = semigroupe des sommes d'elements de S (definition 5.1). Meme protocole
de readout que 6.0 (ridge choisi sur les 20 % finaux du train, reentraine),
splits "val" de 2.4 : NARMA train [200, 3200), val [3200, 4200), test
[4200, 5200) ; phys train [500, 15000), val [15000, 20000), test
[20000, 30000). 3 seeds pour NARMA, deterministe pour phys.

Lecture (7.5) : si le NG-RC minimal approche 0,035 sur NARMA-30, NARMA-30
ne mesure que l'acces a deux retards ; si le NG-RC aux "retards du tenseur"
reste loin derriere sur WH, la difference est l'apport reel de l'etat
recurrent (filtrage).

Sortie : results_p7_ngrc_min.json.
Usage : python p7_5_ngrc_min.py <narma30|wh|silverbox|tous>
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p7_common as C

RJ = C.OUT / "results_p7_ngrc_min.json"
SEEDS_NARMA = [0, 1, 2] if not C.SMOKE else [0, 1]


def semigroup_delays(taus, kmax=30):
    """<S> inter [1, kmax] : toutes les sommes finies d'elements de S."""
    reach = {0}
    frontier = {0}
    while frontier:
        nxt = set()
        for v in frontier:
            for tau in taus:
                w = v + tau
                if w <= kmax and w not in reach:
                    reach.add(w)
                    nxt.add(w)
        frontier = nxt
    return sorted(d for d in reach if d >= 1)


def features(u, delays):
    """[1, u(t-d) pour d dans delays, produits deux a deux (i <= j)].
    Retourne Phi (T, P) float32."""
    T = len(u)
    nd = len(delays)
    n_mon = nd * (nd + 1) // 2
    Phi = np.zeros((T, 1 + nd + n_mon), np.float32)
    Phi[:, 0] = 1.0
    Ud = np.zeros((T, nd), np.float32)
    for j, d in enumerate(delays):
        if d > 0:
            Ud[d:, j] = u[:-d]
        else:
            Ud[:, j] = u
    Phi[:, 1:1 + nd] = Ud
    c = 1 + nd
    for i in range(nd):
        for j in range(i, nd):
            Phi[:, c] = Ud[:, i] * Ud[:, j]
            c += 1
    return Phi


def val_splits(task):
    """Splits 'val' de 2.4."""
    if task.startswith("narma"):
        T = 1600 if C.SMOKE else 5200
        nte = 300 if C.SMOKE else 1000
        wash = 200
        return {"T": T, "wash": wash, "tr_end": T - 2 * nte, "val0": T - 2 * nte,
                "val_end": T - nte, "te0": T - nte, "te1": T}
    T = 30000
    nte = 2000 if C.SMOKE else 10000
    return {"T": T, "wash": 500, "tr_end": 15000, "val0": 15000,
            "val_end": 20000, "te0": 20000, "te1": 30000}


def run(task, delays, results):
    key = f"{task}|d{'_'.join(map(str, delays))}" if len(delays) <= 4 \
        else f"{task}|elected_{len(delays)}delays"
    if key in results["cells"]:
        print(f"{key}: deja fait, skip")
        return
    sp = val_splits(task)
    tests, nfeat = [], 0
    seeds = SEEDS_NARMA if task.startswith("narma") else [0]
    for seed in seeds:
        if task.startswith("narma"):
            U, y, regen = C.narma30(seed)
            u = U[:, 0].astype(np.float64)
        else:
            U, y = C.load_phys(task)
            u = U[:, 0].astype(np.float64)
        Phi = features(u, delays)
        nfeat = Phi.shape[1]
        yv = np.asarray(y, np.float64)
        # ridge choisi sur le bloc de validation (6.0), reentraine sur le train
        best_lam, best_val = None, np.inf
        for lam in C.RIDGES:
            A = Phi[sp["wash"]:sp["tr_end"]].astype(np.float64)
            w = C._ridge_solve(A, yv[sp["wash"]:sp["tr_end"]], lam)
            resid = Phi[sp["val0"]:sp["val_end"]].astype(np.float64) @ w - yv[sp["val0"]:sp["val_end"]]
            val = float(np.sqrt(np.mean(resid ** 2)) / np.std(yv[sp["val0"]:sp["val_end"]]))
            if val < best_val:
                best_val, best_lam = val, lam
        w = C._ridge_solve(Phi[sp["wash"]:sp["tr_end"]].astype(np.float64),
                           yv[sp["wash"]:sp["tr_end"]], best_lam)
        pred = Phi[sp["te0"]:sp["te1"]].astype(np.float64) @ w
        nrmse = float(np.sqrt(np.mean((pred - yv[sp["te0"]:sp["te1"]]) ** 2))
                      / np.std(yv[sp["te0"]:sp["te1"]]))
        tests.append(nrmse)
        print(f"{key} seed{seed}: NRMSE={nrmse:.4f} lam={best_lam:.0e} ({nfeat} feats)",
              flush=True)
    cell = {"delays": list(map(int, delays)), "n_feat": int(nfeat),
            "tests": tests, "test_mean": float(np.mean(tests)),
            "test_std": float(np.std(tests)),
            "seeds": {str(s): v for s, v in zip(seeds, tests)}}
    results["cells"][key] = cell
    C.save_json(results, RJ)


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "tous"
    results = C.load_json(RJ)
    C.meta_init(results, "p7_5_ngrc_min.py",
                note="splits val de 2.4, readout 6.0")
    todo = {
        "narma30": ([("min", [0, 1, 30]),
                     ("elected", [0] + semigroup_delays([1, 10]))]),
        "wh": [("elected", [0] + semigroup_delays([4, 7]))],
        "silverbox": [("elected", [0] + semigroup_delays(list(range(2, 18))))],
    }
    if which == "tous":
        for task, runs in todo.items():
            for name, delays in runs:
                run(task, delays, results)
    else:
        for name, delays in todo[which]:
            run(which, delays, results)
    C.aggregate(results)
    C.save_json(results, RJ)
    print("COMPLETE")


if __name__ == "__main__":
    main()
