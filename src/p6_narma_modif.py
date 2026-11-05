"""Partie 6 : test 6.6 : trois NARMA modifiees (variantes A/B/C de 6.0).

Si tau=1 est vital uniquement parce que la cible contient u(t-1), deplacer
ce retard doit deplacer la tranche vitale, et lisser l'entree doit rendre
le trou benin. 8 ensembles (budget 400k), variantes std (reference), A
(u(t).u(t-30)), B (u(t-2).u(t-30)), C (entree lissee), meilleur rho dans
{0.5, 0.9} par seed, 5 seeds, protocole 6.0.

Predictions pre-enregistrees :
  A : {2..10} a moins de 15 % de {1..10} ; pairs ~= impairs
  B : {3..10} et {3,10} tombent au palier du milieu (~0.75)
  C : l'ecart {2..10}-{1..10} reduit d'au moins moitie vs std

Sortie : p6/results_p6_narma_modif.json
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np, json, time, sys
from multiprocessing import Pool
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from p6_common import (make_reservoir, run_states, augment, ridge_select, nrmse_of,
                       narma_x, BUDGET, SEEDS, TIME_BUDGET, save, OUT)

SETS = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [2, 3, 4, 5, 6, 7, 8, 9, 10],
        [3, 4, 5, 6, 7, 8, 9, 10], [2, 4, 6, 8, 10], [1, 3, 5, 7, 9],
        [1, 10], [2, 10], [3, 10]]
VARIANTS = ["std", "A", "B", "C"]
RHOS = [0.5, 0.9]
T = 5200
NAME = "narma_modif"
_CACHE = {}


def sname(S):
    return "{" + ",".join(map(str, S)) + "}"


def one(args):
    S, variant, rho, seed = args
    key = (seed, variant)
    if key not in _CACHE:
        _CACHE[key] = narma_x(seed, 30, T, variant)
    u, y, gen = _CACHE[key]
    N = int(round((BUDGET / len(S)) ** 0.5))
    Win, W = make_reservoir(N, S, rho, seed)
    F = run_states(u, S, Win, W)
    wash = max(200, 3 * max(S))
    i_te = T - 1000
    Ftr = augment(F[wash:i_te]); ztr = y[wash:i_te]
    Fte = augment(F[i_te:]); zte = y[i_te:]
    pred, lam, _, _ = ridge_select(Ftr, ztr, Fte)
    return (sname(S), variant, rho, seed, nrmse_of(pred, zte), lam, gen, N)


def analyse(res):
    agg = res["agg"]
    agg["means"] = {}
    for S in SETS:
        for v in VARIANTS:
            cell = res["cells"].get(f"{sname(S)}|{v}", {})
            ss = cell.get("seeds", {})
            row = {}
            for rho in RHOS:
                vals = [ss[str(s)][f"nrmse_rho{rho}"] for s in SEEDS
                        if str(s) in ss and f"nrmse_rho{rho}" in ss[str(s)]]
                if len(vals) == len(SEEDS):
                    row[rho] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
            if len(row) == len(RHOS):
                agg["means"][f"{sname(S)}|{v}"] = row
    m = agg["means"]

    def get(s, v, rho=None):
        if f"{s}|{v}" not in m:
            return None
        if rho:
            return m[f"{s}|{v}"][rho]["mean"]
        return min(r["mean"] for r in m[f"{s}|{v}"].values())

    # A : {2..10} a moins de 15 % de {1..10} ; pairs vs impairs
    a1, a2 = get("{1,2,3,4,5,6,7,8,9,10}", "A"), get("{2,3,4,5,6,7,8,9,10}", "A")
    if a1 and a2:
        agg["A"] = {"set110": a1, "set210": a2, "ratio": a2 / a1,
                    "rehabilite (<=1.15)": a2 <= 1.15 * a1,
                    "pairs": get("{2,4,6,8,10}", "A"), "impairs": get("{1,3,5,7,9}", "A")}
    # B : {3..10} et {3,10} au palier du milieu
    b1, b2 = get("{3,4,5,6,7,8,9,10}", "B"), get("{3,10}", "B")
    if b1:
        agg["B"] = {"set310": b1, "set3_10": b2,
                    "palier_milieu (0.7-0.9)": (0.7 <= b1 <= 0.9) and (b2 is None or 0.7 <= b2 <= 0.9)}
    # C : ecart {2..10}-{1..10} reduit de moitie vs std
    def gap(v):
        g1 = get("{1,2,3,4,5,6,7,8,9,10}", v)
        g2 = get("{2,3,4,5,6,7,8,9,10}", v)
        return None if not (g1 and g2) else g2 - g1
    gs, gc = gap("std"), gap("C")
    if gs is not None and gc is not None:
        agg["C"] = {"gap_std": gs, "gap_C": gc, "reduction": 1 - gc / gs,
                    "reduit_de_moitie": gc <= 0.5 * gs}
    return agg


def main():
    path = OUT / f"results_p6_{NAME}.json"
    if path.exists():
        res = json.load(open(path))
    else:
        res = {"cells": {}, "agg": {}, "meta": {"sets": [sname(s) for s in SETS],
               "variants": VARIANTS, "rhos": RHOS, "seeds": SEEDS, "T": T,
               "predictions": {"A": "{2..10} <= 15% de {1..10}",
                               "B": "{3..10},{3,10} au palier du milieu",
                               "C": "ecart {2..10}-{1..10} reduit de moitie"}}}
    t0 = time.time()
    todo = []
    for S in SETS:
        for v in VARIANTS:
            cell = res["cells"].setdefault(f"{sname(S)}|{v}", {"seeds": {}})
            for rho in RHOS:
                for s in SEEDS:
                    sd = cell["seeds"].setdefault(str(s), {})
                    if f"nrmse_rho{rho}" not in sd:
                        todo.append((S, v, rho, s))
    print(f"{len(todo)} runs restants", flush=True)
    if todo:
        with Pool(2) as pool:
            for S, v, rho, seed, nrmse, lam, gen, N in pool.imap_unordered(one, todo, chunksize=4):
                cell = res["cells"][f"{S}|{v}"]
                sd = cell["seeds"].setdefault(str(seed), {})
                sd[f"nrmse_rho{rho}"] = nrmse
                sd["lam_rho" + str(rho)] = lam
                sd["gen"] = gen
                sd["N"] = N
                if time.time() - t0 > TIME_BUDGET:
                    print("BUDGET : checkpoint", flush=True)
                    analyse(res); save(res, NAME)
                    return
    analyse(res)
    save(res, NAME)
    print("COMPLETE", flush=True)
    for k in sorted(res["agg"]["means"]):
        row = res["agg"]["means"][k]
        print(f"{k}: rho0.5={row[0.5]['mean']:.3f} rho0.9={row[0.9]['mean']:.3f}", flush=True)
    print("A:", res["agg"].get("A"), flush=True)
    print("B:", res["agg"].get("B"), flush=True)
    print("C:", res["agg"].get("C"), flush=True)


main()
