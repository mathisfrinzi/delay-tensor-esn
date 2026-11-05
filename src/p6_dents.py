"""Partie 6 : test 6.3 : la dent de scie en x.

A ensemble fixe, changer l'ordre de NARMA d'un seul pas fait-il sauter la
performance la ou l'arithmetique le predit ?
  {1,10} : NRMSE(x) doit avoir des minima a x = 20, 30, 40 (multiples de
  10) et des maxima a x = 29, 39 ; NRMSE(29)/NRMSE(30) >= 2.
  {1,7} : periode 7 (bons 21, 28, 35, 42 ; mauvais 27, 34, 41).
  {1..10} : plat (CV < 10 % sur 25..45).
  rho = 0,9 : dents au moins 2x plus faibles qu'a 0,5.
  Autocorrelation de NRMSE(x) detrendee : pic au decalage 10 ({1,10}) et
  7 ({1,7}).

x in {20..45}, rho in {0.5, 0.9}, 5 seeds, protocole 6.0.
Sortie : results/results_p6_dents.json
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np, json, time, sys
from multiprocessing import Pool
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from p6_common import (make_reservoir, run_states, augment, ridge_select, nrmse_of,
                       narma_x, SEEDS, TIME_BUDGET, save, OUT)

XS = list(range(20, 46))
SETS = [("{1,10}", [1, 10], 447), ("{1,7}", [1, 7], 447),
        ("{1..10}", list(range(1, 11)), 200), ("esn", [1], 632)]
RHOS = [0.5, 0.9]
T = 5200
NAME = "dents"
_CACHE = {}


def one(args):
    sname, S, N, rho, x, seed = args
    key = (seed, x)
    if key not in _CACHE:
        _CACHE[key] = narma_x(seed, x, T, "std")
    u, y, gen = _CACHE[key]
    Win, W = make_reservoir(N, S, rho, seed)
    F = run_states(u, S, Win, W)
    wash = max(200, 3 * max(S))
    i_te = T - 1000
    Ftr = augment(F[wash:i_te]); ztr = y[wash:i_te]
    Fte = augment(F[i_te:]); zte = y[i_te:]
    pred, lam, _, _ = ridge_select(Ftr, ztr, Fte)
    return (sname, rho, x, seed, nrmse_of(pred, zte), lam, gen)


def detrended_acf_peak(vals, maxlag=15):
    """Pic d'autocorrelation de la serie detrendee lineairement."""
    v = np.asarray(vals, float)
    d = v - np.polyval(np.polyfit(range(len(v)), v, 1), range(len(v)))
    d = d - d.mean()
    denom = float(d @ d)
    if denom <= 0:
        return None
    acf = [float(np.correlate(d, d, "full")[len(d) - 1 + L] / denom) for L in range(maxlag + 1)]
    return int(np.argmax(acf[1:]) + 1), acf


def analyse(res):
    import numpy as np
    agg = res["agg"]
    agg["means"] = {}
    for sname, _, _ in SETS:
        for rho in RHOS:
            for x in XS:
                k = f"{sname}|x{x}|rho{rho}"
                cell = res["cells"].get(k, {})
                ss = [cell["seeds"][str(s)]["nrmse"] for s in SEEDS if str(s) in cell.get("seeds", {})]
                if len(ss) == len(SEEDS):
                    agg["means"][k] = {"mean": float(np.mean(ss)), "std": float(np.std(ss))}

    def m(sname, rho, x):
        return agg["means"].get(f"{sname}|x{x}|rho{rho}", {}).get("mean")

    # P-dents {1,10} a rho=0.5
    if all(m("{1,10}", 0.5, x) is not None for x in XS):
        mm = {x: m("{1,10}", 0.5, x) for x in XS}
        agg["P_1_10_rho05"] = {
            "m29": mm[29], "m30": mm[30], "m39": mm[39], "m40": mm[40], "m20": mm[20],
            "ratio_29_30": mm[29] / mm[30],
            "ratio_29_30_ge2": mm[29] >= 2 * mm[30],
            "ratio_39_40_ge2": mm[39] >= 2 * mm[40],
            "minima_aux_multiples_de_10": (mm[20] < mm[21] and mm[20] < mm[19] if 19 in mm else True)
                                          and mm[30] < mm[29] and mm[30] < mm[31]
                                          and mm[40] < mm[39] and mm[40] < mm[41],
            "serie": mm}
    # {1,7} a rho=0.5
    if all(m("{1,7}", 0.5, x) is not None for x in XS):
        mm7 = {x: m("{1,7}", 0.5, x) for x in XS}
        good = [21, 28, 35, 42]; bad = [27, 34, 41]
        agg["P_1_7_rho05"] = {
            "bons": {x: mm7[x] for x in good}, "mauvais": {x: mm7[x] for x in bad},
            "rapports_mauvais_sur_bon_suivant": {b: mm7[b] / mm7[b + 1] for b in bad},
            "tous_les_mauvais_1.5x_pires": all(mm7[b] >= 1.5 * mm7[b + 1] for b in bad),
            "serie": mm7}
    # {1..10} plat
    for rho in RHOS:
        vals = [m("{1..10}", rho, x) for x in range(25, 46)]
        if all(v is not None for v in vals):
            v = np.array(vals)
            agg[f"P_1_10_dense_plat_rho{rho}"] = {
                "cv": float(v.std() / v.mean()), "plat_cv_lt_10pct": bool(v.std() / v.mean() < 0.10)}
    # autocorrelations detrendees
    for sname, _, _ in SETS:
        for rho in RHOS:
            vals = [m(sname, rho, x) for x in XS]
            if all(v is not None for v in vals):
                r = detrended_acf_peak(vals)
                if r:
                    agg[f"acf_{sname}_rho{rho}"] = {"peak_lag": r[0], "acf": r[1]}
    # amplitude des dents rho 0.9 vs 0.5
    for sname in ("{1,10}", "{1,7}"):
        a5 = [m(sname, 0.5, x) for x in XS]
        a9 = [m(sname, 0.9, x) for x in XS]
        if all(v is not None for v in a5 + a9):
            amp5 = max(a5) - min(a5); amp9 = max(a9) - min(a9)
            agg[f"amplitude_{sname}"] = {
                "amp_rho05": amp5, "amp_rho09": amp9,
                "ratio_09_sur_05": amp9 / amp5 if amp5 > 0 else None,
                "dents_2x_plus_faibles_a_09": bool(amp9 <= 0.5 * amp5)}
    return agg


def main():
    from pathlib import Path
    path = OUT / f"results_p6_{NAME}.json"
    if path.exists():
        res = json.load(open(path))
    else:
        res = {"cells": {}, "agg": {},
               "meta": {"xs": XS, "sets": {s: N for s, _, N in SETS}, "rhos": RHOS,
                        "seeds": SEEDS, "T": T,
                        "predictions": {"{1,10}": "minima 20/30/40, maxima 29/39, m29/m30 >= 2",
                                        "{1,7}": "periode 7 (bons 21,28,35,42)",
                                        "{1..10}": "CV < 10% sur 25..45",
                                        "acf": "pic au decalage 10 resp. 7 apres detendance",
                                        "rho": "dents >= 2x plus faibles a 0.9"}}}
    t0 = time.time()
    todo = []
    for sname, S, N in SETS:
        for rho in RHOS:
            for x in XS:
                k = f"{sname}|x{x}|rho{rho}"
                cell = res["cells"].setdefault(k, {"seeds": {}})
                for s in SEEDS:
                    if str(s) not in cell["seeds"]:
                        todo.append((sname, S, N, rho, x, s))
    print(f"{len(todo)} runs restants", flush=True)
    if todo:
        with Pool(2) as pool:
            for sname, rho, x, seed, nrmse, lam, gen in pool.imap_unordered(one, todo, chunksize=4):
                cell = res["cells"][f"{sname}|x{x}|rho{rho}"]
                cell["seeds"][str(seed)] = {"nrmse": nrmse, "lam": lam, "gen": gen}
                if time.time() - t0 > TIME_BUDGET:
                    print("BUDGET : checkpoint", flush=True)
                    analyse(res); save(res, NAME)
                    return
    analyse(res)
    save(res, NAME)
    print("COMPLETE", flush=True)
    a = res["agg"]
    print("m29/m30:", round(a.get("P_1_10_rho05", {}).get("ratio_29_30", float("nan")), 2),
          "| acf {1,10} rho0.5:", a.get("acf_{1,10}_rho0.5", {}).get("peak_lag"),
          "| acf {1,7} rho0.5:", a.get("acf_{1,7}_rho0.5", {}).get("peak_lag"), flush=True)


main()
