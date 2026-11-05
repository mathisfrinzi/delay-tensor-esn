"""Partie 6 : test 6.4 : le crossover x*(rho), pre-enregistre.

ESN N=632 contre tenseur {1..10} N=200, au MEME rho. x* = point ou
log NRMSE_ESN - log NRMSE_tenseur change de signe (interpolation lineaire).
Previsions ecrites avant (theta = 1e-2) :
  rho=0.7 -> 6.5 ; 0.8 -> 10.3 ; 0.85 -> 14.2 ; 0.9 -> 21.9 ; 0.95 -> 44.9
Succes si |x*_mesure - x*_predit| <= max(5 ; 20 % de x*_predit) pour au
moins 4 des 5 rho. Version corrigee : rho_eff^2 = kappa*rho^2, kappa mesure
pour l'ESN a chaque rho. Secondaire : a rho=0.9, N_ESN in {316, 632, 1000}
(tenseur au meme budget, N = round(N_ESN/sqrt(10))).

Phase 1 grossiere x in {5..70 par pas de 5}, phase 2 pas de 1 autour du
crossover, phase 3 balayage de N a rho=0.9. 10 seeds, protocole 6.0.

Sortie : results/results_p6_crossover.json
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np, json, time, sys
from multiprocessing import Pool
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from p6_common import (make_reservoir, run_states, augment, ridge_select, nrmse_of,
                       narma_x, TIME_BUDGET, save, OUT)

RHOS = [0.7, 0.8, 0.85, 0.9, 0.95]
XS_COARSE = list(range(5, 71, 5))
T = 5200
SEEDS = list(range(10))
THETA = 1e-2
NAME = "crossover"
# label -> (S, N)
MAIN = {"esn": ([1], 632), "tens": (list(range(1, 11)), 200)}
SWEEP = [("esn", [1], 316), ("tens", list(range(1, 11)), 100),
         ("esn", [1], 1000), ("tens", list(range(1, 11)), 316)]
_CACHE = {}


def one(args):
    label, S, N, rho, x, seed = args
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
    return (label, N, rho, x, seed, nrmse_of(pred, zte), lam, gen)


def kappa_esn(rho, seed):
    """kappa = mean_t,i (1 - x_i(t)^2)^2 pour l'ESN N=632 sur u NARMA."""
    N = 632
    u, _, _ = narma_x(seed, 30, T, "std")
    Win, W = make_reservoir(N, [1], rho, seed)
    F = run_states(u, [1], Win, W)[200:].astype(np.float64)
    return float(np.mean((1.0 - F ** 2) ** 2))


def ckey(label, N, rho, x):
    return f"{label}|N{N}|rho{rho}|x{x}"


def run_pool(todo, res, t0):
    with Pool(2) as pool:
        for label, N, rho, x, seed, nrmse, lam, gen in pool.imap_unordered(one, todo, chunksize=4):
            cell = res["cells"].setdefault(ckey(label, N, rho, x), {"seeds": {}})
            cell["seeds"][str(seed)] = {"nrmse": nrmse, "lam": lam, "gen": gen}
            if time.time() - t0 > TIME_BUDGET:
                return False
    return True


def mean_nrmse(res, label, N, rho, x):
    cell = res["cells"].get(ckey(label, N, rho, x), {})
    ss = [cell["seeds"][str(s)]["nrmse"] for s in SEEDS if str(s) in cell.get("seeds", {})]
    return (float(np.mean(ss)) if len(ss) == len(SEEDS) else None)


def crossover_x(res, label_e, label_t, N_e, N_t, rho, xs):
    """x* par interpolation lineaire sur d(x) = log NRMSE_esn - log NRMSE_tens."""
    d, xvals = [], []
    for x in sorted(xs):
        me, mt = mean_nrmse(res, label_e, N_e, rho, x), mean_nrmse(res, label_t, N_t, rho, x)
        if me is not None and mt is not None and me > 0 and mt > 0:
            d.append((x, float(np.log(me) - np.log(mt))))
            xvals.append(x)
    for (x1, d1), (x2, d2) in zip(d, d[1:]):
        if d1 < 0 <= d2:
            return float(x1 + (x2 - x1) * (-d1) / (d2 - d1)), d
    if d and d[0][1] >= 0:
        return "<5", d
    if d and d[-1][1] < 0:
        return ">70", d
    return None, d


def analyse(res):
    agg = res["agg"]
    agg["kappa_esn"] = res.get("kappa_esn", {})
    agg["x_star"] = {}
    for rho in RHOS:
        xs = sorted(set(XS_COARSE) | set(res.get("fine_xs", {}).get(str(rho), [])))
        xs_c, d = crossover_x(res, "esn", "tens", 632, 200, rho, xs)
        pred_raw = float(np.log(THETA) / (2 * np.log(rho)))
        kap = res.get("kappa_esn", {}).get(str(rho))
        pred_kap = float(np.log(THETA) / np.log(kap * rho * rho)) if kap else None
        ok_raw = isinstance(xs_c, float) and abs(xs_c - pred_raw) <= max(5, 0.2 * pred_raw)
        ok_kap = (isinstance(xs_c, float) and pred_kap is not None
                  and abs(xs_c - pred_kap) <= max(5, 0.2 * pred_kap))
        # theta qu'il aurait fallu (si crossover mesure)
        theta_needed = None
        if isinstance(xs_c, float) and kap:
            # e(ESN) ~ (kappa rho^2)^x ; e(x*) = theta => theta = (kappa rho^2)^{x*}
            theta_needed = float((kap * rho * rho) ** xs_c)
        agg["x_star"][str(rho)] = {
            "x_mesure": xs_c, "pred_brut": pred_raw, "pred_kappa": pred_kap,
            "ok_brut": bool(ok_raw), "ok_kappa": bool(ok_kap),
            "theta_necessaire": theta_needed, "d": d}
    # balayage de N a rho = 0.9
    agg["n_sweep_rho09"] = {}
    for (e_lab, e_S, N_e), (t_lab, t_S, N_t) in [
            (("esn", [1], 316), ("tens", list(range(1, 11)), 100)),
            (("esn", [1], 632), ("tens", list(range(1, 11)), 200)),
            (("esn", [1], 1000), ("tens", list(range(1, 11)), 316))]:
        xs = sorted(set(XS_COARSE) | set(res.get("fine_xs", {}).get(f"{e_lab}{N_e}", [])))
        xs_c, d = crossover_x(res, e_lab, t_lab, N_e, N_t, 0.9, xs)
        agg["n_sweep_rho09"][f"N{N_e}"] = {"x_mesure": xs_c, "d": d}
    n_ok = sum(1 for rho in RHOS if agg["x_star"][str(rho)].get("ok_brut"))
    n_ok_k = sum(1 for rho in RHOS if agg["x_star"][str(rho)].get("ok_kappa"))
    agg["verdict"] = {"critere_brut_4sur5": bool(n_ok >= 4), "n_ok_brut": n_ok,
                      "n_ok_kappa": n_ok_k,
                      "critere_kappa_4sur5": bool(n_ok_k >= 4)}
    return agg


def main():
    path = OUT / f"results_p6_{NAME}.json"
    if path.exists():
        res = json.load(open(path))
    else:
        res = {"cells": {}, "agg": {}, "kappa_esn": {}, "fine_xs": {},
               "meta": {"rhos": RHOS, "xs_coarse": XS_COARSE, "T": T, "seeds": SEEDS,
                        "theta": THETA,
                        "predictions": {"0.7": 6.5, "0.8": 10.3, "0.85": 14.2,
                                        "0.9": 21.9, "0.95": 44.9},
                        "critere": "|x*-x*_pred| <= max(5; 20%) pour >= 4/5 rho"}}
    t0 = time.time()

    def missing(label, S, N, rho, xs):
        out = []
        for x in xs:
            cell = res["cells"].get(ckey(label, N, rho, x), {})
            for s in SEEDS:
                if str(s) not in cell.get("seeds", {}):
                    out.append((label, S, N, rho, x, s))
        return out

    # phase 1 : grossier, 5 rho, esn 632 / tens 200
    todo = []
    for rho in RHOS:
        for label, (S, N) in MAIN.items():
            todo += missing(label, S, N, rho, XS_COARSE)
    if todo:
        print(f"P1 : {len(todo)} runs", flush=True)
        if not run_pool(todo, res, t0):
            analyse(res); save(res, NAME); print("BUDGET P1"); return
    # x_c grossier par rho
    for rho in RHOS:
        key = str(rho)
        if key not in res["fine_xs"]:
            xs_c, _ = crossover_x(res, "esn", "tens", 632, 200, rho, XS_COARSE)
            if isinstance(xs_c, float):
                xc = int(round(xs_c))
                res["fine_xs"][key] = [x for x in range(max(5, xc - 6), min(70, xc + 6) + 1)
                                       if x not in XS_COARSE]
            else:
                res["fine_xs"][key] = []
    # phase 2 : fin
    todo = []
    for rho in RHOS:
        fine = res["fine_xs"].get(str(rho), [])
        for label, (S, N) in MAIN.items():
            todo += missing(label, S, N, rho, fine)
    if todo:
        print(f"P2 : {len(todo)} runs", flush=True)
        if not run_pool(todo, res, t0):
            analyse(res); save(res, NAME); print("BUDGET P2"); return
    # kappa ESN par rho (3 seeds)
    for rho in RHOS:
        if str(rho) not in res["kappa_esn"]:
            ks = [kappa_esn(rho, s) for s in [0, 1, 2]]
            res["kappa_esn"][str(rho)] = float(np.mean(ks))
            print(f"kappa_esn rho={rho}: {res['kappa_esn'][str(rho)]:.3f}", flush=True)
            if time.time() - t0 > TIME_BUDGET:
                analyse(res); save(res, NAME); print("BUDGET kappa"); return
    # phase 3 : balayage de N a rho=0.9 (316/100 et 1000/316 ; 632/200 deja fait)
    for e_lab, e_S, N_e in [("esn", [1], 316), ("esn", [1], 1000)]:
        t_lab = "tens"
        N_t = round(N_e / np.sqrt(10))
        todo = missing(e_lab, e_S, N_e, 0.9, XS_COARSE)
        todo += missing(t_lab, list(range(1, 11)), N_t, 0.9, XS_COARSE)
        if todo:
            print(f"P3 N={N_e} : {len(todo)} runs", flush=True)
            if not run_pool(todo, res, t0):
                analyse(res); save(res, NAME); print("BUDGET P3a"); return
        key = f"{e_lab}{N_e}"
        if key not in res["fine_xs"]:
            xs_c, _ = crossover_x(res, e_lab, t_lab, N_e, N_t, 0.9, XS_COARSE)
            if isinstance(xs_c, float):
                xc = int(round(xs_c))
                res["fine_xs"][key] = [x for x in range(max(5, xc - 6), min(70, xc + 6) + 1)
                                        if x not in XS_COARSE]
            else:
                res["fine_xs"][key] = []
        todo = missing(e_lab, e_S, N_e, 0.9, res["fine_xs"][key])
        todo += missing(t_lab, list(range(1, 11)), N_t, 0.9, res["fine_xs"][key])
        if todo:
            print(f"P3 fin N={N_e} : {len(todo)} runs", flush=True)
            if not run_pool(todo, res, t0):
                analyse(res); save(res, NAME); print("BUDGET P3b"); return
    analyse(res)
    save(res, NAME)
    print("COMPLETE", flush=True)
    for rho in RHOS:
        v = res["agg"]["x_star"][str(rho)]
        print(f"rho={rho}: x*={v['x_mesure']} pred_brut={v['pred_brut']:.1f} "
              f"pred_kap={v['pred_kappa'] and round(v['pred_kappa'],1)} ok={v['ok_brut']}/{v['ok_kappa']}", flush=True)
    print("verdict:", res["agg"]["verdict"], flush=True)
    print("N-sweep:", {k: v["x_mesure"] for k, v in res["agg"]["n_sweep_rho09"].items()}, flush=True)


main()
