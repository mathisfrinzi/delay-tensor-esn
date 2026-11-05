"""Partie 6 : test 6.2 : balayage S = {1, d} sur NARMA-30.

d in {2..40} U {45,50,60}, N=447, rho in {0.1,0.2,0.3,0.5,0.7,0.9}, 5 seeds,
protocole 6.0. Grandeurs analytiques par d : h(d) = nb de sauts min pour
ecrire 30 (h=30 si d>30), e(30) (renouvellement 5.2, g_tau = rho^2/K),
iota(30) = e(30)/(e(28)+e(29)+e(31)+e(32)) (isolement 5.4).
Top-5 d (a leur meilleur rho) : variante rang faible r = N/4 (6.0/2.2).

Predictions pre-enregistrees (doc 6.2) :
  P1 : Spearman(NRMSE, h) >= 0.6 a rho=0.5
  P2 : d=30 et d=15 battent 0.051 (record {1,10}) a leur meilleur rho
  P3 : d=16..19 (12-15 sauts) >= 2x pires que d=15
  P4 : d=29, d=28 bons ; d>30 retombe > 0.2
  P5 : meilleur rho plus bas pour d=30 que pour d=10

Sortie : p6/results_p6_balayage_d.json
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np, json, time, sys
from multiprocessing import Pool
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from p6_common import (make_reservoir, make_reservoir_rank, augment, ridge_select,
                       nrmse_of, narma_x, renewal_e, SEEDS, TIME_BUDGET, save, OUT)

DS = list(range(2, 41)) + [45, 50, 60]
DS_EARLY = [10, 15, 30, 16, 17, 18, 19, 28, 29, 40, 45, 50, 60]
DS_ORDER = DS_EARLY + [d for d in DS if d not in DS_EARLY]
RHOS = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]  # 0.1 et 0.2 ajoutés : 0.3 était le bord de la grille
X, T, N = 30, 5200, 447
RANK_R = N // 4  # 111
NAME = "balayage_d"
_CACHE = {}


def run_pair(u, d, Win, W1, Wd):
    """x(t) = tanh(W_in [u(t),1] + W1 x(t-1) + Wd x(t-d)), etat nul avant 0."""
    Tn = len(u)
    ub = np.stack([u, np.ones_like(u)], 1).astype(np.float32)
    F = np.empty((Tn, Win.shape[1]), np.float32)
    ring = np.zeros((max(d, 1), Win.shape[1]), np.float32)
    x1 = np.zeros(Win.shape[1], np.float32)
    for t in range(Tn):
        x_ = np.tanh(ub[t] @ Win + W1 @ x1 + Wd @ ring[t % d])
        F[t] = x_
        x1 = x_
        ring[t % d] = x_
    return F


def one(args):
    d, rho, seed, rank = args
    if (seed, "std") not in _CACHE:
        _CACHE[(seed, "std")] = narma_x(seed, X, T, "std")
    u, y, gen = _CACHE[(seed, "std")]
    if rank:
        Win, W = make_reservoir_rank(N, [1, d], rho, seed, RANK_R)
    else:
        Win, W = make_reservoir(N, [1, d], rho, seed)
    F = run_pair(u, d, Win, W[0], W[1])
    wash = max(200, 3 * d)
    i_te = T - 1000
    Ftr = augment(F[wash:i_te]); ztr = y[wash:i_te]
    Fte = augment(F[i_te:]); zte = y[i_te:]
    pred, lam, val, _ = ridge_select(Ftr, ztr, Fte)
    return (d, rho, seed, rank, nrmse_of(pred, zte), lam, gen, val)


def analytical():
    ana = {}
    for d in DS:
        h = (30 // d + 30 % d) if d <= 30 else 30
        row = {"h": h}
        for rho in RHOS:
            e = renewal_e([1, d], rho, 32)
            row[f"e30_rho{rho}"] = float(e[30])
            row[f"i30_rho{rho}"] = float(e[30] / max(e[28] + e[29] + e[31] + e[32], 1e-300))
        ana[str(d)] = row
    return ana


def cellkey(d, rho, rank=False):
    return f"d{d}{'_r' if rank else ''}|rho{rho}"


def analyse(res):
    """Moyennes par (d, rho) + predictions P1-P5 dans agg."""
    import numpy as np
    agg = res["agg"]
    # 14.2 : deux jeux de moyennes — test (rapportage) et validation (choix
    # du rho). Le meilleur rho d'un d n'est JAMAIS choisi sur le test.
    agg["means"] = {}
    agg["means_val"] = {}
    for d in DS:
        for rho in RHOS:
            for rank in [False, True]:
                k = cellkey(d, rho, rank)
                cell = res["cells"].get(k, {})
                done = [str(s) for s in SEEDS
                        if str(s) in cell.get("seeds", {})
                        and "val" in cell["seeds"][str(s)]]
                if len(done) == len(SEEDS):
                    ss = [cell["seeds"][q]["nrmse"] for q in done]
                    vv = [cell["seeds"][q]["val"] for q in done]
                    agg["means"][k] = {"mean": float(np.mean(ss)), "std": float(np.std(ss))}
                    agg["means_val"][k] = {"mean": float(np.mean(vv)), "std": float(np.std(vv))}
    # P1 : Spearman a rho=0.5
    try:
        from scipy.stats import spearmanr
        pairs = [(res["ana"][str(d)]["h"], agg["means"][cellkey(d, 0.5)]["mean"])
                 for d in DS if cellkey(d, 0.5) in agg["means"]]
        if len(pairs) == len(DS):
            hh, nn = zip(*pairs)
            agg["P1_spearman_h_rho05"] = float(spearmanr(hh, nn).statistic)
    except Exception as e:
        agg["P1_error"] = str(e)
    # helpers meilleur rho (14.2 : choix sur la validation ; la valeur
    # rapportee est le NRMSE test au rho choisi)
    def best(d, rank=False):
        cands = [(agg["means_val"][cellkey(d, r, rank)]["mean"],
                  agg["means"][cellkey(d, r, rank)]["mean"], r) for r in RHOS
                 if cellkey(d, r, rank) in agg["means_val"]]
        return min(cands) if cands else None
    agg["best"] = {}
    for d in DS:
        b = best(d)
        if b: agg["best"][str(d)] = {"nrmse": b[1], "rho": b[2], "val": b[0]}
    if "10" in agg["best"] and "15" in agg["best"] and "30" in agg["best"]:
        agg["P2"] = {"d10": agg["best"]["10"], "d15": agg["best"]["15"],
                     "d30": agg["best"]["30"],
                     "beat_0.051": {k: agg["best"][k]["nrmse"] < 0.051 for k in ["15", "30"]}}
    if all(cellkey(d, 0.5) in agg["means"] for d in [15, 16, 17, 18, 19]):
        m15 = agg["means"][cellkey(15, 0.5)]["mean"]
        m1619 = [agg["means"][cellkey(d, 0.5)]["mean"] for d in [16, 17, 18, 19]]
        agg["P3"] = {"m15_rho05": m15, "m1619_rho05": m1619,
                     "ratio_min": float(min(m1619) / m15), "ratio_mean": float(np.mean(m1619) / m15)}
    if all(str(d) in agg["best"] for d in [28, 29]):
        over30 = [agg["best"][str(d)]["nrmse"] for d in [45, 50, 60] if str(d) in agg["best"]]
        agg["P4"] = {"d28": agg["best"]["28"], "d29": agg["best"]["29"],
                     "over30_best": over30, "over30_gt_0.2": all(v > 0.2 for v in over30)}
    if "10" in agg["best"] and "30" in agg["best"]:
        agg["P5"] = {"best_rho_d10": agg["best"]["10"]["rho"], "best_rho_d30": agg["best"]["30"]["rho"],
                     "P5_ok": agg["best"]["30"]["rho"] < agg["best"]["10"]["rho"]}
    # top-5 pour la variante rang
    if len(agg["best"]) == len(DS):
        # 14.2 : le top-5 (variantes rang faible) est choisi sur la validation
        top5 = sorted(agg["best"].items(), key=lambda kv: kv[1]["val"])[:5]
        agg["top5_rank"] = {k: v for k, v in top5}
    return agg


def main():
    from pathlib import Path
    path = OUT / f"results_p6_{NAME}.json"
    if path.exists():
        res = json.load(open(path))
        res["meta"]["rhos"] = RHOS
        res["ana"] = analytical()
    else:
        res = {"cells": {}, "agg": {}, "ana": analytical(),
               "meta": {"d": DS, "rhos": RHOS, "N": N, "x": X, "T": T, "seeds": SEEDS,
                        "rank_r": RANK_R,
                        "predictions": {"P1": "Spearman(NRMSE,h)>=0.6 a rho=0.5",
                                        "P2": "d=30,15 < 0.051",
                                        "P3": "d=16..19 >= 2x d=15",
                                        "P4": "d=28,29 bons ; d>30 > 0.2",
                                        "P5": "meilleur rho(d=30) < meilleur rho(d=10)"}}}
    t0 = time.time()
    # phase 1 : balayage standard
    todo = []
    for d in DS_ORDER:
        for rho in RHOS:
            cell = res["cells"].setdefault(cellkey(d, rho), {"seeds": {}})
            for s in SEEDS:
                if str(s) not in cell["seeds"] or "val" not in cell["seeds"][str(s)]:
                    todo.append((d, rho, s, False))
    print(f"{len(todo)} runs standard restants", flush=True)
    if todo:
        with Pool(2) as pool:
            for d, rho, seed, rank, nrmse, lam, gen, val in pool.imap_unordered(one, todo, chunksize=4):
                cell = res["cells"].setdefault(cellkey(d, rho, rank), {"seeds": {}})
                cell["seeds"][str(seed)] = {"nrmse": nrmse, "lam": lam, "gen": gen, "val": val}
                if time.time() - t0 > TIME_BUDGET:
                    print("BUDGET phase 1 : checkpoint", flush=True)
                    analyse(res); save(res, NAME)
                    return
        analyse(res)
        # phase 2 : top-5 rang faible au meilleur rho
        top5 = res["agg"].get("top5_rank", {})
        todo2 = []
        for ds, info in top5.items():
            d = int(ds); rho = info["rho"]
            cell = res["cells"].setdefault(cellkey(d, rho, True), {"seeds": {}})
            for s in SEEDS:
                if str(s) not in cell["seeds"] or "val" not in cell["seeds"][str(s)]:
                    todo2.append((d, rho, s, True))
        print(f"{len(todo2)} runs rang faible (top-5)", flush=True)
        if todo2:
            with Pool(2) as pool:
                for d, rho, seed, rank, nrmse, lam, gen, val in pool.imap_unordered(one, todo2):
                    cell = res["cells"].setdefault(cellkey(d, rho, True), {"seeds": {}})
                    cell["seeds"][str(seed)] = {"nrmse": nrmse, "lam": lam, "gen": gen, "val": val}
                    if time.time() - t0 > TIME_BUDGET:
                        print("BUDGET phase 2 : checkpoint", flush=True)
                        analyse(res); save(res, NAME)
                        return
    analyse(res)
    save(res, NAME)
    m = res["agg"]["means"]
    print("COMPLETE", flush=True)
    for d in [10, 15, 30, 16, 17, 18, 19, 28, 29, 45]:
        if cellkey(d, 0.5) in m:
            print(f"d={d}: rho0.5={m[cellkey(d,0.5)]['mean']:.4f}", end="  ")
            print(f"best={res['agg']['best'].get(str(d))}" if str(d) in res["agg"]["best"] else "")
    print("P1:", res["agg"].get("P1_spearman_h_rho05"), flush=True)


main()
