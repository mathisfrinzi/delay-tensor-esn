"""Partie 8 : 8.3 : valider le S calcule contre la force brute.

A budget egal (400k connexions, 5 seeds, protocole 6.0) :
  - S calcule (8.2) pour k in {2,3,4}, SANS les gains (uniforme, rho balaye
    {0.3, 0.5, 0.7, 0.9}) et AVEC les gains optimises (a rho*) ;
  - l'elu du pruning glouton (2.1/2.2) : narma30 {1,10}, silverbox
    {2,3,4,5,7,8,10}, wh {4,7} ; narma25 : pruning ad hoc (replication du
    protocole 2.2, ~54 evaluations) fait ici ;
  - l'elu de la bande continue (2.1) : narma30 [1, 12.415], silverbox
    [2.013, 17.013], wh [3.234, 10.126] ; narma25 : bande ad hoc (section
    doree, ~44 evaluations) ;
  - tenseur uniforme {1..10} et ESN N=632.
Baselines a rho in {0.5, 0.9} (convention des campagnes physiques du projet).

Splits : NARMA protocole 6.0 (T=5200). Silverbox/WH : T=60000, wash 500,
train [500, 40000), test [40000, 60000), ridge protocole 6.0 (lambda sur les
20 % finaux du train).

Cout de conception note : 0 pour 8.2 (scalaires), ~54 evaluations pour le
pruning, ~44 pour la bande (comptes exacts depuis les JSON 2.x).

Sortie : p8/results_p8_conception.json {cells, agg, adhoc, meta}.
"""
import numpy as np, json, time, os, sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p6_common import (make_reservoir, run_states, augment, ridge_select,
                       nrmse_of, narma_x, BUDGET, SEEDS)
from p8_spectre import P8OUT, save_p8, task_data, ensure_datasets

TIME_BUDGET = float(__import__("os").environ.get("TEMPS_MAX", "250"))
N_SEEDS = SEEDS
SBWH_T = 60_000
SBWH_WASH = 500
SBWH_ITE = 40_000
TMAX_BAND = 20


# reservoirs

def make_reservoir_g(N, taus, radii, seed):
    """Comme make_reservoir (ordre 6.0) mais rayon par tranche = radii[i]."""
    rng = np.random.default_rng(seed)
    Win = rng.uniform(-0.5, 0.5, size=(2, N)).astype(np.float32)
    taus = sorted(taus)
    K = len(taus)
    W = np.empty((K, N, N), np.float32)
    for i, tau in enumerate(taus):
        M = rng.normal(0, 1, (N, N)).astype(np.float32)
        r = max(np.abs(np.linalg.eigvals(M.astype(np.float64))).max(), 1e-12)
        W[i] = M * (radii[i] / r)
    return Win, W


def run_one(spec):
    """Un run complet. spec: dict(task, tag, taus, radii|None, rho, seed)."""
    task, tag, taus, radii, rho, seed = (spec["task"], spec["tag"],
                                         spec["taus"], spec["radii"],
                                         spec["rho"], spec["seed"])
    K = len(taus)
    N = int(round((BUDGET / K) ** 0.5))
    if radii is None:
        Win, W = make_reservoir(N, taus, rho, seed)
    else:
        Win, W = make_reservoir_g(N, taus, radii, seed)
    if task.startswith("narma"):
        x = 30 if task == "narma30" else 25
        u, y, gen = narma_x(seed, x)
        T = len(u)
        F = run_states(u, taus, Win, W)
        wash = max(200, 3 * max(taus))
        i_te = T - 1000
    else:
        u50, y50 = task_data(task, 0)
        u, y = u50[:SBWH_T], y50[:SBWH_T]
        F = run_states(np.asarray(u, np.float32), taus, Win, W)
        wash = SBWH_WASH
        i_te = SBWH_ITE
    Ftr = augment(F[wash:i_te]); ztr = y[wash:i_te]
    Fte = augment(F[i_te:]); zte = y[i_te:]
    pred, lam, val, _ = ridge_select(Ftr, ztr, Fte)
    return {"task": task, "tag": tag, "rho": rho, "seed": seed,
            "nrmse": nrmse_of(pred, zte), "val": val, "lam": lam, "N": N}


# bandes continues (gains par couche, comme 2.1)

def band_gains(tau0, tau1):
    ks = np.arange(1, TMAX_BAND + 1)
    return np.clip(ks - tau0 + 1, 0, 1) * np.clip(tau1 - ks + 1, 0, 1)


def band_spec(tau0, tau1, rho):
    g = band_gains(tau0, tau1)
    act = np.where(g > 0)[0]
    sn = np.sqrt((g ** 2).sum())
    radii = [rho * g[i] / sn for i in act]
    return {"taus": [int(i) + 1 for i in act], "radii": radii}


# recherches ad hoc pour narma25 (replication 2.1/2.2)

def narma25_search_data(seed=0):
    u, y, gen = narma_x(seed, 25, T=5200)
    return np.asarray(u, np.float32), y


def search_score(F, y, tr_sl, va_sl, te_sl):
    Fb = np.hstack([F, np.ones((len(F), 1), np.float32)]).astype(np.float64)
    A = Fb[tr_sl].T @ Fb[tr_sl] + 1e-8 * np.eye(Fb.shape[1])
    Wo = np.linalg.solve(A, Fb[tr_sl].T @ y[tr_sl])
    return float(np.sqrt(np.mean((Fb[va_sl] @ Wo - y[va_sl]) ** 2)) / np.std(y[te_sl]))


def narma25_states(u, N, taus, rho, seed):
    Win, W = make_reservoir(N, taus, rho, seed)
    return run_states(u, taus, Win, W)


def adhoc_prune_narma25(res, t0):
    """Greedy backward depuis {1..10}, N redistribue, seed 0, rho 0.9."""
    if "prune" in res["adhoc"]["narma25"] and res["adhoc"]["narma25"]["prune"].get("done"):
        return
    P = res["adhoc"]["narma25"].setdefault("prune", {"path": [], "kept": list(range(1, 11)),
                                                     "step_scores": {}, "done": False})
    u, y = narma25_search_data(0)
    tr, va, te = slice(200, 3200), slice(3200, 4200), slice(4200, 5200)
    while len(P["kept"]) > 1 and time.time() - t0 < TIME_BUDGET:
        kept = P["kept"]; K = len(kept)
        for k in kept:
            ck = f"K{K}_rem{k}"
            if ck in P["step_scores"]:
                continue
            if time.time() - t0 > TIME_BUDGET:
                break
            cand = [x for x in kept if x != k]
            N = int(round((BUDGET / len(cand)) ** 0.5))
            F = narma25_states(u, N, cand, 0.9, 0)
            P["step_scores"][ck] = search_score(F, y, tr, va, te)
            print(f"[prune25] K={K} rem{k} val={P['step_scores'][ck]:.4f}", flush=True)
            save_p8(res, "conception")
        if f"K{K}_rem{kept[0]}" not in P["step_scores"]:
            break
        scores = {k: P["step_scores"][f"K{K}_rem{k}"] for k in kept}
        bk = min(scores, key=scores.get)
        P["path"].append({"K": K, "removed": bk, "val": scores[bk],
                          "kept_after": [x for x in kept if x != bk]})
        P["kept"] = [x for x in kept if x != bk]
        save_p8(res, "conception")
    if len(P["kept"]) == 1:
        P["done"] = True
        best = min(P["path"], key=lambda s: s["val"])
        P["elu"] = best["kept_after"]
        print("[prune25] elu:", P["elu"], flush=True)


def adhoc_band_narma25(res, t0):
    """Section doree par coordonnee, comme 2.1 (2 passes, depart (1,10)),
    N=200, rho=0.9, seed 0. Cache par (tau0, tau1) : reprise exacte."""
    if "band" in res["adhoc"]["narma25"] and res["adhoc"]["narma25"]["band"].get("done"):
        return
    B = res["adhoc"]["narma25"].setdefault("band", {"history": [], "cache": {}, "done": False})
    u, y = narma25_search_data(0)
    tr, va, te = slice(200, 3200), slice(3200, 4200), slice(4200, 5200)
    INV = (np.sqrt(5) - 1) / 2

    def ev(tau0, tau1):
        key = f"{round(tau0,4)}|{round(tau1,4)}"
        if key in B["cache"]:
            return B["cache"][key]
        bs = band_spec(tau0, tau1, 0.9)
        Win, W = make_reservoir_g(200, bs["taus"], bs["radii"], 0)
        F = run_states(u, bs["taus"], Win, W)
        v = search_score(F, y, tr, va, te)
        B["cache"][key] = v
        B["history"].append({"tau0": round(tau0, 4), "tau1": round(tau1, 4), "val": v})
        save_p8(res, "conception")
        return v

    def golden(fnc, lo, hi):
        a, b = lo, hi
        c = b - INV * (b - a); d = a + INV * (b - a)
        fc, fd = fnc(c), fnc(d)
        for _ in range(7):
            if fc < fd:
                b, d, fd = d, c, fc
                c = b - INV * (b - a); fc = fnc(c)
            else:
                a, c, fc = c, d, fd
                d = a + INV * (b - a); fd = fnc(d)
        return a if fc <= fd else d

    cur = [1.0, 10.0]
    best = [ev(cur[0], cur[1]), cur[0], cur[1]]
    for _pass in range(2):
        for coord in (0, 1):
            if time.time() - t0 > TIME_BUDGET:
                return
            if coord == 0:
                lo, hi = max(1.0, cur[1] - 9.0), cur[1] - 1.0
            else:
                lo, hi = cur[0] + 1.0, 20.5
            m = golden(lambda v: ev(*([v, cur[1]] if coord == 0 else [cur[0], v])), lo, hi)
            cur[coord] = round(m, 4)
            v = ev(cur[0], cur[1])
            if v < best[0]:
                best = [v, cur[0], cur[1]]
    B["done"] = True
    B["elu"] = {"tau0": best[1], "tau1": best[2], "val": best[0]}
    print("[band25] elu:", B["elu"], flush=True)


# specs

def build_specs(res):
    calc = json.load(open(P8OUT / "results_p8_calcul.json"))
    band_elus = {"narma30": (1.0, 12.415), "silverbox": (2.013, 17.013), "wh": (3.234, 10.126)}
    prune_elus = {"narma30": [1, 10], "silverbox": [2, 3, 4, 5, 7, 8, 10], "wh": [4, 7]}
    tasks = ["narma30", "narma25", "silverbox", "wh"]
    specs = []
    for task in tasks:
        cells = res["cells"].setdefault(task, {})
        def add(tag, taus, rhos, radii=None):
            for rho in rhos:
                for seed in N_SEEDS:
                    key = f"{tag}|rho{rho}|s{seed}"
                    if key in cells and "val" in cells[key]:
                        continue  # 14.2 : cellule deja conforme
                    specs.append({"task": task, "tag": tag, "taus": list(taus),
                                  "radii": radii, "rho": rho, "seed": seed})
        # calcules (uniforme, 4 rho) + gains (a rho*)
        for k in ("2", "3", "4"):
            S = calc["tasks"][task]["support"][k]["best"]["S"]
            add(f"calc_k{k}", S, [0.3, 0.5, 0.7, 0.9])
            g = calc["tasks"][task]["gains"][k]
            radii = [max(x, 0.0) ** 0.5 for x in g["g_star"]]
            add(f"calc_k{k}_g", S, [g["rho_star"]], radii)
        # elus
        if task == "narma25":
            P = res["adhoc"]["narma25"].get("prune", {})
            Bn = res["adhoc"]["narma25"].get("band", {})
            if P.get("done"):
                add("prune_elu", P["elu"], [0.5, 0.9])
            if Bn.get("done"):
                for rho in (0.5, 0.9):
                    bs = band_spec(Bn["elu"]["tau0"], Bn["elu"]["tau1"], rho)
                    add("band_elu", bs["taus"], [rho], bs["radii"])
        else:
            add("prune_elu", prune_elus[task], [0.5, 0.9])
            t0e, t1e = band_elus[task]
            for rho in (0.5, 0.9):
                bs = band_spec(t0e, t1e, rho)
                add("band_elu", bs["taus"], [rho], bs["radii"])
        add("unif_1_10", list(range(1, 11)), [0.5, 0.9])
        add("esn", [1], [0.5, 0.9])
    return specs


def analyse(res):
    agg = {}
    for task, cells in res["cells"].items():
        a = {}
        for key, v in cells.items():
            if "val" not in v:
                continue  # 14.2 : cellule ancienne (choix sur le test), exclue
            parts = key.split("|")
            tag = parts[0]; rho = parts[1][3:]; seed = parts[2][1:]
            a.setdefault(tag, {}).setdefault(rho, []).append((v["nrmse"], v["val"]))
        out = {}
        for tag, byrho in a.items():
            # 14.2 : best_rho choisi sur la validation ; mean/std rapportes =
            # NRMSE test au rho choisi.
            m = {rho: {"mean": float(np.mean([t for t, _ in x])),
                       "std": float(np.std([t for t, _ in x])),
                       "mean_val": float(np.mean([w for _, w in x])),
                       "std_val": float(np.std([w for _, w in x]))}
                 for rho, x in byrho.items()}
            best_rho = min(m, key=lambda r: m[r]["mean_val"])
            out[tag] = {"par_rho": m, "best_rho": best_rho,
                        "mean": m[best_rho]["mean"], "std": m[best_rho]["std"],
                        "mean_val": m[best_rho]["mean_val"]}
        agg[task] = out
    res["agg"] = agg


def main():
    path = P8OUT / "results_p8_conception.json"
    if path.exists():
        res = json.load(open(path))
    else:
        res = {"cells": {}, "agg": {}, "adhoc": {"narma25": {}}, "meta": {
            "budget": BUDGET, "seeds": list(N_SEEDS), "t_sbwh": SBWH_T,
            "splits_sbwh": [SBWH_WASH, SBWH_ITE, SBWH_T],
            "rho_calcules": [0.3, 0.5, 0.7, 0.9], "rho_baselines": [0.5, 0.9],
            "critere": "calcule a moins de 10 % de l'elu pruning sur >= 3/4 taches"}}
    t0 = time.time()
    # 1) recherches ad hoc narma25 (pruning puis bande)
    res["adhoc"].setdefault("narma25", {})
    adhoc_prune_narma25(res, t0)
    if time.time() - t0 > TIME_BUDGET:
        print("BUDGET apres prune25 : checkpoint"); return
    adhoc_band_narma25(res, t0)
    if time.time() - t0 > TIME_BUDGET:
        print("BUDGET apres band25 : checkpoint"); return
    # 2) runs de validation
    ensure_datasets()
    specs = build_specs(res)
    if not specs:
        analyse(res); save_p8(res, "conception")
        print("COMPLETE"); return
    with Pool(2) as pool:
        for out in pool.imap_unordered(run_one, specs):
            cells = res["cells"].setdefault(out["task"], {})
            cells[f"{out['tag']}|rho{out['rho']}|s{out['seed']}"] = {
                "nrmse": out["nrmse"], "val": out["val"],
                "lam": out["lam"], "N": out["N"]}
            analyse(res)
            save_p8(res, "conception")
            if time.time() - t0 > TIME_BUDGET:
                pool.terminate()
                print(f"BUDGET : checkpoint ({len(specs)} restants)")
                return
    analyse(res)
    res["meta"]["status"] = "COMPLETE"
    save_p8(res, "conception")
    print("COMPLETE")


if __name__ == "__main__":
    main()
