"""Partie 4 : 4.1/4.2 : le terme d'ordre 2 minimal contre la profondeur.

Équation 4D minimale (une paire bilineaire, rapport/3_calculer_S.md §4.1) :
  x(t) = tanh( W_in [u(t),1] + sum_{tau in S} W_tau x(t-tau)
               + W2 ( x(t-tau1) o x(t-tau2) ) )
Budget 2x = 800 000 connexions recurrentes :
  esn800   : S={1}, N=894                     (N^2)
  3d800    : S={1..10}, N=283                 (10 N^2)
  3dstack  : 2 couches {1..10}, N=200 chacune (2 x 10 N^2) : 4.2
  4d_*     : S={1..10} + 1 paire, N=270       (11 N^2)
Paires : 4d_gen (1,8) ; 4d_att appariee a la tache : parite2/parite6/mc :
(10,11), narma30 : (1,30) ; 4d_ctrl (1,1). W2 : meme recette que W_tau
(rayon rho/sqrt(K+1)), tirage 6.0 : W_in, W_tau par tau croissant, W2.

Taches (protocole 6.0, T=5200, train [wash,4200), test [4200,5200), 5 seeds) :
  parite2/parite6 : XOR de la fenetre u(t-D-L+1..t-D), L=2/6, D=10, u binaire
  mc              : MC_k = correlation^2(u(t-k)), k=1..60, u iid U[0;0.5]
(amendement avant tout run : k<=20 sature a MC=20/20 pour toutes les
configs, smoke test : k=1..60 pour discriminer)
  narma30         : narma_x standard
Rhos : parite {0.5,0.9,0.99} ; mc {0.5,0.9} ; narma30 {0.3,0.5,0.9}.

Predictions pre-enregistrees : P1-P6 dans rapport/3_calculer_S.md.

Protocole corrige (14.2, 2026-11-02) : le meilleur rho est choisi sur la VALIDATION
(les 20 % finaux du bloc d'entrainement, readout ajuste sur les 80 % premiers), jamais
sur le test. Chaque cellule porte "val" (erreur de validation) et "nrmse"/"mc" (test) ;
les cellules sans "val" (ancien choix sur le test) sont ignorees et recalculees.

Sortie : p4/results_p4_tenseur.json {cells, agg, meta} (COMPLETE quand tout fait).
"""
import numpy as np, json, time, sys
from pathlib import Path
from multiprocessing import Pool

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from p6_common import (make_reservoir, run_states, augment, ridge_select,
                       nrmse_of, narma_x, LAMBDAS, SEEDS)

TIME_BUDGET = float(__import__("os").environ.get("TEMPS_MAX", "250"))
B2 = 800_000
T = 5200
I_TE = T - 1000
P4OUT = Path(__file__).resolve().parent.parent / "results"
P4OUT.mkdir(parents=True, exist_ok=True)
TAUS = list(range(1, 11))
TASKS = ["parite2", "parite6", "mc", "narma30"]
TAGS = ["esn800", "3d800", "3dstack", "4d_gen", "4d_att", "4d_ctrl"]
PAIR_ATT = {"parite2": (10, 11), "parite6": (10, 11), "mc": (10, 11),
            "narma30": (1, 30)}
RHOS = {"parite2": [0.5, 0.9, 0.99], "parite6": [0.5, 0.9, 0.99],
        "mc": [0.5, 0.9], "narma30": [0.3, 0.5, 0.9]}


def save_p4(results, name):
    results["meta"]["status"] = ("COMPLETE" if all(
        len(results["cells"].get(t, {})) >= len(TAGS) * len(RHOS[t]) * len(SEEDS)
        for t in TASKS) else "PARTIAL")
    json.dump(results, open(P4OUT / f"results_p4_{name}.json", "w", encoding="utf-8"), indent=1)


# reservoirs

def make_reservoir4(N, taus, pair, rho, seed):
    """3D + une paire bilineaire. Echelle commune rho/sqrt(K+1)."""
    rng = np.random.default_rng(seed)
    Win = rng.uniform(-0.5, 0.5, size=(2, N)).astype(np.float32)
    K = len(taus) + 1
    W = np.empty((len(taus), N, N), np.float32)
    for i, tau in enumerate(sorted(taus)):
        M = rng.normal(0, 1, (N, N)).astype(np.float32)
        r = max(np.abs(np.linalg.eigvals(M.astype(np.float64))).max(), 1e-12)
        W[i] = M * ((rho / np.sqrt(K)) / r)
    M2 = rng.normal(0, 1, (N, N)).astype(np.float32)
    r2 = max(np.abs(np.linalg.eigvals(M2.astype(np.float64))).max(), 1e-12)
    W2 = M2 * ((rho / np.sqrt(K)) / r2)
    return Win, W, W2


def run_states4(u, taus, Win, W, pair, W2):
    """run_states + terme W2 @ (x(t-tau1) * x(t-tau2))."""
    N = Win.shape[1]
    tmax = max(max(taus), max(pair))
    sel = np.array(sorted(taus)) - 1
    s1, s2 = pair[0] - 1, pair[1] - 1
    hist = np.zeros((tmax, N), np.float32)
    F = np.empty((len(u), N), np.float32)
    ub = np.stack([u, np.ones_like(u)], 1).astype(np.float32)
    WT = [np.ascontiguousarray(W[i]) for i in range(len(taus))]
    W2T = np.ascontiguousarray(W2)
    for t in range(len(u)):
        acc = ub[t] @ Win
        for i, s in enumerate(sel):
            acc += WT[i] @ hist[s]
        acc += W2T @ (hist[s1] * hist[s2])
        x_ = np.tanh(acc)
        F[t] = x_
        hist = np.concatenate([x_[None], hist[:-1]], axis=0)
    return F


def make_stack(seed, rho):
    """Deux couches {1..10} N=200. Tirage : Win1, W1 (tau croissant),
    Win2, W2 (tau croissant)."""
    rng = np.random.default_rng(seed)
    N = 200
    Win1 = rng.uniform(-0.5, 0.5, size=(2, N)).astype(np.float32)
    W1 = np.empty((10, N, N), np.float32)
    for i in range(10):
        M = rng.normal(0, 1, (N, N)).astype(np.float32)
        r = max(np.abs(np.linalg.eigvals(M.astype(np.float64))).max(), 1e-12)
        W1[i] = M * ((rho / np.sqrt(10)) / r)
    # entree couche 2 = [x1(t), 1], echelle 1/sqrt(N)
    Win2 = (rng.uniform(-1, 1, size=(N + 1, N)) / np.sqrt(N)).astype(np.float32)
    W2s = np.empty((10, N, N), np.float32)
    for i in range(10):
        M = rng.normal(0, 1, (N, N)).astype(np.float32)
        r = max(np.abs(np.linalg.eigvals(M.astype(np.float64))).max(), 1e-12)
        W2s[i] = M * ((rho / np.sqrt(10)) / r)
    return Win1, W1, Win2, W2s


def run_states_stack(u, Win1, W1, Win2, W2s):
    """Couche 1 sur u, couche 2 sur x1(t) ; retourne [F1, F2]."""
    F1 = run_states(u, TAUS, Win1, W1)
    N = Win1.shape[1]
    ub2 = np.hstack([F1, np.ones((len(u), 1), np.float32)])
    tmax = 10
    hist = np.zeros((tmax, N), np.float32)
    F2 = np.empty((len(u), N), np.float32)
    WT = [np.ascontiguousarray(W2s[i]) for i in range(10)]
    sel = np.arange(10)
    for t in range(len(u)):
        acc = ub2[t] @ Win2
        for i, s in enumerate(sel):
            acc += WT[i] @ hist[s]
        x_ = np.tanh(acc)
        F2[t] = x_
        hist = np.concatenate([x_[None], hist[:-1]], axis=0)
    return np.hstack([F1, F2])


# taches

def parity_gen(seed, L, D=10):
    rng = np.random.default_rng(seed)
    u = rng.integers(0, 2, size=T).astype(np.float32)
    y = np.zeros(T, np.float64)
    cs = np.concatenate([[0.0], np.cumsum(u)])
    for t in range(D + L - 1, T):
        y[t] = (cs[t - D + 1] - cs[t - D + 1 - L]) % 2
    return u, y


def mc_gen(seed):
    rng = np.random.default_rng(seed + 777)
    return rng.uniform(0, 0.5, size=T).astype(np.float32)


def task_target(task, seed):
    """Retourne (u, y) : pour mc, y=None (cibles u(t-k) implicites)."""
    if task == "parite2":
        return parity_gen(seed, 2)
    if task == "parite6":
        return parity_gen(seed, 6)
    if task == "mc":
        return mc_gen(seed), None
    if task == "narma30":
        u, y, gen = narma_x(seed, 30)
        return u, y
    raise ValueError(task)


# un run

def run_one(spec):
    task, tag, rho, seed = spec["task"], spec["tag"], spec["rho"], spec["seed"]
    u, y = task_target(task, seed)
    if tag == "esn800":
        N = 894
        Win, W = make_reservoir(N, [1], rho, seed)
        F = run_states(u, [1], Win, W)
        gen = None
    elif tag == "3d800":
        N = 283
        Win, W = make_reservoir(N, TAUS, rho, seed)
        F = run_states(u, TAUS, Win, W)
        gen = None
    elif tag == "3dstack":
        N = 200
        Win1, W1, Win2, W2s = make_stack(seed, rho)
        F = run_states_stack(u, Win1, W1, Win2, W2s)
        gen = None
    else:  # 4d_gen / 4d_att / 4d_ctrl
        N = 270
        if tag == "4d_gen":
            pair = (1, 8)
        elif tag == "4d_att":
            pair = PAIR_ATT[task]
        else:
            pair = (1, 1)
        Win, W, W2 = make_reservoir4(N, TAUS, pair, rho, seed)
        F = run_states4(u, TAUS, Win, W, pair, W2)
        gen = None
    if task == "narma30":
        gen = None
    wash = max(200, 3 * max(TAUS))
    out = {"task": task, "tag": tag, "rho": rho, "seed": seed, "N": N}
    if task == "mc":
        # MC_k = correlation^2 sur le test, k=1..20 ; mc = somme
        Ftr = augment(F[wash:I_TE].astype(np.float64))
        Fte = augment(F[I_TE:].astype(np.float64))
        per_k, val_k = [], []
        m = int(0.8 * len(Ftr))
        for k in range(1, 61):
            ztr = u[wash - k:I_TE - k].astype(np.float64)
            zte = u[I_TE - k:T - k].astype(np.float64)
            pred, pval = ridge_mc(Ftr, ztr, Fte)
            a = pred - pred.mean(); b = zte - zte.mean()
            r = float((a @ b) / np.sqrt((a @ a) * (b @ b) + 1e-30))
            per_k.append(max(r * r, 0.0))
            zv = ztr[m:]
            av = pval - pval.mean(); bv = zv - zv.mean()
            rv = float((av @ bv) / np.sqrt((av @ av) * (bv @ bv) + 1e-30))
            val_k.append(max(rv * rv, 0.0))
        out["mc"] = float(sum(per_k))
        out["per_k"] = per_k
        out["mc_val"] = float(sum(val_k))
    else:
        Ftr = augment(F[wash:I_TE]); ztr = y[wash:I_TE]
        Fte = augment(F[I_TE:]); zte = y[I_TE:]
        pred, lam, val, _ = ridge_select(Ftr, ztr, Fte)
        out["nrmse"] = nrmse_of(pred, zte)
        out["lam"] = lam
        out["val"] = val
    return out


def ridge_mc(Ftr, ztr, Fte):
    """Ridge avec lambda choisi sur les 20 % finaux du train (protocole 6.0)."""
    n = len(Ftr)
    m = int(0.8 * n)
    A_main = Ftr[:m].T @ Ftr[:m]
    A_full = Ftr.T @ Ftr
    eye = np.eye(Ftr.shape[1])
    zm = ztr[:m]
    best = (np.inf, None)
    for lam in LAMBDAS:
        Wo = np.linalg.solve(A_main + lam * eye, Ftr[:m].T @ zm)
        v = float(np.sqrt(np.mean((Ftr[m:] @ Wo - ztr[m:]) ** 2)))
        if v < best[0]:
            best = (v, lam)
    Wo = np.linalg.solve(A_full + best[1] * eye, Ftr.T @ ztr)
    Wm = np.linalg.solve(A_main + best[1] * eye, Ftr[:m].T @ zm)
    return Fte @ Wo, Ftr[m:] @ Wm      # (prediction test, prediction validation)


# orchestration

def build_specs(res):
    specs = []
    for task in TASKS:
        cells = res["cells"].get(task, {})
        for tag in TAGS:
            for rho in RHOS[task]:
                for seed in SEEDS:
                    key = f"{tag}|rho{rho}|s{seed}"
                    c = cells.get(key)
                    if c is None or ("val" not in c and "mc_val" not in c):
                        specs.append({"task": task, "tag": tag,
                                      "rho": rho, "seed": seed})
    return specs


def analyse(res):
    agg = {}
    for task, cells in res["cells"].items():
        a = {}
        for key, v in cells.items():
            metric, vmetric = ("mc", "mc_val") if task == "mc" else ("nrmse", "val")
            if vmetric not in v:
                continue          # 14.2 : ancienne cellule (rho choisi sur le test), exclue
            parts = key.split("|")
            a.setdefault(parts[0], {}).setdefault(parts[1][3:], []).append((v[metric], v[vmetric]))
        out = {}
        for tag, byrho in a.items():
            m = {rho: {"mean": float(np.mean([t for t, _ in x])), "std": float(np.std([t for t, _ in x])),
                       "mean_val": float(np.mean([w for _, w in x]))}
                 for rho, x in byrho.items()}
            # 14.2 : rho choisi sur la validation ; mean/std rapportes = test a ce rho
            best = (max if task == "mc" else min)(m, key=lambda r: m[r]["mean_val"])
            out[tag] = {"par_rho": m, "best_rho": best,
                        "mean": m[best]["mean"], "std": m[best]["std"],
                        "mean_val": m[best]["mean_val"]}
        agg[task] = out
    res["agg"] = agg


def main():
    path = P4OUT / "results_p4_tenseur.json"
    if path.exists():
        res = json.load(open(path, encoding="utf-8-sig"))
    else:
        res = {"cells": {}, "agg": {}, "meta": {
            "budget": B2, "seeds": list(SEEDS), "T": T, "test": [I_TE, T],
            "taus": TAUS, "pairs_att": {k: list(v) for k, v in PAIR_ATT.items()},
            "rhos": RHOS, "tags": TAGS,
            "accounting": "3dstack: poids inter-couches non comptes (comme W_in)",
            "predictions": "P1-P6 dans rapport/3_calculer_S.md (pre-enregistrees)"}}
    t0 = time.time()
    specs = build_specs(res)
    if not specs:
        analyse(res); save_p4(res, "tenseur"); print("COMPLETE"); return
    print(f"p4_tenseur : {len(specs)} cellules a calculer, 2 processus ({time.strftime('%H:%M:%S')})", flush=True)
    done = 0
    with Pool(2) as pool:
        for out in pool.imap_unordered(run_one, specs):
            done += 1
            el = time.time() - t0
            print(f"  [{done}/{len(specs)}] {out['task']} {out['tag']} rho={out['rho']} seed={out['seed']}"
                  f" | {el/60:.1f} min ecoulees, reste ~{el/done*(len(specs)-done)/60:.0f} min", flush=True)
            cells = res["cells"].setdefault(out["task"], {})
            key = f"{out['tag']}|rho{out['rho']}|s{out['seed']}"
            cell = {"N": out["N"]}
            if "nrmse" in out:
                cell["nrmse"] = out["nrmse"]; cell["lam"] = out["lam"]
                cell["val"] = out["val"]
            else:
                cell["mc"] = out["mc"]; cell["per_k"] = out["per_k"]; cell["mc_val"] = out["mc_val"]
            cells[key] = cell
            analyse(res)
            save_p4(res, "tenseur")
            if time.time() - t0 > TIME_BUDGET:
                pool.terminate()
                print(f"BUDGET : checkpoint ({len(specs)} restants)")
                return
    analyse(res)
    save_p4(res, "tenseur")
    print("COMPLETE")


if __name__ == "__main__":
    main()
