"""Partie 4 : 4.3 : le rang du terme d'ordre 2 (CP factorise, budget 800k).

Terme bilineaire CP de rang R par paire :
  out_i = sum_r A[i,r] (B_r . x(t-tau1)) (C_r . x(t-tau2))
Calibrage : B, C ~ N(0, 1/sqrt(N)) ; A ~ N(0, rho/sqrt(11 R)) pour une paire
(magnitude ~ rho/sqrt(K+1), comme le dense). A budget strictement egal 800k :
  4d_cp8     : R=8,  N=281, une paire appariee      10 N^2 + 3RN
  4d_cp32    : R=32, N=277, une paire appariee
  4d_cp16x6  : 6 paires (1,5),(1,10),(1,15),(1,20),(1,25),(1,30),
               rang 16 chacune, N=268               10 N^2 + 6*3*16*N
Taches : parite2 (paire (10,11)), narma30 (paire (1,30) / 6 paires).
Rhos : parite2 {0.5,0.9,0.99}, narma30 {0.3,0.5,0.9}. 5 seeds.
Predictions P7/P8 dans rapport/3_calculer_S.md. Sortie : p4/results_p4_tenseur.json
(memes cells, tags 4d_cp*).
"""
import numpy as np, json, time, sys
from pathlib import Path
from multiprocessing import Pool

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from p6_common import (make_reservoir, run_states, augment, ridge_select,
                       nrmse_of, narma_x, SEEDS)
from p4_tenseur import (P4OUT, TAUS, T, I_TE, task_target, analyse, save_p4,
                        TIME_BUDGET, B2)

PAIR_ATT = {"parite2": [(10, 11)], "narma30": [(1, 30)]}
PAIRS6 = [(1, 5), (1, 10), (1, 15), (1, 20), (1, 25), (1, 30)]
TAGS = {"parite2": ["4d_cp8", "4d_cp32"], "narma30": ["4d_cp8", "4d_cp32", "4d_cp16x6"]}
RHOS = {"parite2": [0.5, 0.9, 0.99], "narma30": [0.3, 0.5, 0.9]}
CFG = {"4d_cp8": {"R": 8, "N": 281}, "4d_cp32": {"R": 32, "N": 277},
       "4d_cp16x6": {"R": 16, "N": 268}}


def make_reservoir_cp(N, taus, pairs, R, rho, seed):
    """3D standard (rho/sqrt(K)) + R par paire : B, C ~ N(0,1/sqrt(N)),
    A ~ N(0, rho/sqrt((K+P) R)). Tirage 6.0 : W_in, W_tau, puis par paire
    B, C, A."""
    rng = np.random.default_rng(seed)
    Win = rng.uniform(-0.5, 0.5, size=(2, N)).astype(np.float32)
    K = len(taus)
    P = len(pairs)
    W = np.empty((K, N, N), np.float32)
    for i, tau in enumerate(sorted(taus)):
        M = rng.normal(0, 1, (N, N)).astype(np.float32)
        r = max(np.abs(np.linalg.eigvals(M.astype(np.float64))).max(), 1e-12)
        W[i] = M * ((rho / np.sqrt(K)) / r)
    cp = []
    sa = rho / np.sqrt((K + P) * R)
    for p in pairs:
        B = (rng.normal(0, 1 / np.sqrt(N), (R, N))).astype(np.float32)
        C = (rng.normal(0, 1 / np.sqrt(N), (R, N))).astype(np.float32)
        A = (rng.normal(0, sa, (N, R))).astype(np.float32)
        cp.append((A, B, C))
    return Win, W, cp


def run_states_cp(u, taus, Win, W, pairs, cp):
    N = Win.shape[1]
    tmax = max(max(taus), max(max(p) for p in pairs))
    sel = np.array(sorted(taus)) - 1
    selP = [(a - 1, b - 1) for a, b in pairs]
    hist = np.zeros((tmax, N), np.float32)
    F = np.empty((len(u), N), np.float32)
    ub = np.stack([u, np.ones_like(u)], 1).astype(np.float32)
    WT = [np.ascontiguousarray(W[i]) for i in range(len(taus))]
    CT = [(np.ascontiguousarray(A), np.ascontiguousarray(B),
           np.ascontiguousarray(C)) for A, B, C in cp]
    for t in range(len(u)):
        acc = ub[t] @ Win
        for i, s in enumerate(sel):
            acc += WT[i] @ hist[s]
        for (A, B, C), (s1, s2) in zip(CT, selP):
            acc += A @ ((B @ hist[s1]) * (C @ hist[s2]))
        x_ = np.tanh(acc)
        F[t] = x_
        hist = np.concatenate([x_[None], hist[:-1]], axis=0)
    return F


def run_one(spec):
    task, tag, rho, seed = spec["task"], spec["tag"], spec["rho"], spec["seed"]
    N = CFG[tag]["N"]
    R = CFG[tag]["R"]
    pairs = PAIRS6 if tag == "4d_cp16x6" else PAIR_ATT[task]
    u, y = task_target(task, seed)
    Win, W, cp = make_reservoir_cp(N, TAUS, pairs, R, rho, seed)
    F = run_states_cp(u, TAUS, Win, W, pairs, cp)
    wash = max(200, 3 * max(TAUS))
    Ftr = augment(F[wash:I_TE]); ztr = y[wash:I_TE]
    Fte = augment(F[I_TE:]); zte = y[I_TE:]
    pred, lam, val, _ = ridge_select(Ftr, ztr, Fte)
    return {"task": task, "tag": tag, "rho": rho, "seed": seed, "N": N,
            "nrmse": nrmse_of(pred, zte), "lam": lam, "val": val}


def main():
    path = P4OUT / "results_p4_tenseur.json"
    res = json.load(open(path, encoding="utf-8-sig"))
    res.setdefault("cells", {})
    t0 = time.time()
    specs = []
    for task in ("parite2", "narma30"):
        cells = res["cells"].get(task, {})
        for tag in TAGS[task]:
            for rho in RHOS[task]:
                for seed in SEEDS:
                    key = f"{tag}|rho{rho}|s{seed}"
                    if key not in cells or "val" not in cells[key]:   # 14.2 : ancienne cellule (rho choisi sur le test) recalculee
                        specs.append({"task": task, "tag": tag,
                                      "rho": rho, "seed": seed})
    if not specs:
        analyse(res); save_p4(res, "tenseur"); print("COMPLETE"); return
    print(f"p4_cp : {len(specs)} cellules a calculer ({time.strftime('%H:%M:%S')})", flush=True)
    done = 0
    with Pool(2) as pool:
        for out in pool.imap_unordered(run_one, specs):
            cells = res["cells"].setdefault(out["task"], {})
            cells[f"{out['tag']}|rho{out['rho']}|s{out['seed']}"] = {
                "nrmse": out["nrmse"], "lam": out["lam"], "N": out["N"], "val": out["val"]}
            done += 1
            el = time.time() - t0
            print(f"  [{done}/{len(specs)}] {out['task']} {out['tag']} rho={out['rho']} seed={out['seed']}"
                  f" | {el/60:.1f} min, reste ~{el/done*(len(specs)-done)/60:.0f} min", flush=True)
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
