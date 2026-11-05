"""Partie 6.1 : Les trous de memoire sont-ils exactement vides ? (sanite)

S : {1..10}, {9,10}, {3,7}, {2,5}, {4,6,9}, {2,4,6,8,10}, {16..20} ;
rho in {0.5, 0.9} ; u iid U[0;0.5] ; T=12000 (washout 1000, train 8000,
test 3000) ; 5 seeds. Cibles lineaires z(t)=u(t-k), k=1..40, capacite
MC_k = r^2 de Pearson sur le test. Plancher de bruit : 40 cibles nulles iid
independantes, traitees pareil ; seuil = 99e centile toutes configs et seeds
confondues. Produits u(t-a)u(t-b), 1<=a<b<=20, pour {3,7} et {2,5} seulement.

Predictions (pre-enregistrees, rapport/2_regle.md) : MC_k sous le seuil pour tout
k hors <S> (ex. {3,7} : k=1,2,4,5,8,11 ; {16..20} : k=1..15 et 21..31) ;
MC_k decroissant avec le nombre minimal de sauts ; produits non nuls seulement
si a ET b dans <S>. Falsification : un seul k non representable au-dessus du
seuil dans >= 2 seeds = bug.
Checkpoint : p6/results_p6_trous.json."""
import sys, time, json
import numpy as np
from p6_common import (OUT, SEEDS, BUDGET, LAMBDAS, TIME_BUDGET,
                       make_reservoir, run_states, augment, semigroup, save)

T, WASH, NTE = 12000, 1000, 3000
I_TR0, I_TR1 = WASH, T - NTE          # train [1000, 9000)
M = int(0.8 * (I_TR1 - I_TR0))        # main 6400 / val 1600
KMAX = 40
ENSEMBLES = [list(range(1, 11)), [9, 10], [3, 7], [2, 5], [4, 6, 9],
             [2, 4, 6, 8, 10], list(range(16, 21))]
PROD_SETS = [[3, 7], [2, 5]]
RHOS = [0.5, 0.9]

name = "trous"
RJ = OUT / f"results_p6_{name}.json"
res = json.load(open(RJ)) if RJ.exists() else {"cells": {}, "meta": {}}
cells = res.setdefault("cells", {})
t0 = time.time()

for S in ENSEMBLES:
    sname = "S" + "{" + ",".join(map(str, S)) + "}"
    for rho in RHOS:
        key = f"{sname}|{rho}"
        cell = cells.setdefault(key, {"S": S, "rho": rho, "seeds": {}})
        att40, h40 = semigroup(S, KMAX)
        att20, _ = semigroup(S, 20)
        for seed in SEEDS:
            if str(seed) in cell["seeds"]:
                continue
            if time.time() - t0 > TIME_BUDGET:
                save(res, name); print("time budget : reprendre", flush=True)
                sys.exit(0)
            N = int(round((BUDGET / len(S)) ** 0.5))
            Win, W = make_reservoir(N, S, rho, seed)
            rng = np.random.default_rng(seed + 12345)
            u = rng.uniform(0, 0.5, size=T).astype(np.float32)
            F = run_states(u, S, Win, W)
            Ftr = augment(F[I_TR0:I_TR1]); Fte = augment(F[I_TR1:])
            d = Ftr.shape[1]; eye = np.eye(d)
            A_main = Ftr[:M].T @ Ftr[:M]
            A_full = Ftr.T @ Ftr

            # retards u(t-k) et produits
            Zd = np.zeros((KMAX + 1, T), np.float32)
            Zd[0] = u
            for k in range(1, KMAX + 1):
                Zd[k, k:] = u[:T - k]
            targets = {f"k{k}": (Zd[k, I_TR0:I_TR1].astype(np.float64),
                                 Zd[k, I_TR1:].astype(np.float64))
                       for k in range(1, KMAX + 1)}
            # 40 cibles nulles iid (meme loi, independantes de u)
            for j in range(40):
                v = np.random.default_rng(10_000 + seed * 100 + j).uniform(0, 0.5, T)
                targets[f"n{j}"] = (v[I_TR0:I_TR1].astype(np.float64),
                                   v[I_TR1:].astype(np.float64))
            prods = {}
            if S in PROD_SETS:
                for a in range(1, 21):
                    for b in range(a + 1, 21):
                        z = (Zd[a] * Zd[b]).astype(np.float64)
                        prods[f"{a}_{b}"] = (z[I_TR0:I_TR1], z[I_TR1:])
            targets.update({f"p{k}": v for k, v in prods.items()})

            mc = {}
            for tn, (ztr, zte) in targets.items():
                best = (np.inf, None)
                for lam in LAMBDAS:
                    Wo = np.linalg.solve(A_main + lam * eye, Ftr[:M].T @ ztr[:M])
                    v = float(np.sqrt(np.mean((Ftr[M:] @ Wo - ztr[M:]) ** 2)))
                    if v < best[0]:
                        best = (v, lam)
                Wo = np.linalg.solve(A_full + best[1] * eye, Ftr.T @ ztr)
                pred = Fte @ Wo
                sz = zte.std()
                if sz > 0 and pred.std() > 0:
                    r = float(np.corrcoef(pred, zte)[0, 1])
                else:
                    r = 0.0
                mc[tn] = max(r * r, 0.0)

            cell["seeds"][str(seed)] = {
                "N": N,
                "mc": [mc[f"k{k}"] for k in range(1, KMAX + 1)],
                "nulls": [mc[f"n{j}"] for j in range(40)],
                "products": {k: v for k, v in mc.items() if k.startswith("p")},
            }
            save(res, name)
            print(f"{key} seed{seed} N={N} done ({time.time()-t0:.0f}s)", flush=True)

# agregats
nulls_all = []
for key, cell in cells.items():
    for s, sd in cell["seeds"].items():
        nulls_all.extend(sd["nulls"])
if nulls_all:
    res["agg"] = {"seuil_bruit_p99": float(np.percentile(nulls_all, 99)),
                  "n_nulls": len(nulls_all)}
    # verdict trous : pour chaque config, k hors <S> au-dessus du seuil ?
    seuil = res["agg"]["seuil_bruit_p99"]
    verdicts = {}
    for key, cell in cells.items():
        att40, _ = semigroup(cell["S"], KMAX)
        holes_bad = []
        for s, sd in cell["seeds"].items():
            for k in range(1, KMAX + 1):
                if not att40[k] and sd["mc"][k - 1] > seuil:
                    holes_bad.append((s, k))
        verdicts[key] = {"fuites": holes_bad, "n_fuites": len(holes_bad)}
    res["agg"]["verdicts_trous"] = verdicts
    save(res, name)
print(f"COMPLETE : seuil p99 = {res.get('agg', {}).get('seuil_bruit_p99')}")
