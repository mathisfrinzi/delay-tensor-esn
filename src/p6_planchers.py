"""Partie 6 : test 6.7 : planchers d'un oracle sans reservoir.

Question : quel est le NRMSE plancher sur NARMA-30 d'un readout ridge qui
connait EXACTEMENT des retards de u (et leurs produits 2 a 2), mais n'a
aucun etat ?
  L1 = {0..60}          : tous les retards utiles -> prediction < 0.15
  L2 = L1 \\ {1}         : sans u(t-1)            -> ~0.75 +- 0.05
  L3 = L1 \\ {1, 30}     : sans u(t-1) ni u(t-30) -> ~1.0
Deux versions de features :
  prod : [u(t-a)]_{a in L} + [u(t-a)u(t-b)]_{a<b in L} + 1   (1892 pour L1)
  tanh : 2000 neurones aleatoires tanh(X_delays R^T + b) + 1 (couche cachee
         aleatoire fixee par seed, R N(0,1), b U[-0.5,0.5], X normalise par
         std des delays).
T=50000, washout 500, train [500, 40000), test [40000, 50000), 5 seeds,
ridge protocole 6.0 (lambda sur 20 % finaux du train, retrain complet).
Gram accumule par chunks (jamais le X complet en memoire).

Sortie : p6/results_p6_planchers.json {cells, agg, meta}.
"""
import numpy as np, time, sys
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from p6_common import narma_x, nrmse_of, LAMBDAS, SEEDS, TIME_BUDGET, save, OUT

T = 50_000
WASH = 500
I_TE = 40_000
LAGSETS = {"L1": list(range(0, 61)),
           "L2": [a for a in range(0, 61) if a != 1],
           "L3": [a for a in range(0, 61) if a not in (1, 30)]}
VERSIONS = ["prod", "tanh"]
NTANH = 2000
CH = 4000  # lignes par chunk


def delays_block(u, lags, t0, t1):
    """(t1-t0, |L|) : u(t-a) pour t in [t0,t1)."""
    B = np.empty((t1 - t0, len(lags)), np.float64)
    for ia, a in enumerate(lags):
        B[:, ia] = u[t0 - a:t1 - a]
    return B


def build_rows(u, lags, t0, t1, version, rngt=None):
    Xd = delays_block(u, lags, t0, t1)
    if version == "prod":
        P = Xd[:, :, None] * Xd[:, None, :]
        iu, iv = np.triu_indices(len(lags), 1)
        X = np.concatenate([Xd, P[:, iu, iv]], 1)
    else:
        X = np.tanh(Xd @ rngt["R"][:, lags].T + rngt["b"])
    return np.column_stack([X, np.ones(len(X))])


def gram_and_solve(u, lags, y, version, rngt=None):
    """Ridge chunked : retourne (nrmse_test, lam, n_features)."""
    d = (len(lags) + len(lags) * (len(lags) - 1) // 2 + 1) if version == "prod" \
        else (NTANH + 1)
    # bornes du train
    tr0, tr1 = WASH, I_TE
    m = int(0.8 * (tr1 - tr0)) + tr0  # train_main = [tr0, m), val = [m, tr1)
    A_main = np.zeros((d, d)); b_main = np.zeros(d)
    A_full = np.zeros((d, d)); b_full = np.zeros(d)
    zval = None; Xval = None
    for c0 in range(tr0, tr1, CH):
        c1 = min(c0 + CH, tr1)
        X = build_rows(u, lags, c0, c1, version, rngt)
        z = y[c0:c1]
        if c1 <= m:
            A_main += X.T @ X; b_main += X.T @ z
        else:
            if c0 < m:  # chunk a cheval : couper
                Xa = X[:m - c0]; za = z[:m - c0]
                A_main += Xa.T @ Xa; b_main += Xa.T @ za
                X = X[m - c0:]; z = z[m - c0:]
            if Xval is None:
                Xval = X; zval = z
            else:
                Xval = np.concatenate([Xval, X], 0); zval = np.concatenate([zval, z])
    # A_full = A_main (boucle 1) + blocs val [m, tr1) (boucle 2)
    for c0 in range(m, tr1, CH):
        c1 = min(c0 + CH, tr1)
        X = build_rows(u, lags, c0, c1, version, rngt)
        z = y[c0:c1]
        A_full += X.T @ X; b_full += X.T @ z
    A_full += A_main
    b_full += b_main
    eye = np.eye(d)
    best = (np.inf, None)
    for lam in LAMBDAS:
        Wo = np.linalg.solve(A_main + lam * eye, b_main)
        v = float(np.sqrt(np.mean((Xval @ Wo - zval) ** 2)))
        if v < best[0]:
            best = (v, lam)
    lam = best[1]
    Wo = np.linalg.solve(A_full + lam * eye, b_full)
    zte = y[I_TE:T]
    pred = np.empty(len(zte))
    for c0 in range(I_TE, T, CH):
        c1 = min(c0 + CH, T)
        X = build_rows(u, lags, c0, c1, version, rngt)
        pred[c0 - I_TE:c1 - I_TE] = X @ Wo
    return nrmse_of(pred, zte), lam, d


def main():
    import json
    from pathlib import Path
    path = OUT / "results_p6_planchers.json"
    if path.exists():
        res = json.load(open(path))
    else:
        res = {"cells": {v: {L: {} for L in LAGSETS} for v in VERSIONS},
               "agg": {}, "meta": {"T": T, "washout": WASH, "i_te": I_TE,
                                   "lagsets": LAGSETS, "ntanh": NTANH,
                                   "lambda_grid": LAMBDAS,
                                   "predictions": {"L1": "<0.15", "L2": "0.75+-0.05",
                                                   "L3": "~1.0"}}}
    t0 = time.time()
    done = 0
    for seed in SEEDS:
        u, y, gen = narma_x(seed, 30, T, "std")
        for version in VERSIONS:
            rngt = None
            if version == "tanh":
                r = np.random.default_rng(seed + 777)
                R = r.normal(0, 1, (NTANH, 61))
                b = r.uniform(-0.5, 0.5, NTANH)
                rngt = {"R": R, "b": b}
            for Lname, lags in LAGSETS.items():
                cell = res["cells"][version][Lname]
                if str(seed) in cell:
                    continue
                nrmse, lam, d = gram_and_solve(u, lags, y, version, rngt)
                cell[str(seed)] = {"nrmse": nrmse, "lam": lam, "d": d, "gen": gen}
                save(res, "planchers")
                done += 1
                print(f"{version}/{Lname}/seed{seed}: NRMSE={nrmse:.4f} lam={lam:g} d={d}", flush=True)
                if time.time() - t0 > TIME_BUDGET:
                    print("BUDGET : checkpoint save, relancer", flush=True)
                    return
    # agregation
    for version in VERSIONS:
        for Lname in LAGSETS:
            vals = [res["cells"][version][Lname][str(s)]["nrmse"] for s in SEEDS
                    if str(s) in res["cells"][version][Lname]]
            if len(vals) == len(SEEDS):
                res["agg"][f"{version}/{Lname}"] = {
                    "mean": float(np.mean(vals)), "std": float(np.std(vals)),
                    "n_seeds": len(vals)}
                print(f"AGG {version}/{Lname}: {np.mean(vals):.4f} +- {np.std(vals):.4f}", flush=True)
    save(res, "planchers")
    print("COMPLETE", flush=True)


main()
