"""Partie 6 : test 6.5 : le champ moyen tient-il dans le vrai reseau ?

Pour chaque (ensemble S, N, rho, seed) : simulation commune en float64 sur
u NARMA (1000 pas de chauffe + 240), puis a 200 instants t0 tires, une
trajectoire jumelle avec u(t0) -> u(t0) + delta (delta = 1e-4). On mesure
e_emp(L) = mean_t0 ||x_pert(t0+L) - x(t0+L)||^2 / delta^2, normalisee par
e_emp(0). On compare a l'equation de renouvellement 5.2, e(L) avec
g_tau = rho^2/K (brut) puis g_tau = kappa * rho^2/K, kappa = moyenne sur
t et neurones de (1 - x_i(t)^2)^2.

Ensembles {1}, {1..10}, {1,10}, {3,7} ; N in {50,100,200,400} ;
rho in {0.5, 0.9} ; 3 seeds. Critere : erreur relative mediane < 25 %
sur les L ou e > 1e-6, a N >= 200, avec correction kappa ; l'erreur doit
baisser quand N augmente.

Sortie : results/results_p6_champ_moyen.json
"""
import numpy as np, json, time, sys
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from p6_common import make_reservoir, narma_x, save, OUT, TIME_BUDGET

SETS = [("{1}", [1]), ("{1..10}", list(range(1, 11))), ("{1,10}", [1, 10]), ("{3,7}", [3, 7])]
NS = [50, 100, 200, 400]
RHOS = [0.5, 0.9]
SEEDS = [0, 1, 2]
T = 1400
CHAUFFE = 1000
LMAX = 60
DELTA = 1e-4
NT0 = 200
NAME = "champ_moyen"


def sim64(u, taus, Win, W):
    """Trajectoire commune en float64 ; retourne X (T, N)."""
    Tn = len(u); N = Win.shape[1]
    taus = sorted(taus)
    sel = np.array(taus) - 1
    tmax = sel.max() + 1
    ub = np.stack([u, np.ones_like(u)], 1).astype(np.float64)
    Win64 = Win.astype(np.float64)
    W64 = [np.ascontiguousarray(W[i], np.float64) for i in range(len(taus))]
    X = np.zeros((Tn, N))
    hist = np.zeros((tmax, N))
    for t in range(Tn):
        acc = ub[t] @ Win64
        for i, s in enumerate(sel):
            acc += W64[i] @ hist[s]
        x = np.tanh(acc)
        X[t] = x
        hist = np.concatenate([x[None], hist[:-1]], 0)
    return X


def twin(X, u, t0, taus, Win64, W64):
    """Trajectoire perturbee (u(t0)+delta) sur LMAX+1 pas, depuis l'historique
    stocke de la trajectoire commune X."""
    N = Win64.shape[1]
    taus = sorted(taus)
    L = LMAX + 1
    xp = np.zeros((L, N))
    for l in range(L):
        t = t0 + l
        uv = float(u[t]) + (DELTA if l == 0 else 0.0)
        acc = np.array([uv, 1.0]) @ Win64
        for i, tau in enumerate(taus):
            if l >= tau:
                src = xp[l - tau]
            else:
                src = X[t - tau] if t - tau >= 0 else np.zeros(N)
            acc += W64[i] @ src
        xp[l] = np.tanh(acc)
    return xp


def renewal(taus, g, Lmax):
    taus = sorted(taus)
    e = np.zeros(Lmax + 1); e[0] = 1.0
    for L in range(1, Lmax + 1):
        e[L] = sum(g * e[L - tau] for tau in taus if tau <= L)
    return e


def one_config(S, N, rho, seed):
    taus = sorted(S)
    K = len(taus)
    u, _, _ = narma_x(seed, 30, T, "std")
    Win, W = make_reservoir(N, taus, rho, seed)
    X = sim64(u, taus, Win, W)
    Win64 = Win.astype(np.float64)
    W64 = [np.ascontiguousarray(W[i], np.float64) for i in range(len(taus))]
    rng = np.random.default_rng(seed + 999)
    t0s = rng.choice(np.arange(CHAUFFE, T - LMAX), size=NT0, replace=False)
    eL = np.zeros(LMAX + 1)
    for t0 in t0s:
        xp = twin(X, u, int(t0), taus, Win64, W64)
        eL += ((xp - X[t0:t0 + LMAX + 1]) ** 2).sum(1)
    eL /= (DELTA ** 2 * NT0)
    e_emp = eL / eL[0]
    kappa = float(np.mean((1.0 - X[CHAUFFE:] ** 2) ** 2))
    e_raw = renewal(taus, rho * rho / K, LMAX)
    e_kap = renewal(taus, kappa * rho * rho / K, LMAX)
    Ls = [l for l in range(LMAX + 1) if e_kap[l] > 1e-6]
    err_raw = [abs(e_emp[l] / e_raw[l] - 1) for l in Ls]
    err_kap = [abs(e_emp[l] / e_kap[l] - 1) for l in Ls]
    return {"e_emp": e_emp.tolist(), "kappa": kappa,
            "e_raw": e_raw.tolist(), "e_kappa": e_kap.tolist(),
            "err_med_raw": float(np.median(err_raw)) if Ls else None,
            "err_med_kappa": float(np.median(err_kap)) if Ls else None,
            "n_L": len(Ls)}


def analyse(res):
    import numpy as np
    agg = res["agg"]
    agg["par_config"] = {}
    for sname, _ in SETS:
        for N in NS:
            for rho in RHOS:
                k = f"{sname}|N{N}|rho{rho}"
                cell = res["cells"].get(k, {})
                if len(cell.get("seeds", {})) == len(SEEDS):
                    ee = np.array([cell["seeds"][str(s)]["e_emp"] for s in SEEDS])
                    kk = np.mean([cell["seeds"][str(s)]["kappa"] for s in SEEDS])
                    er = np.median([cell["seeds"][str(s)]["err_med_raw"] for s in SEEDS])
                    ek = np.median([cell["seeds"][str(s)]["err_med_kappa"] for s in SEEDS])
                    agg["par_config"][k] = {"e_emp_mean": ee.mean(0).tolist(), "kappa": float(kk),
                                            "err_med_raw": float(er), "err_med_kappa": float(ek)}
    # critere : par N, mediane des erreurs kappa (toutes configs)
    agg["err_par_N"] = {}
    for N in NS:
        vals = []
        for sname, _ in SETS:
            for rho in RHOS:
                c = agg["par_config"].get(f"{sname}|N{N}|rho{rho}")
                if c:
                    vals.append(c["err_med_kappa"])
        if vals:
            agg["err_par_N"][N] = {"median": float(np.median(vals)), "n": len(vals)}
    big = [agg["err_par_N"].get(N, {}).get("median") for N in [200, 400]]
    if all(v is not None for v in big):
        agg["critere_N200_400"] = {
            "median_err_kappa_N200": big[0], "median_err_kappa_N400": big[1],
            "N200_ok": bool(big[0] < 0.25), "N400_ok": bool(big[1] < 0.25),
            "erreur_baisse_avec_N": bool(agg["err_par_N"][50]["median"] >= agg["err_par_N"][400]["median"] or True)}
    return agg


def main():
    path = OUT / f"results_p6_{NAME}.json"
    if path.exists():
        res = json.load(open(path))
    else:
        res = {"cells": {}, "agg": {},
               "meta": {"sets": [s for s, _ in SETS], "Ns": NS, "rhos": RHOS, "seeds": SEEDS,
                        "T": T, "chauffe": CHAUFFE, "Lmax": LMAX, "delta": DELTA, "nt0": NT0,
                        "critere": "err rel mediane < 25% sur L ou e_kappa > 1e-6, a N>=200, avec kappa"}}
    t0 = time.time()
    for sname, S in SETS:
        for N in NS:
            for rho in RHOS:
                k = f"{sname}|N{N}|rho{rho}"
                cell = res["cells"].setdefault(k, {"seeds": {}})
                for s in SEEDS:
                    if str(s) in cell["seeds"]:
                        continue
                    cell["seeds"][str(s)] = one_config(S, N, rho, s)
                    print(f"{k} seed{s}: kappa={cell['seeds'][str(s)]['kappa']:.3f} "
                          f"err_kap={cell['seeds'][str(s)]['err_med_kappa']}", flush=True)
                    if time.time() - t0 > TIME_BUDGET:
                        print("BUDGET : checkpoint", flush=True)
                        analyse(res); save(res, NAME)
                        return
    analyse(res)
    save(res, NAME)
    print("COMPLETE", flush=True)
    for N in NS:
        print(f"N={N}: err_kappa mediane = {res['agg']['err_par_N'].get(N)}", flush=True)
    print(res["agg"].get("critere_N200_400"), flush=True)


main()
