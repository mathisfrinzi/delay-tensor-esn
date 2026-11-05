"""Partie 8 : 8.2 : choisir S par le calcul (zero entrainement de reservoir).

1. Support : pour k in {2,3,4}, recherche exhaustive sur S inclus dans {1..60}
   minimisant C(S) = somme_{L in L*} w_L . h_S(L), h_S(L) = nombre minimal de
   sauts pour ecrire L avec S (infini si L inatteignable), par DP rendu de
   monnaie. L* et les poids viennent de 8.1 :
   - NARMA-30/25 : L* = {L : I_single > 5 % de max I}, w_L = I(L)/somme I ;
   - Silverbox/WH (retards mous) : L* = union des blocs de 3 avec
     I_block > 5 % de max I_block, w_L = I_block/3 normalises.
2. Gains : a S fixe, g_tau >= 0 avec somme g = rho^2, maximiser
   min_{L in L*} iota(L) (indice d'isolement 5.4, recurrence de renouvellement).
   Nelder-Mead sur le simplexe (parametrisation softmax), 3 departs,
   rho balaye dans {0.3, 0.5, 0.7, 0.9} ; on garde le rho qui maximise le
   critere. Rayon de la tranche tau ensuite sqrt(g_tau).
3. N = round(sqrt(400k/k)).

Prediction pre-enregistree (doc 8.2) : pour NARMA-30, S = {1, 30} ou {1, 15}.

Sortie : p8/results_p8_calcul.json.
"""
import numpy as np, json, itertools, os, sys
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p8_spectre import P8OUT, save_p8

LMAX = 60
BIG = 1e9
KS = [2, 3, 4]
RHOS = [0.3, 0.5, 0.7, 0.9]
BUDGET = 400_000


def l_star_from_spectre():
    """L* et poids par tache, depuis results_p8_spectre.json."""
    spec = json.load(open(P8OUT / "results_p8_spectre.json"))
    out = {}
    for task, a in spec["agg"].items():
        if task.startswith("narma"):
            I = np.array(a["I_single_mean"])
            sel = np.where(I > 0.05 * I.max())[0]
            w = np.maximum(I[sel], 0)
            w = w / w.sum()
            out[task] = {"L_star": [int(l) for l in sel],
                         "w": [float(x) for x in w], "source": "singles",
                         "I_raw": [float(I[l]) for l in sel]}
        else:
            B = np.array(a["I_block_mean"])
            blocks = [list(range(b, min(b + 3, 61))) for b in range(0, 61, 3)]
            sel = np.where(B > 0.05 * B.max())[0]
            L, w = [], []
            for i in sel:
                for l in blocks[i]:
                    L.append(l)
                    w.append(max(B[i], 0) / len(blocks[i]))
            w = np.array(w)
            w = w / w.sum() if w.sum() > 0 else w
            out[task] = {"L_star": L, "w": [float(x) for x in w],
                         "source": "blocks de 3",
                         "I_block_raw": [float(B[i]) for i in sel]}
    return out


def h_batch(taus_batch, lmax=LMAX):
    """(n_sets, k) -> (n_sets, lmax+1) : nb min de sauts (BIG si inatteignable)."""
    n = len(taus_batch)
    h = np.full((n, lmax + 1), BIG)
    h[:, 0] = 0.0
    rows = np.arange(n)[:, None]
    for L in range(1, lmax + 1):
        idx = L - taus_batch
        valid = idx >= 0
        cand = np.where(valid, h[rows, np.maximum(idx, 0)] + 1.0, BIG)
        h[:, L] = cand.min(1)
    return h


def best_support(L_star, w, k):
    """Exhaustive sur C(n,k) ensembles. Retourne (top, tableau (S, C))."""
    comb = np.array(list(itertools.combinations(range(1, LMAX + 1), k)))
    h = h_batch(comb)
    Ls = np.array(L_star)
    wv = np.array(w)
    cost = h[:, Ls] @ wv
    order = np.argsort(cost)
    top = [{"S": [int(x) for x in comb[i]], "C": float(cost[i]),
            "h": {int(L): float(h[i, L]) for L in L_star}} for i in order[:5]]
    return top


def e_gains(g, taus, lmax=LMAX):
    e = np.zeros(lmax + 1)
    e[0] = 1.0
    gt = dict(zip(taus, g))
    for L in range(1, lmax + 1):
        s = 0.0
        for tau, gv in gt.items():
            if tau <= L:
                s += gv * e[L - tau]
        e[L] = s
    return e


def iota(e, L, lmax=LMAX):
    num = e[L]
    den = 0.0
    for d in (-2, -1, 1, 2):
        if 0 <= L + d <= lmax:
            den += e[L + d]
    return num / den if den > 0 else np.inf


def min_iota(taus, g, L_star):
    e = e_gains(g, taus)
    return min(iota(e, L) for L in L_star)


def optimize_gains(taus, rho, L_star, starts):
    """Nelder-Mead sur le simplexe (softmax). Retourne (min_iota, g)."""
    K = len(taus)
    best = (-np.inf, None)
    for th0 in starts:
        def neg(theta):
            g = rho * rho * np.exp(theta - theta.max())
            g = g / g.sum() * (rho * rho)
            return -min_iota(taus, g, L_star)
        r = minimize(neg, th0, method="Nelder-Mead",
                     options={"maxiter": 3000, "xatol": 1e-10, "fatol": 1e-12})
        v = -r.fun
        if v > best[0]:
            g = rho * rho * np.exp(r.x - r.x.max())
            g = g / g.sum() * (rho * rho)
            best = (v, g)
    return best


def main():
    spec = l_star_from_spectre()
    res = {"tasks": {}, "meta": {"k": KS, "rhos": RHOS, "budget": BUDGET,
                                 "lmax": LMAX,
                                 "prediction_narma30": "S = {1, 30} ou {1, 15}"}}
    for task, info in spec.items():
        L_star, w = info["L_star"], info["w"]
        entry = {"L_star": L_star, "w": w, "source": info["source"],
                 "raw": {k: v for k, v in info.items() if k.startswith("I_")}}
        entry["support"] = {}
        for k in KS:
            top = best_support(L_star, w, k)
            entry["support"][str(k)] = {"best": top[0], "top5": top,
                                        "N": int(round((BUDGET / k) ** 0.5))}
        # gains sur le meilleur S de chaque k, rho balaye
        entry["gains"] = {}
        for k in KS:
            S = entry["support"][str(k)]["best"]["S"]
            # 3 departs : uniforme, one-hot sur chaque moitie, alterne
            starts = [np.zeros(k)]
            oh = np.zeros(k); oh[k // 2] = 4.0
            starts.append(oh)
            starts.append(np.array([2.0 * (-1) ** i for i in range(k)]))
            gk = {}
            for rho in RHOS:
                v, g = optimize_gains(S, rho, L_star, starts)
                gk[str(rho)] = {"min_iota": float(v), "g": [float(x) for x in g]}
            # uniforme pour reference
            for rho in RHOS:
                gu = np.array([rho * rho / k] * k)
                gk[str(rho)]["min_iota_uniforme"] = float(min_iota(S, gu, L_star))
            rho_star = max(RHOS, key=lambda r: gk[str(r)]["min_iota"])
            entry["gains"][str(k)] = {"S": S, "par_rho": gk,
                                     "rho_star": rho_star,
                                     "g_star": gk[str(rho_star)]["g"],
                                     "min_iota_star": gk[str(rho_star)]["min_iota"]}
        res["tasks"][task] = entry
        b2 = entry["support"]["2"]["best"]["S"]
        print(f"{task}: L*={L_star} S2={b2} S3={entry['support']['3']['best']['S']} "
              f"S4={entry['support']['4']['best']['S']}", flush=True)
    res["meta"]["status"] = "COMPLETE"
    save_p8(res, "calcul")
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    main()
