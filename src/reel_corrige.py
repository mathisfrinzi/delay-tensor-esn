"""EXPLORATOIRE (ETTh1 déjà ouverte) : diagnostic corrigé par familles périodiques, puis recette avec ce S.
Diagnostic sur l'entraînement seul : pour chaque période P (2..lmax//2), erreur de validation d'un ridge sur
{0,1,2} ∪ {kP ≤ lmax} ; P* = argmin. Poids w : moitié sur {1,2}, moitié sur les multiples de P*.
S = sauts K = 2, 3, 5 de ces poids ; recette = même recherche que reel_nouveau, avec ces S.
python src/reel_corrige.py etth1_h24 [seed_min seed_max]
"""
import json, sys, time
import numpy as np
import reel_nouveau as RN
import conception as C, p12_synthese as P13, reservoir as R, p7_common as P7

PROTO = "reel_corrige_explo"
FICH = RN.RES / "results_reel_corrige.json"


def periode(d):
    s, lmax = d["splits"], d["lmax"]
    U = d["U"][:s["tr_end"], 0].astype(np.float64)
    y = d["y"][:s["tr_end"]]
    T = len(U)
    v0 = s["wash"] + int(0.8 * (T - s["wash"]))
    best = None
    for P in range(2, lmax // 2 + 1):
        lags = sorted({0, 1, 2} | set(range(P, lmax + 1, P)))
        F = np.zeros((T, len(lags)), np.float32)
        for j, k in enumerate(lags):
            F[k:, j] = U[:T - k]
        v = P7.ridge_readout(F, y, s["wash"], v0, v0, T)[0]
        if best is None or v < best[0]:
            best = (v, P)
    return best[1], best[0]


def main(tache, s0, s1):
    d = RN.charger_reel(tache)
    lmax = d["lmax"]
    P, vP = periode(d)
    w = np.zeros(lmax + 1)
    w[1] = w[2] = 0.25
    mult = list(range(P, lmax + 1, P))
    for k in mult:
        w[k] = 0.5 / len(mult)
    SS = {f"sauts K={K}": [int(t) for t in C.support_sauts(w, K, lmax, lmax)[0]] for K in (2, 3, 5)}
    print("période", P, "val", round(vP, 4), "S", SS, flush=True)
    out = R.lire(FICH)
    out["meta"] = {"proto": PROTO, "periode": P, "val_periode": vP, "S": SS, "exploratoire": True}
    rng = np.random.default_rng(12_000)
    tir = [{"rho": float(rng.uniform(0.1, 1.1)), "fuite": float(rng.uniform(0.05, 1.0)),
            "echelle": float(np.exp(rng.uniform(np.log(0.05), np.log(2.0))))} for _ in range(10)]
    cands = [dict(t, S=S, kind="perm", paire=None, origine=o) for o, S in SS.items() for t in tir]
    dat_ent = P13.augmenter(d, SS["sauts K=3"], None)
    t0 = time.time()
    for seed in range(s0, s1):
        k = f"{tache}|recette_corrigee|s{seed}"
        if out["cells"].get(k, {}).get("proto") != PROTO:
            vals = [R.evaluer(P13.construire(c, RN.BUDGET, seed, 2), d, val=True) for c in cands]
            i = int(np.argmin(vals))
            te = R.evaluer(P13.construire(cands[i], RN.BUDGET, seed, 2), d)
            out["cells"][k] = {"test": float(te), "val": float(vals[i]), "cfg": cands[i], "proto": PROTO}
            print(k, round(te, 4), round(vals[i], 4), int(time.time() - t0), "s", flush=True)
        k = f"{tache}|esn_entree_corrigee|s{seed}"
        if out["cells"].get(k, {}).get("proto") != PROTO:
            cfg, val = RN.regler_famille(dat_ent, "esn_entree", seed, 30)
            te = R.evaluer(P13.construire(cfg, RN.BUDGET, seed, dat_ent["U"].shape[1]), dat_ent)
            out["cells"][k] = {"test": float(te), "val": float(val), "cfg": cfg, "proto": PROTO}
            print(k, round(te, 4), round(val, 4), int(time.time() - t0), "s", flush=True)
        R.ecrire(out, FICH)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], int(a[1]) if len(a) > 2 else 0, int(a[2]) if len(a) > 2 else 10)
