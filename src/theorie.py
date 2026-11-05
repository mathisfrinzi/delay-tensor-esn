"""Vérifications numériques des deux résultats théoriques (réciproque générique, borne par les sauts).

    python src/theorie.py            # imprime les vérifications, écrit results/results_theorie.json

Réseau : z(t) = tanh(W_in·[u(t), 1] + Σ_τ W_τ x(t−τ)), x(t) = (1 − a)·x(t−1) + a·z(t).
J_L(t) = ∂x(t)/∂u(t − L) est calculé exactement le long d'une trajectoire (propagation de la
tangente, D(t) = diag(1 − z(t)²)), sur un réseau tiré comme dans le moteur commun.

Borne (théorème 2) : ‖J_L(t)‖ ≤ a·‖w_u‖·c(L), avec c(0) = 1, c(L) = Σ_{τ∈S'} α_τ c(L − τ),
α_τ = a·‖W_τ‖₂ (+ (1 − a) pour τ = 1), S' = S ∪ {1} si a < 1. Si s = Σ α_τ < 1 : c(L) ≤ s^{h_{S'}(L)}.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reservoir as R   # noqa: E402

LMAX = 120


def sauts(S, lmax):
    h = [0] + [10 ** 9] * lmax
    for L in range(1, lmax + 1):
        h[L] = min((h[L - s] + 1 for s in S if L >= s), default=10 ** 9)
    return np.array(h, float)


def jacobiens(res, u, t0, lmax):
    """‖∂x(t0 + L)/∂u(t0)‖ pour L = 0..lmax, le long de la trajectoire de l'entrée u."""
    N, a = res["N"], float(res["fuite"])
    T = t0 + lmax + 1
    U = np.stack([u[:T], np.ones(T)], 1).astype(np.float64)
    Win = res["Win"].astype(np.float64)
    taus = res["taus"]
    W = {tr[0]: tr for tr in res["tranches"]}
    tmax = max(taus + [1])
    xs = np.zeros((T + tmax, N))        # xs[t + tmax] = x(t)
    zs = np.zeros((T, N))
    for t in range(T):
        drive = U[t] @ Win
        for tau in taus:
            drive += R._applique(W[tau], xs[t - tau + tmax][None, :].astype(np.float32))[0]
        zs[t] = np.tanh(drive)
        xs[t + tmax] = (1 - a) * xs[t - 1 + tmax] + a * zs[t]
    dx = np.zeros((lmax + 1 + tmax, N))   # dx[L + tmax] = ∂x(t0 + L)/∂u(t0)
    out = np.zeros(lmax + 1)
    for L in range(lmax + 1):
        t = t0 + L
        D = 1 - zs[t] ** 2
        drive = Win[0] if L == 0 else np.zeros(N)
        for tau in taus:
            if L - tau >= 0:
                drive = drive + R._applique(W[tau], dx[L - tau + tmax][None, :].astype(np.float32))[0]
        dx[L + tmax] = (1 - a) * dx[L - 1 + tmax] + a * D * drive
        out[L] = np.linalg.norm(dx[L + tmax])
    return out


def norme_op(tr, N):
    if tr[1] == "dense":
        return float(np.linalg.norm(tr[2].astype(np.float64), 2))
    if tr[1] == "perm":
        return float(abs(tr[3]))
    return float(np.abs(tr[2]).max())


def borne(res, lmax):
    a = float(res["fuite"])
    al = {}
    for tr in res["tranches"]:
        al[tr[0]] = al.get(tr[0], 0.0) + a * norme_op(tr, res["N"])
    if a < 1:
        al[1] = al.get(1, 0.0) + (1 - a)
    c = np.zeros(lmax + 1)
    c[0] = 1.0
    for L in range(1, lmax + 1):
        c[L] = sum(v * c[L - t] for t, v in al.items() if L >= t)
    w = float(np.linalg.norm(res["Win"][0].astype(np.float64)))
    return a * w * c, sum(al.values()), sorted(al)


def cas(nom, S, kind, rho, fuite, N=200, seed=0, echelle=1.0):
    res = R.construire(seed, S, N, rho, kinds={t: (kind if t > 1 else "dense") for t in S},
                       echelle=echelle, fuite=fuite)
    rng = np.random.default_rng(100 + seed)
    u = rng.uniform(0, 0.5, 600 + LMAX)
    J = jacobiens(res, u, 400, LMAX)
    B, s, Sp = borne(res, LMAX)
    h = sauts(Sp, LMAX)
    atteint = h < 10 ** 8
    hors = J[~atteint]
    dans = J[atteint]
    viol = int(np.sum(J > B * (1 + 1e-6) + 1e-300))
    geo = s ** np.where(atteint, h, np.inf) if s < 1 else None
    viol_geo = int(np.sum(B[atteint] > B[0] * geo[atteint] * (1 + 1e-9))) if geo is not None else None
    r = {"S": S, "kind": kind, "rho": rho, "fuite": fuite, "N": N, "s": s,
         "max_hors_semigroupe": float(hors.max()) if len(hors) else 0.0,
         "min_dans_semigroupe": float(dans.min()), "violations_borne": viol,
         "violations_borne_geo": viol_geo,
         "rapport_median_J_sur_borne": float(np.median(dans / np.maximum(B[atteint], 1e-300))),
         "J": J.tolist(), "B": B.tolist(), "h": [int(x) if x < 1e8 else -1 for x in h]}
    print(f"{nom:34s} s = {s:6.3f} | hors ⟨S'⟩ max {r['max_hors_semigroupe']:.1e} | dans ⟨S'⟩ min "
          f"{r['min_dans_semigroupe']:.1e} | borne violée {viol} fois | géométrique violée {viol_geo} | "
          f"J/borne médian {r['rapport_median_J_sur_borne']:.2e}")
    return r


if __name__ == "__main__":
    out = {}
    print("Réciproque : J_L ≠ 0 pour tout L ∈ ⟨S'⟩ ; règle : J_L = 0 hors ⟨S'⟩. Borne : J_L ≤ a‖w‖c(L).\n")
    # sans fuite, S lacunaire : ⟨{3, 7}⟩ manque 1, 2, 4, 5, 8, 11
    out["37_contractant"] = cas("{3, 7}, dense, ρ = 0,4, a = 1", [3, 7], "dense", 0.4, 1.0)
    out["37_typique"] = cas("{3, 7}, dense, ρ = 0,9, a = 1", [3, 7], "dense", 0.9, 1.0)
    out["37_perm"] = cas("{3, 7}, permutées, ρ = 0,9, a = 1", [3, 7], "perm", 0.9, 1.0)
    out["130_contractant"] = cas("{1, 30}, dense, ρ = 0,4, a = 1", [1, 30], "dense", 0.4, 1.0)
    out["130_typique"] = cas("{1, 30}, permutées, ρ = 0,9, a = 0,5", [1, 30], "perm", 0.9, 0.5)
    out["esn_contractant"] = cas("ESN {1}, ρ = 0,4, a = 1", [1], "dense", 0.4, 1.0)
    out["esn_typique"] = cas("ESN à fuite {1}, ρ = 0,9, a = 0,3", [1], "dense", 0.9, 0.3, N=632)
    # réciproque sur de nombreux tirages
    nz = []
    for seed in range(20):
        r = cas(f"  tirage {seed} : {{4, 9}}, perm, ρ = 0,8", [4, 9], "perm", 0.8, 1.0, N=100, seed=seed)
        nz.append(r["min_dans_semigroupe"])
    out["tirages_49"] = {"min_dans": nz}
    # la sensibilité suit-elle les sauts ? (corrélation de Spearman entre log ‖J_L‖ et h(L))
    from scipy.stats import spearmanr
    for k in ("37_typique", "130_typique", "130_contractant"):
        r = out[k]
        J, h = np.array(r["J"]), np.array(r["h"])
        m = (h > 0) & (J > 0)
        rs = spearmanr(np.log(J[m]), h[m]).correlation
        r["spearman_logJ_h"] = float(rs)
        print(f"{k:16s} Spearman(log ‖J_L‖, h(L)) = {rs:.2f}", end="")
        if k.startswith("130"):
            print(f" ; ‖J_30‖/‖J_29‖ = {J[30] / J[29]:.1f}, ‖J_60‖/‖J_59‖ = {J[60] / J[59]:.1f}")
        else:
            print()
    # s pour les réservoirs réglés de la recette (partie 4.3)
    import p12_synthese as P13
    rec = R.lire(R.RES / "results_p12_recette.json")["cells"]
    vals = {}
    for k, c in rec.items():
        t, sd = k.split("|")
        if t == "mnist" or "cfg" not in c:
            continue
        r = P13.construire(c["cfg"], 400_000, int(sd[1:]), 2)
        vals.setdefault(t, []).append(borne(r, 1)[1])
    out["s_regles"] = {t: [min(v), float(np.median(v)), max(v)] for t, v in vals.items()}
    for t, v in out["s_regles"].items():
        print(f"s des réservoirs réglés, {t:10s} : min {v[0]:.2f}, médiane {v[1]:.2f}, max {v[2]:.2f}")
    R.ecrire(out, R.RES / "results_theorie.json")
