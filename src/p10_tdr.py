"""Partie 10 : réservoir physique à retards (un nœud non linéaire, N nœuds virtuels).

Modèle discret standard d'un réservoir à retard (Appeltant et al. 2011 ; Köster et al.
2021). Le temps fin g = n·N + i compte le tour n et le nœud virtuel i :

    z[g] = p·z[g−1] + (1 − p)·tanh( η·m_i·u(n) + Σ_k β_k·z[g − D_k] )

  - m_i : masque d'entrée ±1 tiré au hasard ; u(n) est tenu pendant le tour n ;
  - p = exp(−θ/T_nœud) : inertie du nœud (couple deux nœuds virtuels voisins) ;
  - D_k = m_k·N + s_k : longueur de la boucle k, en unités de θ. Elle relie le nœud i
    au nœud i − s_k, m_k tours plus tôt (avec retour au tour précédent si i − s_k < 0) ;
  - normalisation des gains, au choix :
      « quadratique » β_k = β/√K (Σ β_k² = β², première campagne),
      « somme »       β_k = β/K  (Σ β_k = β : même gain de boucle que le retard unique).
    Les gains sont tous positifs et agissent sur une seule variable : linéarisé autour de 0,
    l'état z = 0 est instable dès que Σ β_k > 1. Avec β/√K et β ≥ 0,8, c'est le cas pour
    tout K ≥ 2 : la première campagne à plusieurs retards a tourné en régime saturé ;
  - lecture : ridge sur les N nœuds du tour n, [z[nN], …, z[nN+N−1], 1].

Une boucle est un générateur (m, s) du plan (tour, décalage) de la partie 9. s = 0
est une tranche identité, s ≠ 0 une permutation circulaire (partie 6).

Sous-commandes :
    python src/p10_tdr.py calibrer       # η, β, p par tâche, sur la seed 0, D = N + 1
    P10_N=200 python src/p10_tdr.py ...  # N nœuds virtuels (50 par défaut), fichiers suffixés _N200
    python src/p10_tdr.py balayage       # 10.1 : un seul retard, D de N/2 à 3,5·N
    python src/p10_tdr.py conception     # 10.2 : 2 et 3 retards, normalisation quadratique
    python src/p10_tdr.py tolerance      # 10.3 : retards décalés de ±1, ±2, normalisation quadratique
    python src/p10_tdr.py conception somme   # 10.5 : la même chose avec Σ β_k = β
    python src/p10_tdr.py tolerance somme    # 10.5 : idem
    python src/p10_tdr.py figures        # figures de la partie 10
Chaque sous-commande reprend là où elle s'est arrêtée (JSON dans results/).
"""
import functools
import itertools
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reservoir as R  # noqa: E402
import conception as C  # noqa: E402

P7 = R.P7
RES = R.RES
NV = 20 if R.ESSAI else int(os.environ.get("P10_N", "50"))   # nœuds virtuels (P10_N pour changer N)
LMAX = 40                               # âge maximal considéré (en tours)
SEEDS = [0, 1] if R.ESSAI else [0, 1, 2, 3, 4]
TACHES = ["narma10", "narma30", "mc", "silverbox"]
GRILLE = {"eta": [0.1, 0.5, 1.0], "beta": [0.5, 0.8, 0.95], "p": [0.0, 0.5]}
SUFFIXE_N = "" if NV == 50 else f"_N{NV}"      # fichiers séparés pour N ≠ 50
FICHIER = {k: RES / f"results_p10_{k}{SUFFIXE_N}.json" for k in ("calibrage", "balayage", "conception", "tolerance",
                                                                "conception_somme", "tolerance_somme")}
NORMES = ("quadratique", "somme")
NORME = "quadratique"                   # changé par la ligne de commande (« somme »)


def gains(beta, K, norme=None):
    """Gains β_k des K boucles. « quadratique » : β/√K ; « somme » : β/K."""
    norme = norme or NORME
    return np.full(K, beta / (np.sqrt(K) if norme == "quadratique" else K))

try:
    from numba import njit
except ImportError:                     # repli sans numba : même calcul, plus lent
    def njit(*a, **k):
        def deco(f):
            return f
        return deco if not (a and callable(a[0])) else a[0]


# simulation
@njit(cache=True)
def _boucle(u, masque, eta, p, D, beta):
    T = u.shape[0]
    N = masque.shape[0]
    G = T * N
    z = np.zeros(G, np.float64)
    K = D.shape[0]
    prev = 0.0
    for g in range(G):
        n = g // N
        i = g - n * N
        a = eta * masque[i] * u[n]
        for k in range(K):
            j = g - D[k]
            if j >= 0:
                a += beta[k] * z[j]
        v = p * prev + (1.0 - p) * np.tanh(a)
        z[g] = v
        prev = v
    return z


def simuler(u, retards, N, eta, beta, p, seed):
    """u : (T,) entrée par tour. Renvoie les états (T, N)."""
    rng = np.random.default_rng(seed)
    masque = rng.choice((-1.0, 1.0), size=N)
    D = np.asarray(retards, np.int64)
    b = gains(beta, len(D))
    z = _boucle(np.asarray(u, np.float64), masque, float(eta), float(p), D, b)
    return z.reshape(len(u), N).astype(np.float32)


def evaluer(tache, data, retards, eta, beta, p, seed, val=False):
    if data["type"] == "mc":
        F = simuler(data["u"], retards, NV, eta, beta, p, seed)
        return 60.0 - float(R.mc_rapide(F, data["u"], 60).sum())
    F = simuler(data["U"][:, 0], retards, NV, eta, beta, p, seed)
    s = data["splits"]
    if val:
        v0 = s["wash"] + int(0.8 * (s["tr_end"] - s["wash"]))
        return P7.ridge_readout(F, data["y"], s["wash"], v0, v0, s["tr_end"])[0]
    return P7.ridge_readout(F, data["y"], s["wash"], s["tr_end"], s["te0"], s["te1"])[0]


# arithmétique
def sauts_tours(retards, N, lmax=LMAX):
    """h[L] = nombre minimal de boucles pour qu'une entrée du tour n − L atteigne un
    nœud du tour n (décalage total o dans ]L·N − N ; L·N + N[), sans l'inertie.
    decalages[L] = décalages σ = o − L·N (modulo N) obtenus avec h[L] ou h[L] + 1 boucles."""
    omax = (lmax + 1) * N
    INF = 10 ** 9
    hop = np.full(omax + 1, INF, np.int64)
    hop[0] = 0
    for o in range(1, omax + 1):
        for d in retards:
            if o >= d and hop[o - d] + 1 < hop[o]:
                hop[o] = hop[o - d] + 1
    h = np.full(lmax + 1, np.inf)
    decalages = {}
    for L in range(1, lmax + 1):
        lo, hi = max(L * N - N + 1, 1), min(L * N + N - 1, omax)
        fen = hop[lo:hi + 1]
        if fen.min() >= INF:
            continue
        h[L] = fen.min()
        ok = np.nonzero(fen <= fen.min() + 1)[0] + lo
        decalages[L] = sorted({int((o - L * N) % N) for o in ok})
    return h, decalages


def diversite(retards, N, lmax=LMAX):
    """Δ = nombre de décalages distincts (modulo N) par lesquels les âges 1..lmax
    atteignent l'état. Δ = 1 : toutes les traces arrivent sur les mêmes nœuds
    (cas D = N, tranche identité)."""
    _, dec = sauts_tours(retards, N, lmax)
    return len({s for v in dec.values() for s in v})


def concevoir(w, K, N, mmax=LMAX + 1, smax=2):
    """Retards D_k = m_k·N + s_k (m ≤ mmax, |s| ≤ smax, D ≥ 1) qui minimisent
    Σ_L w_L·h(L), puis maximisent Δ à coût égal. Exhaustif pour K ≤ 2, glouton ensuite."""
    cands = sorted({m * N + s for m in range(0, mmax + 1) for s in range(-smax, smax + 1) if m * N + s >= 1})
    cibles = np.nonzero(w)[0]

    def score(ret):
        h, dec = sauts_tours(ret, N)
        c = float(np.sum(w[cibles] * np.where(np.isfinite(h[cibles]), h[cibles], 1e6)))
        return (c, -len({s for v in dec.values() for s in v}))

    if K == 1:
        return [min(cands, key=lambda d: score([d]))]
    best = min(itertools.combinations(cands, 2), key=lambda c: score(list(c)))
    ret = list(best)
    while len(ret) < K:
        d = min((d for d in cands if d not in ret), key=lambda d: score(ret + [d]))
        ret.append(d)
    return sorted(ret)


@functools.lru_cache(maxsize=None)
def poids_tache(tache):
    """Poids des âges utiles. mc : uniforme sur 1..LMAX. Autres : spectre (partie 5),
    mesuré sur la seed 0, sur une séquence de 20 000 tours."""
    if tache == "mc":
        w = np.ones(LMAX + 1)
        w[0] = 0
        return w / w.sum()
    if tache.startswith("narma"):
        U, y, _ = P7.narma_x(int(tache[5:]), 0, 4000 if R.ESSAI else 20000)
        u = U[:, 0]
    else:
        d = R.charger(tache, 0)
        u, y = d["U"][:, 0], d["y"]
    spec = C.spectre(u, y, lmax=LMAX, produits=True)
    w = C.poids_retards(spec)
    if w is None:
        w = C.enveloppe(spec)
    return w


# sous-commandes
def charger(tache, seed):
    return R.charger(tache, seed)


def calibrer():
    out = R.lire(FICHIER["calibrage"])
    for tache in TACHES:
        if tache in out["cells"]:
            continue
        data = charger(tache, 0)
        best = None
        for eta, beta, p in itertools.product(GRILLE["eta"], GRILLE["beta"], GRILLE["p"]):
            v = evaluer(tache, data, [NV + 1], eta, beta, p, 0, val=True)
            if best is None or v < best[0]:
                best = (v, eta, beta, p)
        out["cells"][tache] = {"val": best[0], "eta": best[1], "beta": best[2], "p": best[3]}
        out["meta"] = {"N": NV, "grille": GRILLE, "retard": "N + 1", "seed": 0}
        R.ecrire(out, FICHIER["calibrage"])
        print("calibré", tache, out["cells"][tache], flush=True)


def reglages(tache):
    c = R.lire(FICHIER["calibrage"])["cells"]
    if tache not in c:
        raise SystemExit("lancer d'abord : python src/p10_tdr.py calibrer")
    return c[tache]["eta"], c[tache]["beta"], c[tache]["p"]


def balayage():
    """10.1 : un seul retard D, de N/2 à 3,5·N, tâches mc et narma10, 5 seeds."""
    out = R.lire(FICHIER["balayage"])
    Ds = list(range(NV // 2, int(3.5 * NV) + 1))
    for tache in ("mc", "narma10"):
        eta, beta, p = reglages(tache)
        for seed in SEEDS:
            data = charger(tache, seed)
            for D in Ds:
                key = f"{tache}|D{D}|s{seed}"
                if key in out["cells"]:
                    continue
                for p_ in sorted({p, 0.0}):
                    out["cells"][f"{tache}|D{D}|p{p_}|s{seed}"] = evaluer(tache, data, [D], eta, beta, p_, seed)
                out["cells"][key] = True
                R.ecrire(out, FICHIER["balayage"])
        print("balayage fait", tache, flush=True)
    out["meta"] = {"N": NV, "D": Ds, "seeds": SEEDS, "diversite": {str(D): diversite([D], NV) for D in Ds},
                   "erreur": "60 − capacité (mc), NRMSE (narma10)"}
    R.ecrire(out, FICHIER["balayage"])


def meilleur_retard_unique(tache):
    """Meilleur D du balayage sur la seed 0 (validation impossible ici : le balayage
    rapporte le test ; on choisit sur la seed 0 et on rapporte les seeds 1 à 4)."""
    b = R.lire(FICHIER["balayage"])["cells"]
    eta, beta, p = reglages(tache)
    cands = [(v, int(k.split("|")[1][1:])) for k, v in b.items()
             if k.startswith(f"{tache}|D") and k.endswith(f"|p{p}|s0") and not isinstance(v, bool)]
    return min(cands)[1] if cands else NV + 1


def familles(tache, K):
    w = poids_tache(tache)
    calc = concevoir(w, K, NV)
    reso = sorted({max(1, (d + NV // 2) // NV) * NV for d in calc})
    while len(reso) < K:
        reso.append(reso[-1] + NV)
    fam = {
        "un_retard_N+1": [NV + 1],
        "reguliers": [k * NV + 1 for k in range(1, K + 1)],
        "calcules": calc,
        "calcules_s0": reso,
    }
    rng = np.random.default_rng(100 + K)
    cands = [m * NV + s for m in range(1, LMAX + 2) for s in range(-2, 3)]
    for r in range(3):
        fam[f"aleatoires_{r}"] = sorted(int(x) for x in rng.choice(cands, size=K, replace=False))
    if tache in ("mc", "narma10"):
        fam["un_retard_meilleur"] = [meilleur_retard_unique(tache)]
    return fam, w


def conception():
    """10.2 : à N et masse de rétroaction égaux, retards calculés contre un retard,
    retards réguliers, retards aléatoires, et les mêmes retards calculés ramenés à s = 0."""
    nom_f = "conception" if NORME == "quadratique" else "conception_somme"
    out = R.lire(FICHIER[nom_f])
    for tache in TACHES:
        eta, beta, p = reglages(tache)
        for K in (2, 3):
            fam, w = familles(tache, K)
            out["meta"].setdefault("familles", {})[f"{tache}|K{K}"] = {
                k: {"retards": v, "diversite": diversite(v, NV),
                    "sauts_ponderes": float(np.sum(w * np.nan_to_num(sauts_tours(v, NV)[0], posinf=1e6)))}
                for k, v in fam.items()}
            for seed in SEEDS:
                data = charger(tache, seed)
                for nom, ret in fam.items():
                    key = f"{tache}|K{K}|{nom}|s{seed}"
                    if key in out["cells"]:
                        continue
                    t0 = time.time()
                    out["cells"][key] = evaluer(tache, data, ret, eta, beta, p, seed)
                    R.ecrire(out, FICHIER[nom_f])
                    print(key, round(out["cells"][key], 4), f"{time.time() - t0:.1f}s", flush=True)
    out["meta"].update({"N": NV, "seeds": SEEDS, "reglages": {t: reglages(t) for t in TACHES},
                        "normalisation": NORME,
                        "gain_de_boucle": {t: {f"K{K}": float(gains(reglages(t)[1], K).sum()) for K in (1, 2, 3)}
                                           for t in TACHES}})
    R.ecrire(out, FICHIER[nom_f])


def tolerance():
    """10.3 : retards calculés (K = 3) décalés de δ ∈ {−2, −1, +1, +2} unités θ, un retard
    à la fois puis tous ensemble ; plus D = N − 1, N, N + 1 en retard unique."""
    nom_f = "tolerance" if NORME == "quadratique" else "tolerance_somme"
    out = R.lire(FICHIER[nom_f])
    for tache in ("narma30", "mc"):
        eta, beta, p = reglages(tache)
        calc = concevoir(poids_tache(tache), 3, NV)
        variantes = {"exact": calc}
        for k in range(3):
            for dlt in (-2, -1, 1, 2):
                v = list(calc)
                v[k] = max(1, v[k] + dlt)
                variantes[f"r{k}{dlt:+d}"] = v
        for dlt in (-1, 1):
            variantes[f"tous{dlt:+d}"] = [max(1, d + dlt) for d in calc]
        for D in (NV - 1, NV, NV + 1):
            variantes[f"unique_{D}"] = [D]
        out["meta"].setdefault("variantes", {})[tache] = variantes
        for seed in SEEDS:
            data = charger(tache, seed)
            for nom, ret in variantes.items():
                key = f"{tache}|{nom}|s{seed}"
                if key not in out["cells"]:
                    out["cells"][key] = evaluer(tache, data, ret, eta, beta, p, seed)
                    R.ecrire(out, FICHIER[nom_f])
    out["meta"].update({"N": NV, "seeds": SEEDS, "normalisation": NORME})
    R.ecrire(out, FICHIER[nom_f])


# figures
def figures():
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["svg.fonttype"] = "none"
    import matplotlib.pyplot as plt

    BLUE, ORANGE, AQUA, INK, INK2, GRID, SURF = ("#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e",
                                                 "#d9d8d3", "#fcfcfb")

    def style(ax):
        ax.set_facecolor(SURF)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=INK2, labelsize=9)
        ax.grid(axis="y", color=GRID, lw=0.8)

    b = R.lire(FICHIER["balayage"])
    if b["cells"] and "D" in b["meta"]:          # balayage terminé seulement
        Ds = b["meta"]["D"]
        fig, axs = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
        fig.patch.set_facecolor(SURF)
        for ax, tache, lab in ((axs[0], "mc", "capacité mémoire (sur 60)"), (axs[1], "narma10", "NRMSE NARMA-10")):
            style(ax)
            for p_, c in zip(sorted({0.0, reglages(tache)[2]}), (BLUE, ORANGE)):
                vals = [np.mean([b["cells"].get(f"{tache}|D{D}|p{p_}|s{s}", np.nan) for s in SEEDS]) for D in Ds]
                vals = [60 - v for v in vals] if tache == "mc" else vals
                ax.plot(np.array(Ds) / NV, vals, color=c, lw=1.6, label=f"inertie p = {p_}")
            ax.set_ylabel(lab, fontsize=10, color=INK)
            ax.legend(fontsize=9, frameon=False)
        style(axs[2])
        axs[2].plot(np.array(Ds) / NV, [b["meta"]["diversite"][str(D)] for D in Ds], color=AQUA, lw=1.6)
        axs[2].set_ylabel("décalages distincts Δ", fontsize=10, color=INK)
        axs[2].set_xlabel("longueur du retard D / N (en tours)", fontsize=10, color=INK)
        for ax in axs:
            for r in (1, 1.5, 2, 2.5, 3):
                ax.axvline(r, color=INK2, lw=0.8, ls=(0, (3, 3)))
        fig.tight_layout()
        fig.savefig(RES / f"partie10_balayage{SUFFIXE_N}.svg", facecolor=SURF)

    for suffixe in ("", "_somme"):
        c = R.lire(FICHIER["conception" + suffixe])
        if not c["cells"]:
            continue
        ordre = ["un_retard_N+1", "un_retard_meilleur", "reguliers", "aleatoires", "calcules_s0", "calcules"]
        noms = {"un_retard_N+1": "1 retard\nN+1", "un_retard_meilleur": "1 retard\nmeilleur",
                "reguliers": "réguliers", "aleatoires": "aléatoires\n(moy. 3)", "calcules_s0": "calculés\ns = 0",
                "calcules": "calculés"}
        fig, axs = plt.subplots(1, len(TACHES), figsize=(16, 4.6))
        fig.patch.set_facecolor(SURF)
        for ax, tache in zip(axs, TACHES):
            style(ax)
            for j, (K, col) in enumerate(((2, BLUE), (3, AQUA))):
                for i, nom in enumerate(ordre):
                    if nom == "aleatoires":
                        v = [c["cells"].get(f"{tache}|K{K}|aleatoires_{r}|s{s}") for r in range(3) for s in SEEDS]
                    else:
                        v = [c["cells"].get(f"{tache}|K{K}|{nom}|s{s}") for s in SEEDS]
                    v = [x for x in v if x is not None]
                    if v:
                        ax.bar(i + (j - 0.5) * 0.38, np.mean(v), width=0.36, color=col,
                               label=f"K = {K}" if i == 0 else None)
            ax.set_xticks(range(len(ordre)))
            ax.set_xticklabels([noms[n] for n in ordre], fontsize=7.5, color=INK)
            ax.set_title(tache + (" (60 − capacité)" if tache == "mc" else " (NRMSE)"), fontsize=10.5,
                         color=INK, loc="left")
        axs[0].legend(fontsize=9, frameon=False)
        fig.tight_layout()
        fig.savefig(RES / f"partie10_conception{suffixe}{SUFFIXE_N}.svg", facecolor=SURF)

    for suffixe in ("", "_somme"):
        t = R.lire(FICHIER["tolerance" + suffixe])
        if not t["cells"]:
            continue
        fig, axs = plt.subplots(1, 2, figsize=(13, 4.2))
        fig.patch.set_facecolor(SURF)
        for ax, tache in zip(axs, ("narma30", "mc")):
            style(ax)
            var = list(t["meta"]["variantes"].get(tache, {}))
            m = [np.mean([t["cells"].get(f"{tache}|{v}|s{s}", np.nan) for s in SEEDS]) for v in var]
            ax.bar(range(len(var)), m, color=[AQUA if v == "exact" else BLUE for v in var])
            ax.set_xticks(range(len(var)))
            ax.set_xticklabels(var, rotation=60, fontsize=7.5, color=INK)
            ax.set_title(tache, fontsize=10.5, color=INK, loc="left")
        fig.tight_layout()
        fig.savefig(RES / f"partie10_tolerance{suffixe}{SUFFIXE_N}.svg", facecolor=SURF)
    print("figures écrites")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if len(sys.argv) > 2:
        if sys.argv[2] not in NORMES:
            raise SystemExit(f"normalisation inconnue : {sys.argv[2]} (choix : {', '.join(NORMES)})")
        NORME = sys.argv[2]
    {"calibrer": calibrer, "balayage": balayage, "conception": conception, "tolerance": tolerance,
     "figures": figures}.get(cmd, lambda: print(__doc__))()
