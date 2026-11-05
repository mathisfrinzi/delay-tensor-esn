"""Partie 11 : entrées lisses, placer la mémoire là où la tâche en a besoin.

11.1  banc        Noyaux plantés, entrée AR(1) de lissage φ ∈ {0 ; 0,5 ; 0,8 ; 0,95}.
                  Pour chaque ensemble S : erreur du réservoir et plancher d'information
                  des âges ⟨S⟩ (prédicteur ridge sur ces seuls âges et leurs produits).
11.2  conception  Choix de (S, fuite, ρ) par recouvrement entre l'enveloppe de la tâche
                  et le profil de mémoire du réservoir linéarisé, sans entraîner de
                  réservoir ; comparé au choix par les sauts (partie 5), à une recherche
                  de 60 évaluations dans le même espace, et à un ESN à fuite réglé par
                  60 évaluations.
11.3  pertinence  Sur 40 candidats tirés au hasard : corrélation de rang entre le score de
                  recouvrement et l'erreur de test.

    python src/p11_trous_mous.py banc
    python src/p11_trous_mous.py conception
    python src/p11_trous_mous.py pertinence
    python src/p11_trous_mous.py figures
Reprise automatique : chaque cellule terminée est écrite dans results/results_p11_*.json.
"""
import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reservoir as R  # noqa: E402
import conception as C  # noqa: E402

RES = R.RES
SEEDS = [0, 1] if R.ESSAI else [0, 1, 2, 3, 4]
T = 1600 if R.ESSAI else 6000
T_LONG = 4000 if R.ESSAI else 20000     # séquence longue pour spectres et planchers
LMAX = 40
BUDGET = 400_000
PHIS = [0.0, 0.5, 0.8, 0.95]
NOYAUX = ["pics", "bande", "ilots"]
RHOS = [0.3, 0.5, 0.7, 0.9]
FUITES = [1.0, 0.7, 0.4, 0.2]
N_RECHERCHE = 10 if R.ESSAI else 60
FICHIER = {k: RES / f"results_p11_{k}.json" for k in ("banc", "conception", "pertinence")}


# tâches plantées
def entree_ar1(phi, T, seed):
    rng = np.random.default_rng(seed)
    e = rng.normal(size=T + 500)
    u = np.zeros_like(e)
    for t in range(1, len(e)):
        u[t] = phi * u[t - 1] + np.sqrt(1 - phi ** 2) * e[t]
    return u[500:]


def noyau(nom):
    L = np.arange(LMAX + 1, dtype=float)
    k = np.zeros(LMAX + 1)
    if nom == "bande":
        k[1:] = np.exp(-L[1:] / 8.0) * np.cos(2 * np.pi * L[1:] / 12.0)
    elif nom == "ilots":
        k[4:8] = 1.0
        k[15:21] = 1.0
    return k / max(np.sqrt((k ** 2).sum()), 1e-12)


def cible(nom, u, seed):
    """pics  : y = 1,5·u(t−1)·u(t−30) + 0,5·u(t−30)
    bande : y = s + 0,3·s², s = Σ_L κ_L u(t−L), κ amorti oscillant (période 12)
    ilots : idem avec κ = 1 sur les âges 4 à 7 et 15 à 20
    Bruit d'observation : 1 % de l'écart-type."""
    T = len(u)
    if nom == "pics":
        y = np.zeros(T)
        y[30:] = 1.5 * u[29:-1] * u[:-30] + 0.5 * u[:-30]
    else:
        k = noyau(nom)
        s = np.convolve(u, k)[:T]
        y = s + 0.3 * s ** 2
    rng = np.random.default_rng(seed + 31)
    return y + 0.01 * y.std() * rng.normal(size=T)


def tache_plantee(nom, phi, seed, T=T):
    u = entree_ar1(phi, T, seed)
    y = cible(nom, u, seed)
    U = np.stack([u, np.ones_like(u)], 1).astype(np.float32)
    nte = 300 if R.ESSAI else 1000
    return {"U": U, "y": y, "type": "seq", "splits": {"wash": 200, "tr_end": T - nte, "te0": T - nte, "te1": T}}


def donnees(tache, seed):
    """tache : 'pics|0.8' (noyau planté, φ) ou 'silverbox' / 'wh'."""
    if "|" in tache:
        nom, phi = tache.split("|")
        return tache_plantee(nom, float(phi), seed)
    return R.charger(tache, seed)


def longue(tache):
    """Séquence longue pour spectre et planchers (seed 900, jamais utilisée en test)."""
    if "|" in tache:
        nom, phi = tache.split("|")
        u = entree_ar1(float(phi), T_LONG, 900)
        return u, cible(nom, u, 900)
    d = R.charger(tache, 0)
    s = d["splits"]
    return d["U"][:s["tr_end"], 0], d["y"][:s["tr_end"]]


# évaluation
def eval_config(tache, data, cfg, seed, val=False):
    S = cfg["S"]
    N = R.N_pour_budget(BUDGET, len(S))
    res = R.construire(seed, S, N, cfg["rho"], fuite=cfg.get("fuite", 1.0), echelle=cfg.get("echelle", 1.0))
    return R.evaluer(res, data, val=val)


def meilleur_rho(tache, data, S, seed, fuite=1.0):
    best = None
    for rho in RHOS:
        cfg = {"S": S, "rho": rho, "fuite": fuite}
        v = eval_config(tache, data, cfg, seed, val=True)
        if best is None or v < best[0]:
            best = (v, rho)
    cfg = {"S": S, "rho": best[1], "fuite": fuite}
    return eval_config(tache, data, cfg, seed), best[1]


# 11.1 banc
def ensembles_banc(tache):
    u, y = longue(tache)
    spec = C.spectre(u, y, lmax=LMAX)
    w = C.poids_retards(spec)
    if w is None:
        w = C.enveloppe(spec)
    return {
        "ESN {1}": [1],
        "{1..10}": list(range(1, 11)),
        "pairs {2,4,6,8,10}": [2, 4, 6, 8, 10],
        "{3..10}": list(range(3, 11)),
        "sauts K=2": C.support_sauts(w, 2, LMAX, 40)[0],
        "sauts K=3": C.support_sauts(w, 3, LMAX, 40)[0],
    }, spec


def banc():
    out = R.lire(FICHIER["banc"])
    for nom, phi in itertools.product(NOYAUX, PHIS):
        tache = f"{nom}|{phi}"
        if f"{tache}|meta" not in out["cells"]:
            ens, spec = ensembles_banc(tache)
            u, y = longue(tache)
            out["cells"][f"{tache}|meta"] = {
                "ensembles": ens, "spectre": spec,
                "plancher_tous": C.plancher(u, y, range(LMAX + 1)),
                "planchers": {k: C.plancher(u, y, C.atteignables(S, LMAX)) for k, S in ens.items()}}
            R.ecrire(out, FICHIER["banc"])
        ens = out["cells"][f"{tache}|meta"]["ensembles"]
        for seed in SEEDS:
            data = donnees(tache, seed)
            for k, S in ens.items():
                key = f"{tache}|{k}|s{seed}"
                if key in out["cells"]:
                    continue
                err, rho = meilleur_rho(tache, data, S, seed)
                out["cells"][key] = {"test": err, "rho": rho}
                R.ecrire(out, FICHIER["banc"])
        print("banc fait", tache, flush=True)
    out["meta"] = {"T": T, "T_long": T_LONG, "budget": BUDGET, "rhos": RHOS, "seeds": SEEDS, "phis": PHIS}
    R.ecrire(out, FICHIER["banc"])


# 11.2 conception
def candidats():
    Ss = [[1], [1, 2, 3], list(range(1, 6)), list(range(1, 11))]
    Ss += [[1, d] for d in range(2, 41, 2)]
    Ss += [list(range(k0, k1 + 1)) for k0 in (1, 2, 3, 4) for k1 in (8, 12, 16, 20)]
    out = []
    for S, a, rho in itertools.product(Ss, FUITES, RHOS):
        out.append({"S": S, "fuite": a, "rho": rho})
    return out


def profil(cfg, seed=0):
    N = R.N_pour_budget(BUDGET, len(cfg["S"]))
    res = R.construire(seed, cfg["S"], N, cfg["rho"], fuite=cfg["fuite"])
    return R.profil_lineaire(res, LMAX)


def scores(tache, cfgs):
    u, y = longue(tache)
    spec = C.spectre(u, y, lmax=LMAX)
    E = C.enveloppe(spec)
    return [C.recouvrement(E, profil(c)) for c in cfgs], spec, E


def recherche(tache, data, seed, famille, n=N_RECHERCHE):
    """Recherche aléatoire de n évaluations sur la validation, puis test du meilleur.
    famille 'tenseur' : (S, fuite, ρ) tirés dans candidats() ; échelle d'entrée log-uniforme
    sur [0,05 ; 2]. famille 'esn_fuite' : S = {1}, ρ ∈ [0,3 ; 1,1], fuite ∈ [0,05 ; 1],
    échelle log-uniforme sur [0,05 ; 2]."""
    rng = np.random.default_rng(1000 + seed)
    cands = candidats()
    best = None
    for _ in range(n):
        if famille == "tenseur":
            cfg = dict(cands[rng.integers(len(cands))])
        else:
            cfg = {"S": [1], "rho": float(rng.uniform(0.3, 1.1)), "fuite": float(rng.uniform(0.05, 1.0))}
        cfg["echelle"] = float(np.exp(rng.uniform(np.log(0.05), np.log(2.0))))
        v = eval_config(None, data, cfg, seed, val=True)
        if best is None or v < best[0]:
            best = (v, cfg)
    return eval_config(None, data, best[1], seed), best[1]


TACHES_11_2 = ["bande|0.5", "bande|0.9", "ilots|0.5", "ilots|0.9", "silverbox", "wh"]


def conception():
    out = R.lire(FICHIER["conception"])
    cands = candidats()
    for tache in TACHES_11_2:
        mk = f"{tache}|meta"
        if mk not in out["cells"]:
            sc, spec, E = scores(tache, cands)
            i = int(np.argmax(sc))
            w = C.poids_retards(spec)
            w = C.enveloppe(spec) if w is None else w
            out["cells"][mk] = {"profil": cands[i], "score_profil": sc[i], "enveloppe": list(E),
                                "sauts K=2": C.support_sauts(w, 2, LMAX, 40)[0],
                                "sauts K=3": C.support_sauts(w, 3, LMAX, 40)[0]}
            R.ecrire(out, FICHIER["conception"])
        m = out["cells"][mk]
        for seed in SEEDS:
            data = donnees(tache, seed)
            todo = {
                "profil": lambda: (eval_config(tache, data, m["profil"], seed), m["profil"]),
                "sauts K=2": lambda: meilleur_rho(tache, data, m["sauts K=2"], seed),
                "sauts K=3": lambda: meilleur_rho(tache, data, m["sauts K=3"], seed),
                "{1..10}": lambda: meilleur_rho(tache, data, list(range(1, 11)), seed),
                "recherche tenseur": lambda: recherche(tache, data, seed, "tenseur"),
                "recherche ESN à fuite": lambda: recherche(tache, data, seed, "esn_fuite"),
            }
            for nom, f in todo.items():
                key = f"{tache}|{nom}|s{seed}"
                if key in out["cells"]:
                    continue
                err, cfg = f()
                out["cells"][key] = {"test": err, "cfg": cfg}
                R.ecrire(out, FICHIER["conception"])
                print(key, round(err, 4), flush=True)
    out["meta"] = {"budget": BUDGET, "seeds": SEEDS, "n_recherche": N_RECHERCHE, "candidats": len(cands),
                   "evaluations": {"profil": 0, "sauts K=2": len(RHOS), "sauts K=3": len(RHOS),
                                   "{1..10}": len(RHOS), "recherche tenseur": N_RECHERCHE,
                                   "recherche ESN à fuite": N_RECHERCHE}}
    R.ecrire(out, FICHIER["conception"])


# 11.3 pertinence
def pertinence():
    from scipy.stats import spearmanr
    out = R.lire(FICHIER["pertinence"])
    cands = candidats()
    rng = np.random.default_rng(7)
    tirage = [cands[i] for i in rng.choice(len(cands), size=8 if R.ESSAI else 40, replace=False)]
    for tache in TACHES_11_2:
        mk = f"{tache}|meta"
        if mk not in out["cells"]:
            sc, _, _ = scores(tache, tirage)
            out["cells"][mk] = {"scores": sc, "cfgs": tirage}
            R.ecrire(out, FICHIER["pertinence"])
        data = donnees(tache, 0)
        for i, cfg in enumerate(out["cells"][mk]["cfgs"]):
            key = f"{tache}|c{i}"
            if key not in out["cells"]:
                out["cells"][key] = eval_config(tache, data, cfg, 0)
                R.ecrire(out, FICHIER["pertinence"])
        errs = [out["cells"][f"{tache}|c{i}"] for i in range(len(out["cells"][mk]["cfgs"]))]
        r = spearmanr(out["cells"][mk]["scores"], errs)
        out["cells"][f"{tache}|spearman"] = {"rho": float(r.correlation), "p": float(r.pvalue)}
        R.ecrire(out, FICHIER["pertinence"])
        print(tache, out["cells"][f"{tache}|spearman"], flush=True)


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

    b = R.lire(FICHIER["banc"])
    if b["cells"]:
        fig, axs = plt.subplots(1, len(NOYAUX), figsize=(15, 4.6), sharey=False)
        fig.patch.set_facecolor(SURF)
        for ax, nom in zip(axs, NOYAUX):
            style(ax)
            m0 = b["cells"].get(f"{nom}|0.0|meta")
            if not m0:
                continue
            for k, c in zip(m0["ensembles"], (INK2, BLUE, ORANGE, "#eda100", AQUA, "#6250d6")):
                err = [np.mean([b["cells"].get(f"{nom}|{phi}|{k}|s{s}", {}).get("test", np.nan) for s in SEEDS])
                       for phi in PHIS]
                pl = [b["cells"][f"{nom}|{phi}|meta"]["planchers"][k] if f"{nom}|{phi}|meta" in b["cells"]
                      else np.nan for phi in PHIS]
                ax.plot(PHIS, err, "-o", color=c, lw=1.8, ms=4, label=k)
                ax.plot(PHIS, pl, ":", color=c, lw=1.2)
            ax.set_yscale("log")
            ax.set_xlabel("lissage φ de l'entrée", fontsize=10, color=INK)
            ax.set_title(f"noyau « {nom} » (trait plein : réservoir ; pointillé : plancher)", fontsize=10,
                         color=INK, loc="left")
        axs[0].set_ylabel("NRMSE de test", fontsize=10, color=INK)
        axs[-1].legend(fontsize=8, frameon=False)
        fig.tight_layout()
        fig.savefig(RES / "partie11_banc.svg", facecolor=SURF)

    c = R.lire(FICHIER["conception"])
    if c["cells"]:
        noms = ["profil", "sauts K=2", "sauts K=3", "{1..10}", "recherche tenseur", "recherche ESN à fuite"]
        cols = [AQUA, BLUE, BLUE, INK2, ORANGE, "#eda100"]
        fig, axs = plt.subplots(1, len(TACHES_11_2), figsize=(17, 4.4))
        fig.patch.set_facecolor(SURF)
        for ax, tache in zip(axs, TACHES_11_2):
            style(ax)
            for i, (n, col) in enumerate(zip(noms, cols)):
                v = [c["cells"].get(f"{tache}|{n}|s{s}", {}).get("test") for s in SEEDS]
                v = [x for x in v if x is not None]
                if v:
                    ax.bar(i, np.mean(v), color=col, yerr=np.std(v), ecolor=INK2, capsize=2)
            ax.set_xticks(range(len(noms)))
            ax.set_xticklabels(noms, rotation=55, ha="right", fontsize=8, color=INK)
            ax.set_title(tache, fontsize=10.5, color=INK, loc="left")
        axs[0].set_ylabel("NRMSE de test", fontsize=10, color=INK)
        fig.tight_layout()
        fig.savefig(RES / "partie11_conception.svg", facecolor=SURF)
    print("figures écrites")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    {"banc": banc, "conception": conception, "pertinence": pertinence, "figures": figures}.get(
        cmd, lambda: print(__doc__))()
