"""Partie 12 : quand le calcul de S marche-t-il ?

12.0  diagnostic  Sans réservoir. Pour chaque tâche : spectre de retards, platitude N_eff,
                  visibilité V, erreur du prédicteur par âges (oracle), S calculé. Puis
                  les gains déjà mesurés du S calculé (parties 5, 10, 11, 13) placés sur
                  ces trois axes.
12.1  reglage     Réglage égal : S calculé (K = 2 et 3), {1, …, 10}, S cherché dans
                  l'espace A PRIORI de P13.ENSEMBLES (14.1 : plus aucun « élu » des
                  parties 2 et 5), ESN à fuite. Même recherche aléatoire de ρ, de la
                  fuite et de l'échelle d'entrée pour toutes les familles (mêmes
                  tirages), 5 seeds.
12.2  boutons     Le même réglage sur un noyau planté (les deux pics de NARMA-30), avec
                  deux boutons : le lissage φ de l'entrée (baisse la visibilité) et une
                  rétroaction de sortie a (baisse l'explication par âges finis).

    python src/controle_conception.py diagnostic
    python src/controle_conception.py reglage
12.3  nombre      Le même réglage avec beaucoup de retards : S calculé à K = 5 et 10
                  (glouton) et 10 retards tirés au hasard, contre {1, …, 10} de 12.1.

    python src/controle_conception.py boutons
    python src/controle_conception.py nombre
    python src/controle_conception.py figures
Reprise automatique : results/results_controle_*.json. Avec ESSAI=1, tailles réduites
et sorties dans results/essai/.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import lfilter

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reservoir as R  # noqa: E402
import conception as C  # noqa: E402
import p11_trous_mous as P11  # noqa: E402
import p12_synthese as P13  # noqa: E402  (synthèse, partie 13 du rapport)

OUT = R.RES / "essai" if R.ESSAI else R.RES
SEEDS = [0, 1] if R.ESSAI else [0, 1, 2, 3, 4]
N_EVAL = 6 if R.ESSAI else 30
BUDGET = 400_000
T_PLANTE = 1600 if R.ESSAI else 6000
T_LONG = 4000 if R.ESSAI else 20000
LMAX = 40
FICHIER = {k: OUT / f"results_controle_{k}.json" for k in ("diagnostic", "reglage", "boutons", "nombre")}
FAMILLES = ["calcule_K2", "calcule_K3", "uniforme_10", "cherche", "esn_fuite"]
TACHES_REGLAGE = ["narma10", "narma30", "silverbox", "wh", "bande|0.5|0", "ilots|0.5|0"]
PHIS = [0.0, 0.9, 0.98, 0.995, 0.999]
RETROS = [0.0, 0.5, 0.8, 0.9, 0.95, 0.98]
TACHES_BOUTONS = sorted({f"pics|{phi}|0" for phi in PHIS} | {f"pics|0.0|{a if a else 0}" for a in RETROS},
                        key=lambda t: (float(t.split("|")[1]), float(t.split("|")[2])))


# tâches
def plantee(tache, seed, T):
    """'nom|φ|a' : noyau planté de la partie 11, entrée AR(1) de lissage φ, et
    rétroaction de sortie y(t) = a·y(t − 1) + cible(t)."""
    nom, phi, a = tache.split("|")
    u = P11.entree_ar1(float(phi), T, seed)
    y = P11.cible(nom, u, seed)
    if float(a) > 0:
        y = lfilter([1.0], [1.0, -float(a)], y)
    return u, y


def donnees(tache, seed):
    if "|" not in tache:
        return R.charger(tache, seed)
    u, y = plantee(tache, seed, T_PLANTE)
    nte = 300 if R.ESSAI else 1000
    return {"U": np.stack([u, np.ones_like(u)], 1).astype(np.float32), "y": y, "type": "seq",
            "splits": {"wash": 200, "tr_end": T_PLANTE - nte, "te0": T_PLANTE - nte, "te1": T_PLANTE}}


def sequence_longue(tache):
    """Séquence pour le spectre : seed 900 pour les tâches plantées (jamais testée),
    20 000 pas de la seed 0 pour NARMA, partie entraînement de la seed 0 sinon."""
    if "|" in tache:
        return plantee(tache, 900, T_LONG)
    if tache.startswith("narma"):
        U, y, _ = R.P7.narma_x(int(tache[5:]), 0, T_LONG)
        return U[:, 0], y
    d = R.charger(tache, 0)
    s = d["splits"]
    return d["U"][:s["tr_end"], 0], d["y"][:s["tr_end"]]


def analyse(tache):
    """Spectre, mesures de forme, S calculé pour K = 2 et 3."""
    u, y = sequence_longue(tache)
    sp = C.spectre(u, y, lmax=LMAX)
    I = np.clip(np.asarray(sp["I_lag"], float)[:LMAX + 1], 0, None)
    V = float(I.sum())
    w = C.poids_retards(sp)
    origine = "pics"
    if w is None:
        w, origine = C.enveloppe(sp), "enveloppe"
    return {"oracle": float(sp["test_plein"]), "V": V,
            "N_eff": float(V ** 2 / (I ** 2).sum()) if V > 0 else None,
            "I_lag": I.tolist(), "poids": origine,
            "S_K2": C.support_sauts(w, 2, LMAX, LMAX)[0], "S_K3": C.support_sauts(w, 3, LMAX, LMAX)[0]}


# familles, réglage égal
def tirer(famille, rng_p, rng_s, info):
    """Les paramètres continus viennent de rng_p, le même flux pour toutes les familles
    (mêmes tirages de ρ, fuite, échelle d'entrée à chaque évaluation). Le S de la famille
    « cherche » vient d'un flux séparé rng_s. Toutes les tranches de retard sont permutées
    (gratuites, partie 6) : seule la tranche du plus petit retard coûte N²."""
    cfg = {"rho": float(rng_p.uniform(0.3, 1.1)),
           "echelle": float(np.exp(rng_p.uniform(np.log(0.05), np.log(2.0)))),
           "fuite": float(rng_p.uniform(0.05, 1.0)), "kind": "perm"}
    if famille == "calcule_K2":
        cfg["S"] = list(info["S_K2"])
    elif famille == "calcule_K3":
        cfg["S"] = list(info["S_K3"])
    elif famille == "uniforme_10":
        cfg["S"] = list(range(1, 11))
    elif famille == "cherche":
        cfg["S"] = list(P13.ENSEMBLES[rng_s.integers(len(P13.ENSEMBLES))])
    elif famille == "esn_fuite":
        cfg["S"] = [1]
    elif famille in ("calcule_K5", "calcule_K10"):
        cfg["S"] = list(info["S_" + famille.split("_")[1]])
    elif famille == "aleatoire_10":
        cfg["S"] = list(info["S_alea"][str(info["_seed"])])
    else:
        raise ValueError(famille)
    return cfg


def regler(tache, data, famille, seed, info):
    rng_p = np.random.default_rng(20_000 + seed)
    rng_s = np.random.default_rng(30_000 + seed)
    d = R.d_entree(tache)
    best = None
    for _ in range(N_EVAL):
        cfg = tirer(famille, rng_p, rng_s, info)
        v = R.evaluer(P13.construire(cfg, BUDGET, seed, d), data, val=True)
        if best is None or v < best[0]:
            best = (v, cfg)
    return best[1]


def campagne(nom_fichier, taches, familles=None, fn_analyse=None):
    familles = familles or FAMILLES
    fn_analyse = fn_analyse or analyse
    out = R.lire(FICHIER[nom_fichier])
    for tache in taches:
        mk = f"{tache}|meta"
        if mk not in out["cells"]:
            try:
                out["cells"][mk] = fn_analyse(tache)
            except FileNotFoundError as e:
                print("données absentes, tâche sautée :", tache, e, flush=True)
                continue
            R.ecrire(out, FICHIER[nom_fichier])
        info = out["cells"][mk]
        for seed in SEEDS:
            data = None
            info["_seed"] = seed
            for fam in familles:
                key = f"{tache}|{fam}|s{seed}"
                cell = out["cells"].get(key)
                if cell is not None:
                    if fam != "cherche" or (isinstance(cell, dict) and cell.get("proto") == P13.PROTO):
                        continue
                    # 14.1 : l'ancienne famille « cherche » tirait ses ensembles d'un
                    # espace contenant les élus des parties 2 et 5 (trouvés sur ces
                    # mêmes tâches) ; l'espace est désormais a priori : recalcul,
                    # ancienne cellule conservée sous « |ancien_espace ».
                    if isinstance(cell, dict):
                        out["cells"][key + "|ancien_espace"] = cell
                    out["cells"].pop(key, None)
                data = data or donnees(tache, seed)
                cfg = regler(tache, data, fam, seed, info)
                test = R.evaluer(P13.construire(cfg, BUDGET, seed, R.d_entree(tache)), data)
                cell_out = {"test": test, "cfg": cfg}
                if fam == "cherche":
                    cell_out["proto"] = P13.PROTO
                out["cells"][key] = cell_out
                R.ecrire(out, FICHIER[nom_fichier])
                print(key, round(test, 4), cfg["S"], flush=True)
        info.pop("_seed", None)
    out["meta"] = {"budget": BUDGET, "seeds": SEEDS, "n_evaluations": N_EVAL, "familles": familles,
                   "ensembles_cherche": P13.ENSEMBLES, "tranches": "permutées sauf la plus petite",
                   "erreur": "NRMSE"}
    R.ecrire(out, FICHIER[nom_fichier])


def reglage():
    campagne("reglage", TACHES_REGLAGE)


def boutons():
    campagne("boutons", TACHES_BOUTONS)


# 12.3 nombre de retards : S calculé à K = 5 et 10 contre 10 retards au hasard
FAMILLES_NOMBRE = ["calcule_K5", "calcule_K10", "aleatoire_10"]
TACHES_NOMBRE = ["narma10", "narma30", "silverbox", "wh", "bande|0.5|0", "ilots|0.5|0"]


def analyse_nombre(tache):
    """Comme analyse(), plus S calculé pour K = 5 et 10 (glouton à partir de K = 3) et,
    pour chaque seed, 10 retards tirés au hasard : 1 plus 9 retards distincts dans 2..LMAX."""
    a = analyse(tache)
    u, y = sequence_longue(tache)
    sp = C.spectre(u, y, lmax=LMAX)
    w = C.poids_retards(sp)
    if w is None:
        w = C.enveloppe(sp)
    a["S_K5"] = C.support_sauts(w, 5, LMAX, LMAX)[0]
    a["S_K10"] = C.support_sauts(w, 10, LMAX, LMAX)[0]
    a["S_alea"] = {str(s): [1] + sorted(int(d) for d in np.random.default_rng(40_000 + s).choice(
        np.arange(2, LMAX + 1), 9, replace=False)) for s in SEEDS}
    return a


def nombre():
    campagne("nombre", TACHES_NOMBRE, FAMILLES_NOMBRE, analyse_nombre)


# 12.0 diagnostic (sans réservoir)
def gains_existants():
    """Gains déjà mesurés du S calculé : (comparaison, tâche, rapport d'erreurs)."""
    res = R.RES
    lire = lambda f: json.load(open(res / f))  # noqa: E731
    pts = []
    a = lire("results_p8_conception.json")["agg"]
    for t in ("narma30", "narma25", "silverbox", "wh"):
        calc = min(a[t][k]["mean"] for k in ("calc_k2", "calc_k3", "calc_k4"))
        pts.append(("5", "calculé / élagage glouton", t, calc / a[t]["prune_elu"]["mean"]))
    c = lire("results_p11_banc.json")["cells"]
    g = {}
    for k, v in c.items():
        p = k.split("|")
        if p[-1] != "meta":
            g.setdefault((p[0], p[1], p[2]), []).append(v["test"])
    for noy in ("pics", "bande", "ilots"):
        for phi in ("0.0", "0.5", "0.8", "0.95"):
            calc = min(np.mean(g[(noy, phi, e)]) for e in ("sauts K=2", "sauts K=3"))
            fix = min(np.mean(g[(noy, phi, e)]) for e in ("ESN {1}", "{1..10}", "pairs {2,4,6,8,10}", "{3..10}"))
            pts.append(("11 banc", "calculé / meilleur ensemble fixe", f"{noy}|{float(phi)}|0", calc / fix))
    c = lire("results_p11_conception.json")["cells"]
    g = {}
    for k, v in c.items():
        p = k.split("|")
        if p[-1] != "meta":
            g.setdefault(("|".join(p[:-2]), p[-2]), []).append(v["test"])
    for t in ("bande|0.5", "bande|0.9", "ilots|0.5", "ilots|0.9", "silverbox", "wh"):
        calc = min(np.mean(g[(t, m)]) for m in ("sauts K=2", "sauts K=3"))
        pts.append(("11", "calculé (4 éval.) / recherche (60 éval.)", t + ("|0" if "|" in t else ""),
                    calc / np.mean(g[(t, "recherche tenseur")])))
    e = lire("results_p12_equite.json")["cells"]
    r = lire("results_p12_recette.json")["cells"]
    for t in ("narma30", "silverbox", "wh"):
        rec = np.mean([r[f"{t}|s{s}"]["test"] for s in range(5)])
        ref = np.mean([e[f"{t}|tenseur_fuite|s{s}"]["test"] for s in range(5)])
        pts.append(("13", "recette / recherche avec fuite", t, rec / ref))
    d = lire("results_p10_conception_somme.json")["cells"]
    for t in ("narma10", "narma30", "silverbox"):
        m = lambda f: min(np.mean([d[f"{t}|{K}|{f}|s{s}"] for s in range(5)]) for K in ("K2", "K3"))  # noqa: E731
        pts.append(("10", "retards calculés / réguliers (N = 50)", t, m("calcules") / m("reguliers")))
    return pts


def diagnostic():
    out = R.lire(FICHIER["diagnostic"])
    p8 = json.load(open(R.RES / "results_p8_spectre.json"))["agg"]
    pts = gains_existants()
    taches = sorted({p[2] for p in pts} | {"narma10"} | set(TACHES_BOUTONS))
    for t in taches:
        if t in out["cells"]:
            continue
        try:
            out["cells"][t] = dict(analyse(t), source="spectre recalculé")
        except FileNotFoundError:
            # données absentes : spectre enregistré en partie 5 (même définition, âges 0 à 40)
            I = np.clip(np.asarray(p8[t]["I_single_mean"], float)[:LMAX + 1], 0, None)
            V = float(I.sum())
            out["cells"][t] = {"oracle": float(p8[t]["nrmse_full_mean"]), "V": V,
                               "N_eff": float(V ** 2 / (I ** 2).sum()) if V > 0 else None,
                               "I_lag": I.tolist(), "source": "spectre de la partie 5"}
        R.ecrire(out, FICHIER["diagnostic"])
        c = out["cells"][t]
        print(t, "oracle", round(c["oracle"], 4), "V", round(c["V"], 4), "N_eff", c["N_eff"], flush=True)
    out["meta"] = {"gains": pts, "lmax": LMAX,
                   "definitions": {"N_eff": "(Σ I)² / Σ I², I = hausse d'erreur quand l'âge L est retiré",
                                   "V": "Σ_L I_L", "oracle": "erreur de test du prédicteur par âges 0 à 40"}}
    R.ecrire(out, FICHIER["diagnostic"])


# figures
def figures():
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["svg.fonttype"] = "none"
    import matplotlib.pyplot as plt

    INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#d9d8d3", "#fcfcfb"
    COUL = {"5": "#2a78d6", "11 banc": "#1baf7a", "11": "#eb6834", "13": "#8a5cd1", "10": INK2}
    MARQ = {"5": "o", "11 banc": "s", "11": "^", "13": "D", "10": "v"}

    def style(ax):
        ax.set_facecolor(SURF)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=INK2, labelsize=9)
        ax.grid(color=GRID, lw=0.7)

    dg = R.lire(FICHIER["diagnostic"])
    if dg["cells"] and "gains" in dg["meta"]:
        cel = dg["cells"]
        noms = {"silverbox": "Silverbox", "wh": "WH", "narma10": "NARMA-10"}
        fig, axs = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
        fig.patch.set_facecolor(SURF)
        for ax, cle, lab, log in ((axs[0], "N_eff", "platitude : nombre effectif d'âges N_eff", False),
                                  (axs[1], "V", "visibilité V : hausse d'erreur, âges retirés un à un", True),
                                  (axs[2], "oracle", "erreur du prédicteur par âges 0 à 40", True)):
            style(ax)
            for src in COUL:
                sel = [p for p in dg["meta"]["gains"] if p[0] == src and cel.get(p[2], {}).get(cle) is not None]
                if sel:
                    ax.scatter([max(cel[p[2]][cle], 1e-4) for p in sel], [p[3] for p in sel], s=42,
                               c=COUL[src], marker=MARQ[src], label=f"partie {src} : {sel[0][1]}", zorder=3)
            for p in dg["meta"]["gains"]:
                if p[2] in noms and cel.get(p[2], {}).get(cle) is not None:
                    ax.annotate(noms[p[2]], (max(cel[p[2]][cle], 1e-4), p[3]), xytext=(4, 3),
                                textcoords="offset points", fontsize=7.5, color=INK2)
            ax.axhline(1, color=INK, lw=0.8)
            ax.set_yscale("log")
            if log:
                ax.set_xscale("log")
            ax.set_xlabel(lab, fontsize=9.5, color=INK)
        axs[0].set_ylabel("erreur du S calculé / erreur de la référence", fontsize=9.5, color=INK)
        axs[0].legend(fontsize=7, frameon=False, loc="lower right")
        fig.tight_layout()
        fig.savefig(OUT / "controle_diagnostic.svg", facecolor=SURF)

    cols = dict(zip(FAMILLES, ["#1baf7a", "#0f7a55", "#2a78d6", "#8a5cd1", "#eb6834"]))
    noms_fam = {"calcule_K2": "S calculé, K = 2", "calcule_K3": "S calculé, K = 3",
                "uniforme_10": "{1, …, 10}", "cherche": "S cherché (espace a priori)", "esn_fuite": "ESN à fuite"}

    def valeurs(r, t, fam):
        return [r["cells"][f"{t}|{fam}|s{s}"]["test"] for s in SEEDS if f"{t}|{fam}|s{s}" in r["cells"]]

    # 12.1 : barres par tâche
    r = R.lire(FICHIER["reglage"])
    noms_t = {"narma10": "NARMA-10", "narma30": "NARMA-30", "silverbox": "Silverbox",
              "wh": "Wiener-Hammerstein", "bande|0.5|0": "bande, φ = 0,5", "ilots|0.5|0": "îlots, φ = 0,5"}
    taches = [t for t in TACHES_REGLAGE if valeurs(r, t, FAMILLES[0])]
    if taches:
        fig, ax = plt.subplots(figsize=(11, 4.4))
        fig.patch.set_facecolor(SURF)
        style(ax)
        lw = 0.8 / len(FAMILLES)
        for j, fam in enumerate(FAMILLES):
            for i, t in enumerate(taches):
                v = valeurs(r, t, fam)
                ax.bar(i + (j - 2) * lw, np.mean(v), width=lw * 0.95, color=cols[fam],
                       yerr=np.std(v), ecolor=INK2, error_kw={"lw": 0.8}, label=noms_fam[fam] if i == 0 else None)
        ax.set_xticks(range(len(taches)))
        ax.set_xticklabels([noms_t.get(t, t) for t in taches], fontsize=9, color=INK)
        ax.set_ylabel("NRMSE de test (moyenne ± écart-type, 5 seeds)", fontsize=9.5, color=INK)
        ax.set_yscale("log")
        ax.legend(fontsize=8, frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.12))
        fig.tight_layout()
        fig.savefig(OUT / "controle_reglage.svg", facecolor=SURF)

    # 12.3 : rapport à {1, …, 10} selon le nombre de retards
    rn = R.lire(FICHIER["nombre"])
    tn = [t for t in TACHES_NOMBRE if all(f"{t}|{f}|s{s}" in rn["cells"] for f in FAMILLES_NOMBRE for s in SEEDS)]
    if tn:
        r = R.lire(FICHIER["reglage"])
        fams = [("calcule_K2", r, "S calculé, K = 2", "#1baf7a"), ("calcule_K3", r, "S calculé, K = 3", "#0f7a55"),
                ("calcule_K5", rn, "S calculé, K = 5", "#2aa3a3"), ("calcule_K10", rn, "S calculé, K = 10", "#16607a"),
                ("aleatoire_10", rn, "10 retards au hasard", "#9c9a92")]
        fig, ax = plt.subplots(figsize=(10, 4.2))
        fig.patch.set_facecolor(SURF)
        style(ax)
        lw = 0.8 / len(fams)
        for j, (fam, src, lab, col) in enumerate(fams):
            for i, t in enumerate(tn):
                ref = np.mean(valeurs(r, t, "uniforme_10"))
                ax.bar(i + (j - 2) * lw, np.mean(valeurs(src, t, fam)) / ref, width=lw * 0.95, color=col,
                       label=lab if i == 0 else None)
        ax.axhline(1, color=INK, lw=0.9)
        ax.set_xticks(range(len(tn)))
        ax.set_xticklabels([noms_t.get(t, t) for t in tn], fontsize=9, color=INK)
        ax.set_ylabel("erreur / erreur de {1, …, 10} (même réglage)", fontsize=9.5, color=INK)
        ax.set_yscale("log")
        ax.legend(fontsize=8, frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.12))
        fig.tight_layout()
        fig.savefig(OUT / "controle_nombre.svg", facecolor=SURF)

    # 12.2 : erreur selon chaque bouton, avec l'oracle
    r = R.lire(FICHIER["boutons"])
    if any(k.endswith("|meta") for k in r["cells"]):
        fig, axs = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
        fig.patch.set_facecolor(SURF)
        reglages = ((axs[0], [f"pics|{p}|0" for p in PHIS], [float(p) for p in PHIS],
                     "lissage φ de l'entrée (visibilité V qui baisse)", lambda x: -np.log10(1 - x + 1e-3)),
                    (axs[1], [f"pics|0.0|{a if a else 0}" for a in RETROS], [float(a) for a in RETROS],
                     "rétroaction a de la sortie (oracle qui monte)", lambda x: -np.log10(1 - x + 1e-3)))
        for ax, ts, xs, lab, tr in reglages:
            style(ax)
            ok = [(t, x) for t, x in zip(ts, xs) if f"{t}|meta" in r["cells"]]
            X = [tr(x) for _, x in ok]
            for fam in FAMILLES:
                m = [np.mean(valeurs(r, t, fam)) for t, _ in ok]
                ax.plot(X, m, "-o", ms=4, lw=1.4, color=cols[fam], label=noms_fam[fam])
            ax.plot(X, [r["cells"][f"{t}|meta"]["oracle"] for t, _ in ok], "--", color=INK, lw=1,
                    label="oracle (prédicteur par âges)")
            ax.set_xticks(X)
            ax.set_xticklabels([f"{x:g}" for _, x in ok], fontsize=8, color=INK2)
            ax.set_xlabel(lab, fontsize=9.5, color=INK)
            ax.set_yscale("log")
        axs[0].set_ylabel("NRMSE de test (moyenne, 5 seeds)", fontsize=9.5, color=INK)
        axs[1].legend(fontsize=8, frameon=False, loc="lower right")
        fig.tight_layout()
        fig.savefig(OUT / "controle_boutons.svg", facecolor=SURF)
    print("figures écrites")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    {"diagnostic": diagnostic, "reglage": reglage, "boutons": boutons, "nombre": nombre,
     "figures": figures}.get(
        cmd, lambda: print(__doc__))()
