"""Courbes d'apprentissage (partie 14) : l'avantage des retards tient-il quand
les données abondent ?

Les réservoirs du projet sont entraînés sur environ 4 000 pas (NARMA-30) ou
20 000 (Silverbox), alors que le calcul du spectre et les planchers utilisent
20 000 à 50 000 pas. Ce script trace l'erreur de test selon la longueur du bloc
d'entraînement, pour toutes les familles de 13.1 — modèles GELÉS du protocole
14.1 (aucun réglage ici), seule la lecture ridge est réapprise sur chaque
tronçon.

Protocole. Pour chaque longueur L, la lecture est apprise sur [wash ; wash+L]
avec λ choisi sur les 20 % finaux du tronçon, puis évaluée sur un bloc de test
identique pour toutes les longueurs et toutes les familles.

  - NARMA-30 : une longue réalisation NEUVE par seed (seed 600 + s, utilisée nulle
    part ailleurs), de wash + 32 000 + 1 000 pas ; le test est le dernier bloc de
    1 000 pas. L va de 250 à 32 000 pas, soit jusqu'à huit fois le bloc
    d'entraînement habituel (~4 000) : la question posée (partie 14) est ce qui se
    passe quand les données ABONDENT, pas seulement quand elles manquent.
  - Silverbox : découpage du projet (charger()), L jusqu'à 16 000, la limite des
    30 000 points chargés. Aller plus loin demanderait les données d'estimation
    du benchmark (bloc multisine), à faire avec benchmark_final.charger_benchmark.

    python src/courbes_apprentissage.py courbes
    python src/courbes_apprentissage.py figures
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reservoir as R            # noqa: E402
import p12_synthese as P13        # noqa: E402  (partie 13)
import benchmark_final as BF     # noqa: E402  (modèles gelés, lecture ridge)

ESSAI = R.ESSAI
OUT = R.RES / "essai" if ESSAI else R.RES
FICHIER = OUT / "results_courbes_apprentissage.json"
SEEDS = P13.SEEDS
TACHES = ["narma30", "silverbox"]
PROTO = "apprentissage_2"   # 2 : NARMA-30 jusqu'à 32 000 pas sur réalisations neuves
import p7_common as P7            # noqa: E402

DECALAGE_NARMA = 600   # réalisations neuves de NARMA-30 (le test final utilise 300 à 309)
NTEST_NARMA = 1000
if ESSAI:
    LONGUEURS = {"narma30": [100, 400], "silverbox": [200, 400]}
else:
    LONGUEURS = {"narma30": [250, 500, 1000, 2000, 4000, 8000, 16000, 32000],
                "silverbox": [500, 1000, 2000, 4000, 8000, 16000]}


def donnees(tache, s):
    """(U, y, wash, te0, te1). NARMA-30 : longue réalisation neuve ; Silverbox :
    découpage du projet."""
    if tache == "narma30":
        wash = 200
        T = wash + max(LONGUEURS["narma30"]) + NTEST_NARMA
        U, y, _ = P7.narma_x(30, DECALAGE_NARMA + s, T)
        return U, np.asarray(y, np.float32), wash, T - NTEST_NARMA, T
    d = R.charger(tache, s)
    sp = d["splits"]
    return d["U"], d["y"], sp["wash"], sp["te0"], sp["te1"]


def nrmse_sur_test(F, y, wash, fit_end, te0, te1):
    """λ sur les 20 % finaux de [wash:fit_end], refit, NRMSE sur [te0:te1]
    (std de la cible du bloc de test, comme partout dans le projet)."""
    W, _ = BF.fit_lecture(F, y, wash, fit_end)
    pred = BF.biais(F)[te0:te1].astype(np.float64) @ W
    yy = np.asarray(y, np.float64)
    return float(np.sqrt(((pred - yy[te0:te1]) ** 2).mean()) / np.std(yy[te0:te1]))


def courbes():
    out = R.lire(FICHIER)
    equite = R.lire(P13.FICHIER["equite"])
    recette = R.lire(P13.FICHIER["recette"])
    budgets = R.lire(P13.FICHIER["budgets"])
    for tache in TACHES:
        mds = BF.modeles(tache, equite, recette, budgets)   # gel 14.1 exigé
        for fam, s, cfg in mds:
            for s_data in (SEEDS if fam == "ngrc" else [s]):
                keys = [f"{tache}|{fam}|s{s_data}|L{L}" for L in LONGUEURS[tache]]
                if all(k in out["cells"] for k in keys):
                    continue
                U, y, wash, te0, te1 = donnees(tache, s_data)
                if fam == "ngrc":
                    F = BF.ngrc_features(U[:, 0], cfg["D"], cfg["produits"])
                else:
                    Ue = BF.entree(cfg, U)          # esn_entree : retards et produit en entrée
                    res = P13.construire(cfg, P13.BUDGET, s, Ue.shape[1])
                    F = R.simuler(res, Ue)
                for L, key in zip(LONGUEURS[tache], keys):
                    if key in out["cells"]:
                        continue
                    fit_end = min(wash + L, te0)
                    out["cells"][key] = {"nrmse": nrmse_sur_test(F, y, wash, fit_end, te0, te1),
                                         "L_effectif": fit_end - wash, "proto": PROTO}
                    R.ecrire(out, FICHIER)
                    print(key, round(out["cells"][key]["nrmse"], 5), flush=True)
    out["meta"] = {"proto": PROTO, "longueurs": LONGUEURS, "modeles": "gelés, partie 13 protocole 14.1",
                   "lecture": "λ sur les 20 % finaux de chaque tronçon [wash ; wash+L]",
                   "test": "NARMA-30 : 1 000 derniers pas d'une réalisation neuve (seed 600 + s) ; "
                           "Silverbox : bloc de test du protocole ; identique pour toutes les longueurs",
                   "smoke": ESSAI}
    R.ecrire(out, FICHIER)


def figures():
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["svg.fonttype"] = "none"
    import matplotlib.pyplot as plt
    COL = {"esn_fuite": "#eb6834", "esn_entree": "#c8442c", "dts": "#eda100", "tenseur_fuite": "#1baf7a",
           "tenseur_sans_fuite": "#2a78d6", "recette": "#0f7a55", "uniforme_10": "#7a4fd1",
           "ngrc": "#52514e"}
    LAB = {"esn_fuite": "ESN à fuite", "esn_entree": "ESN, retards en entrée", "dts": "DTS-ESN", "tenseur_fuite": "retards, avec fuite",
           "tenseur_sans_fuite": "retards, sans fuite", "recette": "recette",
           "uniforme_10": "dix retards uniformes", "ngrc": "NG-RC"}
    INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#d9d8d3", "#fcfcfb"
    out = R.lire(FICHIER)
    if not out["cells"]:
        print("aucune cellule")
        return
    fig, axs = plt.subplots(1, len(TACHES), figsize=(12, 4.4))
    fig.patch.set_facecolor(SURF)
    for ax, t in zip(axs, TACHES):
        ax.set_facecolor(SURF)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=INK2, labelsize=9)
        ax.grid(color=GRID, lw=0.7)
        for fam in COL:
            xs, ys, es = [], [], []
            for L in LONGUEURS[t]:
                v = [out["cells"].get(f"{t}|{fam}|s{s}|L{L}", {}).get("nrmse") for s in SEEDS]
                v = [x for x in v if x is not None]
                if v:
                    xs.append(L)
                    ys.append(np.mean(v))
                    es.append(np.std(v))
            if xs:
                ax.errorbar(xs, ys, yerr=es, marker="o", ms=4, lw=1.6, color=COL[fam],
                            ecolor=INK2, capsize=2, label=LAB[fam])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("longueur du bloc d'entraînement", fontsize=9.5, color=INK)
        ax.set_title(t, fontsize=10.5, color=INK, loc="left")
    axs[0].set_ylabel("NRMSE de test (moyenne ± écart-type)", fontsize=9.5, color=INK)
    axs[-1].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "courbes_apprentissage.svg", facecolor=SURF)
    print("figures écrites")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    {"courbes": courbes, "figures": figures}.get(cmd, lambda: print(__doc__))()
