"""Synthese 2 : heatmap des rapports vs ESN (methode d'optimisation du tenseur
x dataset). Panneau A : rapport NRMSE/NRMSE(ESN) ; panneau B : rapport de MSE =
carre du rapport NRMSE. MNIST : rapport de taux d'erreur (1-acc)/(1-acc_ESN),
identique dans les deux panneaux.
L'ESN N=632 (budget 400k, tau=1 uniquement, W dense) est evalue sous les DEUX
protocoles de la campagne 2 (std = splits eval_pruned ; val = splits 2.3-2.8,
moins de donnees d'entrainement) et chaque methode est rapportee a la
reference ESN du meme protocole (lignes marquees *).
Sorties : results/results_synthese.json + results/partie2_20_synthese.svg."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, LinearSegmentedColormap
from pathlib import Path

plt.rcParams.update({"svg.fonttype": "none", "font.size": 12})
P2 = Path(__file__).resolve().parent.parent / "results"
J = lambda f: json.load(open(P2 / f))

TASKS = ["narma30", "silverbox", "wh", "mnist"]
TCOL = ["NARMA-30", "Silverbox", "WH", "MNIST (err.)"]

# references ESN N=632, deux protocoles
esn = {"std": {t: J("results_esn_ref.json")[t]["std"]["test_mean"] for t in TASKS},
       "val": {t: J("results_esn_ref.json")[t]["val"]["test_mean"] for t in TASKS}}

def mean_seeds(d):
    return float(np.mean([d[k] for k in d]))

best = lambda t, vals: (max(vals) if t == "mnist" else min(vals))

# valeurs par methode : (libelle, protocole, {task: valeur-test})
rows = []
p2 = J("results_part2.json")
ens = J("results_ensembles.json")["agg"]
pru = J("results_prune_final.json")["cells"]
alp = J("results_alpha10.json")
ban = J("results_band_refine.json")
rhot = J("results_rho_tau.json")
ptau = J("results_p_tau.json")
cp = J("results_cprank.json")
clu = J("results_cluster.json")

rows.append(("Tenseur uniforme {1..10} (N=200)", "std",
             {t: p2[t]["tau_1_10"]["mean"] for t in TASKS}))
rows.append(("2.1 Échantillonné 1 τ sur 2 (N=283)", "std",
             {t: best(t, [ens[f"{t}_even_N283"]["mean"], ens[f"{t}_odd_N283"]["mean"]]) for t in TASKS}))
rows.append(("2.2 Pruning glouton (élu)", "std",
             {t: mean_seeds(pru[t]["seeds"]) for t in TASKS}))
rows.append(("2.3 α_τ appris (multiplicatifs)", "val",
             {t: alp[t]["learned"]["test"] for t in TASKS}))
rows.append(("2.4 Bande continue [τ₀, τ₁]", "val",
             {t: ban[t]["final_band"]["test_mean"] for t in TASKS}))
rows.append(("2.5 Profil ρ_τ imposé (meilleur)", "val",
             {t: best(t, [rhot[t][p]["test_mean"] for p in rhot[t]]) for t in TASKS}))
rows.append(("2.6 Densité p(τ) (meilleure)", "val",
             {t: best(t, [ptau[t][k]["test_mean"] for k in ("learned", "uni02", "uni05")]) for t in TASKS}))
rows.append(("2.7 CP-rang sur élus (meilleur r)", "std",
             {t: best(t, [mean_seeds(cp[t]["ranks"][r]["seeds"]) for r in cp[t]["ranks"]]) for t in TASKS}))
rows.append(("2.8 Mélange 2 clusters", "val",
             {t: clu[t]["test_mean"] for t in TASKS}))

# rapports (MNIST : taux d'erreur ; ailleurs NRMSE) ; MSE = carre
rat_nrmse, rat_mse, notes = [], [], []
for lab, proto, vals in rows:
    rn, rm = [], []
    for t in TASKS:
        e = esn[proto][t]
        if t == "mnist":
            r = (1 - vals[t]) / (1 - e)
            rn.append(r); rm.append(r)  # taux d'erreur : deja un rapport de cout
        else:
            rn.append(vals[t] / e); rm.append((vals[t] / e) ** 2)
    rat_nrmse.append(rn); rat_mse.append(rm)
    notes.append((lab, proto, vals))

rat_nrmse = np.vstack([np.ones(4), np.array(rat_nrmse)])   # ligne ESN = 1,00 en tete
rat_mse = np.vstack([np.ones(4), np.array(rat_mse)])

labels = ["ESN N=632 (référence)"] + [l + (" *" if p == "val" else "") for l, p, _ in notes]

json.dump({"esn_ref": esn,
           "rows": [{"label": l, "proto": p, "values": {t: v for t, v in vals.items()},
                     "ratio_nrmse": {t: r for t, r in zip(TASKS, rn)},
                     "ratio_mse": {t: r for t, r in zip(TASKS, rm)}}
                    for (l, p, vals), rn, rm in zip(notes, rat_nrmse[1:], rat_mse[1:])]},
          open(P2 / "results_synthese.json", "w"), indent=1)

# figure : 2 panneaux propres (A rapport NRMSE, B rapport MSE)
# palette maison : vert tenseur -> beige neutre -> rouge ESN
CMAP = LinearSegmentedColormap.from_list(
    "syn", ["#00441b", "#1a9850", "#91cf60", "#d9ef8b", "#ffffbf",
            "#fee08b", "#fc8d59", "#d62728"])

def fr(v):
    d = 2 if v >= 0.0995 else (3 if v >= 0.00995 else 4)
    return ("×" + f"{v:.{d}f}").replace(".", ",") if v < 0.995 else "1,00"

def heat(ax, M, vmin, title, ylabels=False):
    norm = LogNorm(vmin=vmin, vmax=1.0)
    im = ax.imshow(M, cmap=CMAP, norm=norm, aspect="auto")
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_xticks(range(4), TCOL, fontsize=12)
    ax.tick_params(length=0)
    if ylabels:
        ax.set_yticks(range(len(labels)), labels, fontsize=12)
    else:
        ax.set_yticks([])
    # grille blanche entre cellules + separation sous la ligne ESN
    ax.set_xticks(np.arange(-.5, 4, 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(labels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2.5)
    ax.axhline(0.5, color="#333333", linewidth=2.5)
    for s in ax.spines.values():
        s.set_visible(False)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            rgba = CMAP(norm(M[i, j]))
            lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            ax.text(j, i, fr(M[i, j]), ha="center", va="center", fontsize=12.5,
                    fontweight="bold", color="white" if lum < 0.5 else "#1a1a1a")
    return im

fig, axes = plt.subplots(1, 2, figsize=(15.5, 7.0), width_ratios=[1, 1])
imA = heat(axes[0], rat_nrmse, 0.03, "A : Rapport NRMSE / NRMSE(ESN)", ylabels=True)
imB = heat(axes[1], rat_mse, 0.003, "B : Rapport MSE / MSE(ESN)  (= A²)")

# losange orange : meilleure methode par colonne (hors ligne ESN), panneau A
for j in range(4):
    i = int(np.argmin(rat_nrmse[1:, j])) + 1
    axes[0].plot(j, i, marker="D", color="#ff8c00", markersize=11,
                 markeredgecolor="#1a1a1a", markeredgewidth=0.9, zorder=5)

cbA = fig.colorbar(imA, ax=axes[0], shrink=0.82, pad=0.03, ticks=[0.03, 0.1, 0.3, 1.0])
cbA.set_label("× mieux que l'ESN (vert) … ESN = 1", fontsize=10.5)
cbA.ax.set_yticklabels(["×0,03", "×0,1", "×0,3", "ESN"])
cbA.ax.tick_params(labelsize=10)
cbB = fig.colorbar(imB, ax=axes[1], shrink=0.82, pad=0.03, ticks=[0.003, 0.01, 0.1, 1.0])
cbB.set_label("rapport de MSE, échelle log", fontsize=10.5)
cbB.ax.set_yticklabels(["×0,003", "×0,01", "×0,1", "ESN"])
cbB.ax.tick_params(labelsize=10)

fig.suptitle("Synthèse 2 : chaque méthode d'optimisation du tenseur vs ESN N=632, à budget de connexions égal (400k)",
             fontsize=14.5, y=0.985)
fig.text(0.012, 0.015,
         "* méthode évaluée sous le protocole à-validation (≈ 5k points d'entraînement en moins) ; "
         "rapportée à l'ESN N=632 recalculé sous le même protocole.   "
         "MNIST (err.) : rapport de taux d'erreur (1−acc)/(1−acc_ESN), identique en A et B.   "
         "◆ meilleure méthode par colonne.",
         fontsize=9.5, color="#555555")
fig.subplots_adjust(left=0.235, right=0.965, top=0.88, bottom=0.075, wspace=0.28)
fig.savefig(P2 / "partie2_20_synthese.svg", bbox_inches="tight")
print("ok : ratios NRMSE vs ESN :")
for lab, r in zip(labels, rat_nrmse):
    print(f"{lab:<36}" + "".join(f"{v:>10.3f}" for v in r))
