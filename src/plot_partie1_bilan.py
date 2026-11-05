"""Partie 1 : bilan visuel des campagnes 1.2 / 1.2bis / 1.3 (remplace trois tableaux).

Chiffres recopiés des tableaux de rapport/1_ou_ca_gagne.md (moyennes 3 seeds), aucun recalcul.
Valeur tracée = erreur(tenseur τ=1..10, N=200) / erreur(ESN N=632), même budget 400k :
  - NRMSE pour les tâches de régression,
  - taux d'erreur (1 − acc) pour les classifications.
< 1 : le tenseur gagne.  Exception signalée : Melbourne comparé à l'ESN N=200
(l'ESN N=632 y surapprend, 633 features pour 2 550 points).

Sortie : results/partie1_bilan.svg
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt
import numpy as np

RES = Path(__file__).resolve().parents[1] / "results"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

FINE, COURT, EXACT = "mémoire longue et fine", "mémoire courte / lisse / statique", "recombinaison non linéaire exacte"
# (nom, tenseur, ESN632, est_classif, famille)
data = [
    ("NARMA-10", 0.262, 0.177, False, COURT),
    ("NARMA-30", 0.388, 0.586, False, FINE),
    ("Mackey-Glass τ=17, h=10", 0.0116, 0.0126, False, FINE),
    ("Hénon h=1", 0.319, 0.0040, False, COURT),
    ("MNIST lignes 5k", 82.5, 64.0, True, FINE),
    ("MNIST lignes 60k", 87.3, 72.3, True, FINE),
    ("Fashion-MNIST 5k", 78.2, 70.0, True, FINE),
    ("MNIST pixel/pixel", 49.9, 35.5, True, FINE),
    ("Melbourne (vs ESN N=200)", 0.622, 1.761, False, FINE),
    ("Sunspots", 0.424, 0.592, False, FINE),
    ("ETTh1", 0.267, 0.259, False, COURT),
    ("Silverbox", 0.284, 0.597, False, FINE),
    ("Wiener-Hammerstein", 0.092, 0.443, False, FINE),
    ("Santa Fe (600 pts)", 0.051, 0.024, False, COURT),
    ("Parité L=2, D=10", 0.612, 0.204, False, EXACT),
    ("Markov dwell 20", 0.255, 0.247, False, COURT),
    ("NARMA-30 bruit σ=0,1", 0.666, 0.905, False, FINE),
    ("MNIST colonnes", 82.3, 70.1, True, FINE),
    ("MNIST spirale", 82.5, 83.2, True, COURT),
]
ratios = []
for name, t, e, cl, fam in data:
    r = (100 - t) / (100 - e) if cl else t / e
    ratios.append((r, name, fam))
ratios.sort()

col = {FINE: BLUE, COURT: ORANGE, EXACT: AQUA}
fig, ax = plt.subplots(figsize=(9.2, 6.6))
fig.patch.set_facecolor(SURF)
ax.set_facecolor(SURF)
for i, (r, name, fam) in enumerate(ratios):
    ax.hlines(i, 1, r, color=GRID, lw=2, zorder=1)
    ax.scatter(r, i, s=60, color=col[fam], edgecolor=SURF, linewidth=1.4, zorder=3)
    ax.text(r * (1.12 if r > 1 else 0.89), i, f"×{r:.2f}" if r < 10 else f"×{r:.0f}",
            va="center", ha="left" if r > 1 else "right", fontsize=8.5, color=INK2)
ax.set_yticks(range(len(ratios)))
ax.set_yticklabels([n for _, n, _ in ratios], fontsize=9, color=INK)
ax.set_xscale("log")
ax.set_xlim(0.08, 200)
ax.axvline(1, color=INK2, lw=1.2)
ax.text(0.95, len(ratios) - 0.2, "← le tenseur gagne", ha="right", fontsize=9, color=INK)
ax.text(1.05, len(ratios) - 0.2, "l'ESN gagne →", ha="left", fontsize=9, color=INK)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(INK2)
ax.tick_params(colors=INK2, labelsize=9)
ax.grid(axis="x", color=GRID, lw=0.8, which="major")
ax.set_axisbelow(True)
ax.set_xlabel("erreur tenseur / erreur ESN, même budget 400k connexions (échelle log)", fontsize=10, color=INK)
for fam, c in col.items():
    ax.scatter([], [], s=60, color=c, label=fam)
ax.legend(fontsize=9, frameon=False, loc="lower right")
ax.set_ylim(-0.8, len(ratios) + 0.3)
ax.set_title("Partie 1 en une image : le tenseur gagne exactement là où la cible\n"
             "a besoin d'informations anciennes, précises et distinctes", fontsize=11, color=INK, loc="left")
fig.tight_layout()
fig.savefig(RES / "partie1_bilan.svg", facecolor=SURF)
print("ok bilan")
