"""Partie 6 : figure 6.7 : les paliers sont des planchers d'information.

6 barres (3 lagsets x 2 versions de features) en echelle log + les 3
paliers measures des parties precedentes (0,32-0,39 / 0,75-0,84 / 1,03-1,10)
et la prediction NG-RC 0,178 de 2.4.

Sortie : partie6_planchers.svg
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SURF = "#fcfcfb"
ROOT = __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__))), "results")
res = json.load(open(f"{ROOT}/results_p6_planchers.json"))
agg = res["agg"]


def style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


fig, ax = plt.subplots(figsize=(9.2, 5.6), facecolor=SURF)
style(ax)
xpos = np.arange(3)
w = 0.34
for off, ver, col, lab in ((-w / 2, "prod", BLUE, "retards + produits 2 à 2 (1892 features)"),
                           (w / 2, "tanh", AQUA, "2000 neurones aléatoires tanh")):
    means = [agg[f"{ver}/{L}"]["mean"] for L in ["L1", "L2", "L3"]]
    stds = [agg[f"{ver}/{L}"]["std"] for L in ["L1", "L2", "L3"]]
    ax.bar(xpos + off, means, w, color=col, edgecolor=SURF, lw=1, label=lab, zorder=3)
    for x, m, s in zip(xpos + off, means, stds):
        ax.text(x, m * 1.35, f"{m:.4f}" if m < 0.01 else f"{m:.3f}", ha="center",
                va="bottom", fontsize=9, color=INK)
# paliers mesurés des parties 1-2 (reference)
paliers = [(0.32, 0.39, "palier bas mesuré\n(0,32-0,39)"), (0.75, 0.84, "palier du milieu\n(0,75-0,84)"),
           (1.03, 1.10, "palier haut = hasard\n(1,03-1,10)")]
for (lo, hi, lab), x in zip(paliers, xpos):
    ax.add_patch(plt.Rectangle((x - 0.43, lo), 0.86, hi - lo, color=INK2, alpha=0.12, lw=0, zorder=2))
    ax.text(x + 0.47, (lo + hi) / 2, lab, fontsize=8.3, color=INK2, va="center")
ax.axhline(0.15, color=ORANGE, lw=1.1, ls=(0, (4, 3)), zorder=2)
ax.text(2.55, 0.15, "prédiction $L_1$ < 0,15\n(NG-RC 2.4 : 0,178)", fontsize=8.3, color=ORANGE, va="center", ha="left")
ax.set_yscale("log")
ax.set_ylim(5e-4, 3)
ax.set_xlim(-0.75, 3.6)
ax.set_xticks(xpos)
ax.set_xticklabels(["$L_1$ = retards 0..60\n(u(t−1) ET u(t−30) connus)",
                    "$L_2$ sans 1\n(u(t−1) inconnu)",
                    "$L_3$ sans 1 ni 30\n(u(t−30) inconnu)"], fontsize=9.5, color=INK)
ax.set_ylabel("NRMSE plancher de l'oracle sans réservoir\n(T = 50 000, moyenne 5 seeds)", fontsize=10, color=INK)
ax.set_title("6.7 : les trois paliers de NARMA-30 sont des planchers d'information : sans u(t−1),\n"
             "aucun réservoir ne pouvait faire mieux que 0,75 ; sans u(t−30) non plus, rien", fontsize=11.5, color=INK, loc="left")
ax.legend(fontsize=9, frameon=False, loc="upper right")
fig.tight_layout()
fig.savefig(f"{ROOT}/partie6_planchers.svg", facecolor=SURF)
print("ok planchers")
