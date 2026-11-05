"""Partie 6 : figure 6.6 : trois NARMA modifiees (A/B/C).

Barres groupees : 8 ensembles x 4 variantes (std, A, B, C), meilleur rho
par seed puis moyenne, echelle log. Annotations des trois predictions.

Sortie : partie6_narma_modif.svg
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
res = json.load(open(f"{ROOT}/results_p6_narma_modif.json"))
agg = res["agg"]
SETS = ["{1,2,3,4,5,6,7,8,9,10}", "{2,3,4,5,6,7,8,9,10}", "{3,4,5,6,7,8,9,10}",
        "{2,4,6,8,10}", "{1,3,5,7,9}", "{1,10}", "{2,10}", "{3,10}"]
PRETTY = ["{1..10}", "{2..10}", "{3..10}", "{2,4,6,8,10}", "{1,3,5,7,9}", "{1,10}", "{2,10}", "{3,10}"]
VCOL = {"std": INK2, "A": BLUE, "B": ORANGE, "C": AQUA}


def style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


fig, ax = plt.subplots(figsize=(12.2, 5.8), facecolor=SURF)
style(ax)
xpos = np.arange(len(SETS))
w = 0.2
for i, v in enumerate(["std", "A", "B", "C"]):
    ys = []
    for s in SETS:
        row = agg["means"][f"{s}|{v}"]
        ys.append(min(r["mean"] for r in row.values()))
    ax.bar(xpos + (i - 1.5) * w, ys, w * 0.92, color=VCOL[v], edgecolor=SURF, lw=0.8,
           label={"std": "standard : u(t−1)·u(t−30)", "A": "A : u(t)·u(t−30)",
                  "B": "B : u(t−2)·u(t−30)", "C": "C : entrée lissée (moy. 3 pas)"}[v], zorder=3)
ax.set_yscale("log")
ax.set_ylim(1e-2, 2.0)
ax.set_xticks(xpos)
ax.set_xticklabels(PRETTY, fontsize=9.5, color=INK)
ax.set_ylabel("NRMSE NARMA-30, meilleur ρ ∈ {0,5 ; 0,9} (log)", fontsize=10, color=INK)
# annotations des predictions
ann = [(1, 0.40, "A réhabilite {2..10} :\n0,37 vs 0,35 (ratio 1,07)", BLUE, (0, 46)),
       (2, 0.76, "B fait tomber {3..10}\nau palier : 0,75", ORANGE, (30, 14)),
       (1, 1.0, "C réduit l'écart\n{2..10}−{1..10} de 78 %", AQUA, (58, -6))]
for x, y, txt, col, off in ann:
    ax.annotate(txt, (x, y), textcoords="offset points", xytext=off, fontsize=8.6, color=col,
               arrowprops=dict(arrowstyle="->", color=col, lw=1.0))
ax.axhline(0.757, color=INK, lw=0.9, ls=(0, (2, 2)), alpha=0.5)
ax.text(7.55, 0.79, "plancher $L_2$ (6.7) : 0,757", fontsize=8, color=INK2, ha="right")
ax.set_title("6.6 : déplacer le retard vital déplace la tranche vitale : les trois prédictions A, B, C tombent\n"
             "(moyenne 5 seeds, meilleur ρ par seed ; protocole 6.0)", fontsize=11.5, color=INK, loc="left")
ax.legend(fontsize=9, frameon=False, loc="upper right", ncol=1)
fig.tight_layout()
fig.savefig(f"{ROOT}/partie6_narma_modif.svg", facecolor=SURF)
print("ok narma_modif")
