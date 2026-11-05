"""Partie 8 : figure 8.1 : spectre de retards des quatre taches.

4 panneaux (NARMA-30, NARMA-25, Silverbox, WH) : importance par ablation
simple I(L) (tiges bleues, normalisees par le max du panneau) et par blocs de
3 (barres bleu clair), losanges orange sur L*. Les deux NARMA ont des pics
nets ; les systemes physiques sont plats (trou mou, 5.1).

Sortie : partie8_spectre.svg (+ QA PNG).
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
DIAM = "#ff8c00"

ROOT = __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__))), "results")
res = json.load(open(f"{ROOT}/results_p8_spectre.json"))
LAGS = np.arange(0, 61)
BLOCKS = [list(range(b, min(b + 3, 61))) for b in range(0, 61, 3)]


def style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(color=GRID, lw=0.7)
    ax.set_axisbelow(True)


TASKS = [("narma30", "NARMA-30 : pics nets : I(1) = 0,757, I(30) = 0,756"),
         ("narma25", "NARMA-25 : miroir : I(25) = 0,755, I(1) = 0,752"),
         ("silverbox", "Silverbox : plat : max I(L) = 0,0067 (trou mou)"),
         ("wh", "Wiener-Hammerstein : ultra-plat : max I(L) ≈ 0 (trou mou intégral)")]

fig, axs = plt.subplots(2, 2, figsize=(13.2, 7.2), facecolor=SURF)
for (task, title), ax in zip(TASKS, axs.ravel()):
    style(ax)
    a = res["agg"][task]
    I = np.array(a["I_single_mean"])
    B = np.array(a["I_block_mean"])
    mx = max(I.max(), B.max())
    if mx <= 0:
        mx = 1.0
    In, Bn = I / mx, B / mx
    # blocs : barres bleu clair au centre du bloc
    bc = np.array([np.mean(b) for b in BLOCKS])
    ax.bar(bc, Bn, width=2.4, color=BLUE, alpha=0.28, edgecolor="none",
           label="ablation par blocs de 3")
    ax.vlines(LAGS, 0, In, color=BLUE, lw=1.6, label="ablation simple I(L)")
    if task in ("narma30", "narma25"):
        # pics nets : losanges sur les retards utiles L*
        L_star = a["L_star"]
        ax.plot(L_star, [In[l] for l in L_star], "D", color=DIAM, ms=7,
                mec=INK, mew=0.7, ls="none", label="retards utiles (L*)")
    else:
        # plat : losanges sur les blocs utiles B*
        B_star = a["B_star"]
        idx = [b[0] // 3 for b in B_star]
        ax.plot([np.mean(b) for b in B_star], [Bn[i] for i in idx],
                "D", color=DIAM, ms=7, mec=INK, mew=0.7, ls="none",
                label="blocs utiles (B*)")
    ax.set_title(title, fontsize=11, color=INK, loc="left")
    ax.set_xlim(-2.5, 62)
    ax.set_ylim(0, 1.12)
    ax.set_xticks([0, 10, 20, 30, 40, 50, 60])
    ax.set_xlabel("retard L", fontsize=9, color=INK2)
    ax.set_ylabel("I(L) normalisé", fontsize=9, color=INK2)
    if task == "silverbox":
        ax.annotate("bloc [57-59] : 0,0051\n(bis insolite)", xy=(58, Bn[19]),
                    xytext=(42, 0.62), fontsize=8.5, color=INK2,
                    arrowprops=dict(arrowstyle="->", color=INK2, lw=0.8))
    if task == "wh":
        ax.annotate("blocs utiles : 0-5 et 15-26\n(à 0,0005 près : tout est mou)",
                    xy=(20, 0.9), xytext=(30, 0.62), fontsize=8.5, color=INK2,
                    arrowprops=dict(arrowstyle="->", color=INK2, lw=0.8))

handles, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9,
           frameon=False, bbox_to_anchor=(0.5, 0.015))
fig.suptitle("8.1 : le spectre de retards mesuré par l'oracle : pics nets sur NARMA, plat intégral sur les systèmes physiques\n"
             "L* = {L : I(L) > 5 % de max I} pour NARMA ; par blocs de 3 pour Silverbox/WH (retards mous) : la prédiction de 8.1 tombe",
             fontsize=11.5, color=INK, x=0.01, ha="left", y=1.03)
fig.tight_layout(rect=[0, 0.05, 1, 0.97])
fig.savefig(f"{ROOT}/partie8_spectre.svg", facecolor=SURF, bbox_inches="tight")
print("ok spectre")
