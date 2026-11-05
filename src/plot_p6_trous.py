"""Partie 6 : figure 6.1 : les trous de memoire sont vides.

Grille 7 ensembles x 2 rho : MC_k par retard k (moyenne 5 seeds, echelle
log), retards atteignables = disques pleins, trous predits = cercles
vides, seuil de bruit p99 = ligne pointillee. Panneau droit : produits
u(t-a)u(t-b) pour {3,7} et {2,5} a rho=0,9 (prediction : non nuls ssi
a ET b atteignables).

Sortie : p6/../partie6_trous.svg (a pousser dans results/).
"""
import json, sys
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt
from p6_common import semigroup

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SURF = "#fcfcfb"
ROOT = __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__))), "results")
res = json.load(open(f"{ROOT}/results_p6_trous.json"))
SEUIL = res["agg"]["seuil_bruit_p99"]
SET_ORDER = ["S{1,2,3,4,5,6,7,8,9,10}", "S{9,10}", "S{3,7}", "S{2,5}", "S{4,6,9}",
             "S{2,4,6,8,10}", "S{16,17,18,19,20}"]
PRETTY = ["{1..10}", "{9,10}", "{3,7}", "{2,5}", "{4,6,9}", "{2,4,6,8,10}", "{16..20}"]


def style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)


def get_mc(key):
    cell = res["cells"][key]
    return np.mean([cell["seeds"][s]["mc"] for s in cell["seeds"]], 0)


fig = plt.figure(figsize=(14.5, 7.0), facecolor=SURF)
gs = fig.add_gridspec(2, 9, width_ratios=[1] * 7 + [0.12, 1.35], hspace=0.42, wspace=0.42)
axs = {}
for r, rho in enumerate([0.5, 0.9]):
    for c, (raw, pretty) in enumerate(zip(SET_ORDER, PRETTY)):
        ax = fig.add_subplot(gs[r, c])
        style(ax)
        key = f"{raw}|{rho}"
        S = res["cells"][key]["S"]
        att, _ = semigroup(S, 40)
        mc = np.maximum(np.array(get_mc(key)), 1e-7)
        ks = np.arange(1, 41)
        att_mask = np.array([att[k] for k in ks])
        ax.vlines(ks, 1e-5, mc, color=GRID, lw=0.7, zorder=1)
        ax.plot(ks[att_mask], mc[att_mask], "o", ms=3.4, color=BLUE, zorder=3,
                markeredgecolor=SURF, markeredgewidth=0.5)
        ax.plot(ks[~att_mask], mc[~att_mask], "o", ms=3.8, mfc="none", mec=INK,
                mew=0.9, zorder=3)
        ax.axhline(SEUIL, color=ORANGE, lw=1.0, ls=(0, (4, 3)), zorder=2)
        ax.set_yscale("log")
        ax.set_ylim(1e-5, 2)
        ax.set_xlim(0, 41)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        if r == 0:
            ax.set_title(pretty, fontsize=10, color=INK)
        if c == 0:
            ax.set_ylabel(f"ρ = {rho}\nMC_k (log)", fontsize=9, color=INK2)
        if r == 1:
            ax.set_xlabel("retard k", fontsize=8.5, color=INK2)

# panneau droit : produits pour {3,7} et {2,5} a rho=0.9
for r, raw in enumerate(["S{3,7}", "S{2,5}"]):
    ax = fig.add_subplot(gs[r, 8])
    style(ax)
    key = f"{raw}|0.9"
    cell = res["cells"][key]
    S = cell["S"]
    att, _ = semigroup(S, 20)
    Z = np.full((20, 20), np.nan)
    prods = {}
    for s in cell["seeds"].values():
        for ab, v in s.get("products", {}).items():
            prods.setdefault(ab, []).append(v)
    for ab, vals in prods.items():
        a, b = map(int, ab.lstrip('p').split("_"))
        Z[a - 1, b - 1] = np.maximum(np.mean(vals), 1e-7)
    ZZ = np.where(np.isnan(Z), 1e-7, Z)
    im = ax.imshow(ZZ, origin="lower", cmap="Blues", norm=matplotlib.colors.LogNorm(1e-5, 1.0),
                   extent=[0.5, 20.5, 0.5, 20.5], aspect="auto")
    # contours predits : a et b atteignables
    for a in range(1, 21):
        for b in range(a + 1, 21):
            if att[a] and att[b]:
                ax.add_patch(plt.Rectangle((b - 0.75, a - 0.75), 1.5, 1.5, fill=False,
                                           edgecolor=ORANGE, lw=1.0))
    ax.set_title(f"produits u(t−a)·u(t−b), {raw[1:-1]}, ρ=0,9\n(cadrés orange = a ET b atteignables)",
                 fontsize=8.5, color=INK, loc="left")
    ax.set_xlabel("b", fontsize=8.5, color=INK2)
    ax.set_ylabel("a", fontsize=8.5, color=INK2)
    for s in ax.spines.values():
        s.set_visible(False)

fig.suptitle("6.1 : les trous du semi-groupe ⟨S⟩ sont vides : MC = 1 sur les retards atteignables (disques pleins), "
             "sous le seuil de bruit ailleurs (cercles vides)", fontsize=12, color=INK, x=0.01, ha="left", y=0.995)
fig.text(0.01, 0.012, "MC_k = r² prédiction/cible sur le test, moyenne 5 seeds ; seuil = p99 de 2 800 cibles nulles "
         f"(≈ {SEUIL:.4f}). Produits non nuls ssi les deux retards sont atteignables.", fontsize=8.5, color=INK2)
fig.savefig(f"{ROOT}/partie6_trous.svg", facecolor=SURF, bbox_inches="tight")
print("ok trous")
