"""Partie 6 : figure 6.5 : le champ moyen dans le vrai reseau.

4 panneaux (un par ensemble), N = 400 : e_emp(L) (trait plein) contre
e(L) de l'equation de renouvellement avec g = kappa rho^2/K (pointille),
pour rho = 0,5 et 0,9, echelle log. Les L inatteignables ({3,7}) sont
marques d'un diamant gris : e_emp y est EXACTEMENT nul (regle 5.1).

Sortie : partie6_champ_moyen.svg
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
res = json.load(open(f"{ROOT}/results_p6_champ_moyen.json"))
a = res["agg"]
SETLIST = ["{1}", "{1..10}", "{1,10}", "{3,7}"]


def style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(color=GRID, lw=0.7)
    ax.set_axisbelow(True)


fig, axs = plt.subplots(2, 2, figsize=(12.5, 8.0), facecolor=SURF)
fig.tight_layout(h_pad=2.6, w_pad=2.8)
for pi, sname in enumerate(SETLIST):
    ax = axs[pi // 2][pi % 2]
    style(ax)
    unreachable = []
    for rho, col in ((0.5, BLUE), (0.9, ORANGE)):
        c = a["par_config"][f"{sname}|N400|rho{rho}"]
        ee = np.array(c["e_emp_mean"])
        cell = res["cells"][f"{sname}|N400|rho{rho}"]["seeds"]
        ek = np.mean([np.array(cell[s]["e_kappa"]) for s in cell], axis=0)
        Ls = np.arange(len(ee))
        zer = [l for l in Ls if ee[l] <= 0]
        ee = np.maximum(ee, 1e-24)
        ax.plot(Ls, np.maximum(ee, 1e-24), "-", lw=1.6, color=col,
                label=f"e_emp, ρ = {rho}", zorder=3)
        ax.plot(Ls, np.maximum(ek, 1e-24), ls="dashed", lw=1.2, color=col, alpha=0.75,
                label=f"renouvellement κρ²/K, ρ = {rho}", zorder=2)
        zer = [l for l in Ls if ek[l] == 0]
        if zer:
            ax.plot(zer, [2e-24] * len(zer), "D", ms=4.5, color=INK2, zorder=4)
            unreachable = zer
        ax.text(0.985, 0.60 + (0.24 if rho == 0.9 else 0),
                f"err méd. ρ={rho} : {c['err_med_kappa']*100:.0f} %",
                transform=ax.transAxes, ha="right", fontsize=8.3, color=col)
    ax.set_yscale("log")
    ax.set_ylim(1e-24 if sname == "{3,7}" else 1e-8, 8)
    ax.set_xlim(-1, 61)
    ax.set_xlabel("profondeur L", fontsize=9.5, color=INK)
    if pi % 2 == 0:
        ax.set_ylabel("e(L) normalisé (log)", fontsize=9.5, color=INK)
    ttl = f"{sname}  (N = 400)"
    if unreachable:
        ttl += "  ◆ = 0 exact (L ∉ ⟨S⟩)"
    ax.set_title(ttl, fontsize=10.5, color=INK, loc="left")
    if pi == 0:
        ax.legend(fontsize=8, frameon=False, loc="upper right")

fig.suptitle("6.5 : l'équation de renouvellement décrit la forme (et les zéros exacts), mais dépasse le critère de 25 % :\n"
             "à ρ = 0,9 le vrai réseau, près du critique, retient l'énergie de perturbation plus longtemps que le champ moyen",
             fontsize=12, color=INK, x=0.01, ha="left", y=1.04)
fig.savefig(f"{ROOT}/partie6_champ_moyen.svg", facecolor=SURF, bbox_inches="tight")
print("ok champ")
