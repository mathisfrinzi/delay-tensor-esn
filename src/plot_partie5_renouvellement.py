"""Figure du profil de mémoire en champ moyen (rapport, partie 4).

Aucune simulation de réservoir : uniquement la récurrence scalaire
    e(0) = 1,   e(L) = Σ_{τ∈S, τ≤L} g_τ · e(L − τ)
où g_τ = ρ_τ² est le carré du rayon spectral de la tranche τ (pour des
tranches gaussiennes i.i.d. normalisées, E‖W_τ v‖² = ρ_τ² ‖v‖²).
e(L) est l'énergie moyenne de la trace laissée dans l'état par une impulsion
d'entrée vieille de L pas (réseau linéarisé, limite N grand).

    python src/plot_partie5_renouvellement.py     # écrit results/partie5_renouvellement.svg
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def style(ax, grid="y"):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    if grid:
        ax.grid(axis=grid, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def renewal(S, rho, Lmax, g=None):
    g = g if g is not None else [rho ** 2 / len(S)] * len(S)   # normalisation : ρ/√K par tranche
    e = np.zeros(Lmax + 1)
    e[0] = 1.0
    for L in range(1, Lmax + 1):
        e[L] = sum(gi * e[L - s] for s, gi in zip(S, g) if L >= s)
    return e


fig, axs = plt.subplots(1, 2, figsize=(13.5, 4.9), gridspec_kw={"width_ratios": [1.2, 1.1]})
fig.patch.set_facecolor(SURF)

# (a) même masse totale, répartition différente
ax = axs[0]
style(ax)
L = np.arange(0, 41)
cfgs = [("ESN, S = {1}", [1], BLUE), ("S = {1, …, 10}", list(range(1, 11)), ORANGE),
        ("S = {1, 10}", [1, 10], AQUA)]
for name, S, c in cfgs:
    e = renewal(S, 0.9, 40)
    ax.semilogy(L, e, "-", color=c, lw=2, marker="o", ms=3.2, label=name)
ax.set_ylim(1e-4, 1.5)
ax.set_xlabel("âge L de l'entrée (pas)", fontsize=10, color=INK)
ax.set_ylabel("énergie de la trace e(L)", fontsize=10, color=INK)
ax.set_title("(a) À ρ = 0,9, les trois profils ont la même aire\nΣ e(L) = 1/(1−ρ²) ≈ 5,26 : la mémoire est déplacée",
             fontsize=10.5, color=INK, loc="left")
ax.legend(fontsize=8.5, frameon=False, loc="lower left")

# (b) {1,10} : l'arithmétique devient tranchante quand ρ baisse
ax = axs[1]
style(ax)
Lb = np.arange(20, 41)
for rho, c, off, lab in ((0.9, ORANGE, -0.2, "ρ = 0,9"), (0.5, BLUE, 0.2, "ρ = 0,5")):
    e = renewal([1, 10], rho, 40)
    ax.bar(Lb + off, e[Lb], width=0.38, color=c, label=lab)
ax.set_yscale("log")
ax.set_ylim(1e-10, 3)
e5 = renewal([1, 10], .5, 40)
for x, txt in ((29, "29 : 11 sauts"), (30, "30 : 3 sauts")):
    ax.text(x + 0.2, max(e5[x], renewal([1, 10], .9, 40)[x]) * 4, txt, ha="center", fontsize=8.5, color=INK)
ax.set_xticks(range(20, 41, 2))
ax.set_xlabel("âge L de l'entrée (pas)", fontsize=10, color=INK)
ax.set_title("(b) S = {1, 10} : à ρ = 0,5, l'âge 30 (3 sauts) se détache\nde l'âge 29 (11 sauts) ; à ρ = 0,9, beaucoup moins",
             fontsize=10.5, color=INK, loc="left")
ax.legend(fontsize=8.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)
fig.tight_layout()
fig.savefig(RES / "partie5_renouvellement.svg", facecolor=SURF)
print("écrit partie5_renouvellement.svg")
