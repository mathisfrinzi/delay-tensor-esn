"""Partie 6 : figure 6.3 : la dent de scie en x.

(a)(b) NRMSE(x) pour 4 ensembles, rho=0.5 et 0.9 (log), lignes verticales
aux multiples de 10 ; (c) autocorrelation detrendee a rho=0.5 pour {1,10}
et {1,7} : pics aux decalages 10 et 7.

Sortie : partie6_dents.svg
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
res = json.load(open(f"{ROOT}/results_p6_dents.json"))
a = res["agg"]
XS = res["meta"]["xs"]
SETS = [("{1,10}", BLUE, "{1,10}"), ("{1,7}", ORANGE, "{1,7}"),
        ("{1..10}", AQUA, "{1..10}"), ("esn", INK2, "ESN (N=632)")]


def style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(color=GRID, lw=0.7)
    ax.set_axisbelow(True)


fig, axs = plt.subplots(1, 3, figsize=(14.5, 4.9), facecolor=SURF,
                        gridspec_kw={"width_ratios": [1, 1, 0.85]})
fig.tight_layout(w_pad=2.6)
for pi, rho in enumerate([0.5, 0.9]):
    ax = axs[pi]
    style(ax)
    for sname, col, lab in SETS:
        ys = [a["means"][f"{sname}|x{x}|rho{rho}"]["mean"] for x in XS]
        ax.plot(XS, ys, "-o", ms=3.4, lw=1.4, color=col, label=lab, zorder=3)
    for v in [20, 30, 40]:
        ax.axvline(v, color=INK, lw=0.9, ls=(0, (3, 3)), alpha=0.35, zorder=1)
    ax.text(20, 0.0035, "20", ha="center", fontsize=8, color=INK2)
    ax.text(30, 0.0035, "30", ha="center", fontsize=8, color=INK2)
    ax.text(40, 0.0035, "40", ha="center", fontsize=8, color=INK2)
    ax.set_yscale("log")
    ax.set_ylim(3e-3, 1.6)
    ax.set_xlabel("ordre x de la NARMA-x", fontsize=10, color=INK)
    if pi == 0:
        ax.set_ylabel("NRMSE (moyenne 5 seeds, log)", fontsize=10, color=INK)
        ax.legend(fontsize=9, frameon=False, loc="upper left")
    ax.set_title(f"({'a' if pi == 0 else 'b'}) ρ = {rho}" +
                 (" : trois dents complètes, creux aux multiples de 10" if pi == 0 else
                  " : dents ≈ 2× plus faibles"), fontsize=10.5, color=INK, loc="left")
# (c) ACF detrendee rho=0.5
ax = axs[2]
style(ax)
for sname, col in [("{1,10}", BLUE), ("{1,7}", ORANGE)]:
    acf = a[f"acf_{sname}_rho0.5"]["acf"]
    lags = np.arange(len(acf))
    ax.plot(lags[1:], acf[1:], "-o", ms=4, lw=1.3, color=col, label=sname, zorder=3)
    pk = a[f"acf_{sname}_rho0.5"]["peak_lag"]
    ax.plot([pk], [acf[pk]], "D", ms=8, color=ORANGE, zorder=4, markeredgecolor=SURF)
    ax.annotate(f"pic = {pk}", (pk, acf[pk]), textcoords="offset points",
                xytext=(10, -2), fontsize=9, color=INK)
ax.axhline(0, color=INK2, lw=0.8)
ax.set_xlabel("décalage de l'autocorrélation de NRMSE(x)", fontsize=9.5, color=INK)
ax.set_title("(c) période 10 pour {1,10},\npériode 7 pour {1,7} (ρ = 0,5)", fontsize=10.5, color=INK, loc="left")
ax.legend(fontsize=9, frameon=False, loc="upper right")

fig.suptitle("6.3 : la dent de scie en x : changer l'ordre de NARMA d'un pas fait sauter la performance\n"
             "exactement où l'arithmétique des sauts le prédit (NRMSE(29)/NRMSE(30) = 12,5, prédit ≥ 2)",
             fontsize=12, color=INK, x=0.01, ha="left", y=1.06)
fig.savefig(f"{ROOT}/partie6_dents.svg", facecolor=SURF, bbox_inches="tight")
print("ok dents")
