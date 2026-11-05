"""Partie 6 : figure 6.2 : balayage S = {1, d} sur NARMA-30.

4 panneaux : (a) NRMSE vs d par rho (log) + h(d) en escalier ;
(b) scatter NRMSE vs h(d) (Spearman) ; (c) NRMSE vs e(30) analytique ;
(d) top-5 avec rang faible r = N/4.

Sortie : partie6_balayage_d.svg
"""
import json, sys
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from p6_common import renewal_e

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SURF = "#fcfcfb"
ROOT = __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__))), "results")
res = json.load(open(f"{ROOT}/results_p6_balayage_d.json"))
agg = res["agg"]
DS = res["meta"]["d"]
RHOS = res["meta"]["rhos"]
RCOL = {0.1: "#d6e6fb", 0.2: "#aecdf6", 0.3: "#7fb3f0", 0.5: BLUE, 0.7: "#1f5aa8", 0.9: "#0d2f61"}


def style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(color=GRID, lw=0.7)
    ax.set_axisbelow(True)


fig, axs = plt.subplots(2, 2, figsize=(13.0, 8.6), facecolor=SURF)


# (a) NRMSE vs d par rho + h(d)
ax = axs[0, 0]
style(ax)
for rho in RHOS:
    ys = [agg["means"][f"d{d}|rho{rho}"]["mean"] for d in DS]
    ax.plot(DS, np.maximum(ys, 1e-4), "-o", ms=3.2, lw=1.3, color=RCOL[rho], label=f"ρ = {rho}", zorder=3)
ax.set_yscale("log")
ax.set_ylim(3e-3, 1.6)
ax2 = ax.twinx()
ax2.step(DS + [31, 45], [res["ana"][str(d)]["h"] for d in DS] + [30, 30], where="mid",
         color=INK2, lw=1.1, ls=(0, (4, 3)))
ax2.set_ylim(0, 42)
ax2.set_ylabel("h(d) : sauts minimaux pour écrire 30", fontsize=9, color=INK2)
ax2.tick_params(colors=INK2, labelsize=8)
for s in ("top",):
    ax2.spines[s].set_visible(False)
ax.axvspan(15.5, 19.5, color=ORANGE, alpha=0.08, lw=0)
ax.text(17.5, 0.0037, "d = 16..19 :\nh = 12 à 15 sauts", ha="center", fontsize=8.3, color=ORANGE)
for d, lab in ((10, "{1,10} : 0,052"), (15, "{1,15} : 0,014"), (30, "{1,30} : 0,006")):
    ax.annotate(lab, (d, agg["best"][str(d)]["nrmse"]), textcoords="offset points",
                xytext=(8, 6), fontsize=8.3, color=INK,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.7))
ax.set_xlabel("d (second retard, S = {1, d})", fontsize=10, color=INK)
ax.set_ylabel("NRMSE NARMA-30 (moyenne 5 seeds, log)", fontsize=10, color=INK)
ax.set_title("(a) la performance suit le nombre de sauts, pas la valeur de d :\n"
             "d = 15 → 0,014, d = 16 → 0,754 (40× pire)", fontsize=10, color=INK, loc="left")
ax.legend(fontsize=9, frameon=False, loc="center left")

# (b) scatter NRMSE vs h(d)
ax = axs[0, 1]
style(ax)
ys = [agg["means"][f"d{d}|rho0.5"]["mean"] for d in DS]
hs = [res["ana"][str(d)]["h"] for d in DS]
ax.scatter(hs, ys, s=40, color=BLUE, edgecolor=SURF, lw=1, zorder=3)
rho_s = spearmanr(hs, ys).statistic
for d in [10, 15, 16, 29, 30]:
    ax.annotate(f"d={d}", (res["ana"][str(d)]["h"], agg["means"][f"d{d}|rho0.5"]["mean"]),
                textcoords="offset points", xytext=(6, 4), fontsize=8.3, color=INK2)
ax.set_yscale("log")
ax.set_ylim(3e-3, 1.6)
ax.set_xlabel("h(d) : sauts minimaux pour écrire 30 depuis {1, d}", fontsize=10, color=INK)
ax.set_ylabel("NRMSE à ρ = 0,5 (log)", fontsize=10, color=INK)
ax.set_title(f"(b) P1 confirmé : Spearman(NRMSE, h) = {rho_s:.2f} à ρ = 0,5\n(prédiction : ≥ 0,6)", fontsize=10, color=INK, loc="left")

# (c) NRMSE vs e(30) analytique
ax = axs[1, 0]
style(ax)
for rho in RHOS:
    xs = [max(res["ana"][str(d)][f"e30_rho{rho}"], 1.5e-20) for d in DS]
    ys = [agg["means"][f"d{d}|rho{rho}"]["mean"] for d in DS]
    ax.scatter(xs, ys, s=30, color=RCOL[rho], label=f"ρ = {rho}", zorder=3, edgecolor=SURF, lw=0.7)
ax.set_xscale("log")
ax.set_xlim(1e-20, 3)
ax.set_yscale("log")
ax.set_xlabel("e(30) analytique (renouvellement 5.2, échelle log ; sous 10⁻²⁰ : ramené au bord)", fontsize=10, color=INK)
ax.set_ylabel("NRMSE (log)", fontsize=10, color=INK)
ax.set_title("(c) e(30) seule ne suffit pas : au-dessus de 10⁻², tout est bon\n"
             "(1 à 3 sauts) ; le tri fin se fait par l'isolement ι (5.4)", fontsize=10, color=INK, loc="left")
ax.legend(fontsize=9, frameon=False)

# (d) top-5 rang faible
ax = axs[1, 1]
style(ax)
top5 = agg.get("top5_rank", {})
labels, means_std, means_rank = [], [], []
for dstr in sorted(top5, key=lambda k: top5[k]["nrmse"]):
    rho = top5[dstr]["rho"]
    means_std.append(agg["best"][dstr]["nrmse"])
    rk = agg["means"].get(f"d{dstr}_r|rho{rho}")
    means_rank.append(rk["mean"] if rk else np.nan)
    labels.append(f"{{1,{dstr}}}\nρ={rho}")
xpos = np.arange(len(labels))
w = 0.36
ax.bar(xpos - w / 2, means_std, w, color=BLUE, edgecolor=SURF, lw=1, label="standard", zorder=3)
ax.bar(xpos + w / 2, means_rank, w, color=AQUA, edgecolor=SURF, lw=1, label="rang r = N/4", zorder=3)
for x, v in zip(xpos - w / 2, means_std):
    ax.text(x, v * 1.12, f"{v:.4f}", ha="center", fontsize=8, color=INK)
for x, v in zip(xpos + w / 2, means_rank):
    ax.text(x, v * 1.12, f"{v:.4f}", ha="center", fontsize=8, color=INK)
ax.set_yscale("log")
ax.set_ylim(3e-3, 0.4)
ax.set_xticks(xpos)
ax.set_xticklabels(labels, fontsize=9, color=INK)
ax.set_ylabel("NRMSE (log)", fontsize=10, color=INK)
ax.set_title("(d) le rang faible n'améliore pas les nouveaux champions\n"
             "(contrairement au record {1,10} de 2.2 : 0,051 → 0,035)", fontsize=10, color=INK, loc="left")
ax.legend(fontsize=9, frameon=False)

fig.suptitle(f"Balayage S = {{1, d}} : l'arithmétique des sauts domine ({len(DS)} valeurs de d, {len(RHOS)} ρ, 5 seeds, {len(DS) * len(RHOS) * 5} runs)",
             fontsize=12.5, color=INK, x=0.01, ha="left", y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.95], h_pad=3.5, w_pad=5.0)
fig.savefig(f"{ROOT}/partie6_balayage_d.svg", facecolor=SURF, bbox_inches="tight")
print("ok balayage")
