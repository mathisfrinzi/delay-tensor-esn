"""Partie 6 : figure 6.4 : le crossover x*(rho) : la loi du seuil tombe.

(a) d(x) = log NRMSE_ESN - log NRMSE_tenseur par rho, ligne zero, losanges
aux x* mesures ; (b) x* mesure contre les predictions brute (ln theta /
2 ln rho) et corrigee kappa : la prediction explose, la mesure est plate ;
(c) x* en fonction de N_ESN a rho = 0,9.

Sortie : partie6_crossover.svg
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
res = json.load(open(f"{ROOT}/results_p6_crossover.json"))
a = res["agg"]
RHOS = [0.7, 0.8, 0.85, 0.9, 0.95]
RCOL = {0.7: "#0d2f61", 0.8: "#1f5aa8", 0.85: BLUE, 0.9: "#7fb3f0", 0.95: "#b9d4f7"}


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
                        gridspec_kw={"width_ratios": [1.15, 1, 0.75]})
fig.tight_layout(w_pad=2.6)
# (a) courbes d(x)
ax = axs[0]
style(ax)
for rho in RHOS:
    d = a["x_star"][str(rho)]["d"]
    xs, ds = [q[0] for q in d], [q[1] for q in d]
    ax.plot(xs, ds, "-", lw=1.4, color=RCOL[rho], label=f"ρ = {rho}", zorder=3)
    xm = a["x_star"][str(rho)]["x_mesure"]
    ax.plot([xm], [0], "D", ms=8, color=ORANGE, zorder=5, markeredgecolor=SURF)
ax.axhline(0, color=INK, lw=1.0, zorder=2)
ax.set_xlabel("ordre x de la NARMA-x", fontsize=10, color=INK)
ax.set_ylabel("log NRMSE_ESN − log NRMSE_tenseur", fontsize=10, color=INK)
ax.set_title("(a) le crossover est propre à mesurer (losanges = x*)\nmais il arrive toujours vers x ≈ 18-24",
             fontsize=10.5, color=INK, loc="left")
ax.legend(fontsize=8.5, frameon=False, loc="upper right")
# (b) x* vs rho : mesure contre predictions
ax = axs[1]
style(ax)
rhos = np.array(RHOS)
meas = [a["x_star"][str(r)]["x_mesure"] for r in RHOS]
ax.plot(rhos, meas, "D-", ms=9, lw=1.6, color=ORANGE, label="x* mesuré (10 seeds)", zorder=4)
rr = np.linspace(0.68, 0.96, 60)
ax.plot(rr, np.log(1e-2) / (2 * np.log(rr)), ls="dashed", lw=1.6, color=BLUE,
        label="prédit brut : ln θ / (2 ln ρ), θ = 10⁻²", zorder=3)
ax.plot(rhos, [a["x_star"][str(r)]["pred_kappa"] for r in RHOS], ":", lw=1.8, color=AQUA,
        label="prédit corrigé : ln θ / ln(κρ²)", zorder=3)
ax.set_xlabel("ρ (le même pour l'ESN et le tenseur)", fontsize=10, color=INK)
ax.set_ylabel("x* : profondeur où l'ESN décroche", fontsize=10, color=INK)
ax.set_ylim(0, 50)
ax.set_title("(b) FALSIFIÉ : la loi du seuil prédit une explosion 6,5 → 44,9 ;\nla mesure est quasi plate (17,9 → 24,1)",
             fontsize=10.5, color=INK, loc="left")
ax.legend(fontsize=8.5, frameon=False, loc="upper left")
# (c) balayage de N a rho=0.9
ax = axs[2]
style(ax)
Ns = [316, 632, 1000]
xm = [a["n_sweep_rho09"][f"N{N}"]["x_mesure"] for N in Ns]
ax.bar([str(n) for n in Ns], xm, 0.55, color=[AQUA, BLUE, "#0d2f61"], edgecolor=SURF, lw=1, zorder=3)
for i, v in enumerate(xm):
    ax.text(i, v + 0.5, f"{v:.1f}", ha="center", fontsize=9.5, color=INK)
ax.set_ylim(0, 30)
ax.set_xlabel("N de l'ESN (tenseur au même budget)", fontsize=10, color=INK)
ax.set_ylabel("x* à ρ = 0,9", fontsize=10, color=INK)
ax.set_title("(c) x* dépend aussi de N :\nle seuil θ n'est pas une constante",
             fontsize=10.5, color=INK, loc="left")

fig.suptitle("6.4 : crossover x*(ρ) : la loi du seuil 5.3 est falsifiée (1/5 prédictions dans la marge)\n"
             "l'observation de 1.4 (x* ≈ 22 à ρ = 0,9) reste ; c'est l'explication qui tombe",
             fontsize=11.5, color=INK, x=0.01, ha="left", y=1.07)
fig.savefig(f"{ROOT}/partie6_crossover.svg", facecolor=SURF, bbox_inches="tight")
print("ok crossover")
