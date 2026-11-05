"""Figure 1.3quater : replique du sweep {k0..10} sur Silverbox / WH / MNIST vs NARMA-30.
Gauche : erreur relative a la meilleure config (echelle log) par tache. Droite : heatmap
des facteurs d'erreur relative, meilleure config par tache encadree orange.
Convention maison : svg.fonttype='none', annotations FR, NARMA=gris, SB=vert, WH=rouge, MNIST=bleu."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({"svg.fonttype": "none", "font.size": 9.5, "axes.titlesize": 11,
                     "axes.labelsize": 10, "legend.fontsize": 9})

BASE = Path(__file__).resolve().parent.parent / "results"
narma = json.load(open(BASE / "results_narma_slices.json"))["agg"]
phys = json.load(open(BASE / "results_phys_slices.json"))["agg"]
mnist = json.load(open(BASE / "results_mnist_slices.json"))["agg"]

k0s = list(range(1, 11))
labels = [f"{{{k0}..10}}" for k0 in k0s]

# erreur absolue par tache / config (MNIST -> taux d'erreur 1-acc)
err = {}
err["NARMA-30"] = np.array([narma[f"k{k0}_x30"]["mean"] for k0 in k0s])
err["Silverbox"] = np.array([phys[f"silverbox_k{k0}"]["mean"] for k0 in k0s])
err["WH"] = np.array([phys[f"wh_k{k0}"]["mean"] for k0 in k0s])
err["MNIST (taux d'erreur)"] = np.array([1 - mnist[f"mnist_k{k0}"]["mean"] for k0 in k0s])

rel = {t: v / v.min() for t, v in err.items()}
best_k0 = {t: int(np.argmin(v)) + 1 for t, v in err.items()}

colors = {"NARMA-30": "#9e9e9e", "Silverbox": "#1a9850", "WH": "#d62728", "MNIST (taux d'erreur)": "#1f77b4"}

fig, (ax, axh) = plt.subplots(1, 2, figsize=(12.6, 5.0), gridspec_kw={"width_ratios": [1.35, 1]})

for t, v in rel.items():
    ax.plot(k0s, v, "-o", color=colors[t], lw=2, ms=5, label=t, zorder=3)
    ax.plot(k0s[best_k0[t] - 1], v[best_k0[t] - 1], "D", color=colors[t], ms=9,
            markeredgecolor="white", zorder=4)
ax.axhline(1.0, color="#777777", lw=0.8, ls="dashed", zorder=1)
ax.annotate("sans τ=1", xy=(2, rel["NARMA-30"][1]), xytext=(3.3, 2.35),
            arrowprops=dict(arrowstyle="->", color="#555555"), color="#555555")
ax.text(1.02, 1.06, "meilleure config", fontsize=8.5, color="#777777")
ax.set_yscale("log")
ax.set_xticks(k0s, [f"k₀={k0}" for k0 in k0s])
ax.set_xlabel("tranches conservées : τ = {k₀..10}  (budget constant 400k, k₀=1 → {1..10})")
ax.set_ylabel("erreur relative à la meilleure config (log)")
ax.set_title("Retirer les tranches précoces : chaque tâche a son profil")
ax.legend(loc="upper left", frameon=False)
ax.grid(alpha=0.25)

M = np.array([rel[t] for t in ["NARMA-30", "Silverbox", "WH", "MNIST (taux d'erreur)"]])
im = axh.imshow(M, cmap="Reds", aspect="auto", vmin=0.95, vmax=2.1)
axh.set_xticks(range(10), labels, rotation=45)
axh.set_yticks(range(4), ["NARMA-30", "Silverbox", "WH", "MNIST"])
for i in range(4):
    for j in range(10):
        val = M[i, j]
        txtcol = "white" if val > 1.55 else "black"
        axh.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8.5, color=txtcol)
    j = best_k0[["NARMA-30", "Silverbox", "WH", "MNIST (taux d'erreur)"][i]] - 1
    axh.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="#ff8c00", lw=2.4))
axh.set_title("Erreur relative par ensemble τ (1,00 = meilleur)")
axh.set_xlabel("ensemble de tranches")
cb = fig.colorbar(im, ax=axh, shrink=0.85)
cb.set_label("erreur relative (1,0 = meilleure)", fontsize=8.5)

fig.suptitle("1.3quater : La couche critique est une propriété de la tâche : τ=1 vital sur NARMA, prunable sur les systèmes physiques et MNIST",
             fontsize=12, y=1.00)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(BASE / "phys_slices.svg", bbox_inches="tight")
print("ok", (BASE / "phys_slices.svg").stat().st_size, "octets")
print("best:", best_k0)
