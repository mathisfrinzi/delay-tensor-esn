"""Figure 1.3.5 : sweep {k0..20} (solide) vs {k0..10} (pointille, 1.3quater) sur les 4 taches.
4 panneaux : NARMA-30, Silverbox, WH, MNIST. Convention maison : svg.fonttype='none',
annotations FR, meilleure config en losange vert, falaises annotees."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({"svg.fonttype": "none", "font.size": 9, "axes.titlesize": 10,
                     "axes.labelsize": 9, "legend.fontsize": 8})

BASE = Path(__file__).resolve().parent.parent / "results"
t20 = json.load(open(BASE / "results_tau20.json"))["cells"]
p10 = json.load(open(BASE / "results_phys_slices.json"))["cells"]
n10 = json.load(open(BASE / "results_narma_slices.json"))["cells"]
m10 = json.load(open(BASE / "results_mnist_slices.json"))["cells"]

def m(c):
    return float(np.mean(list(c["seeds"].values())))

TASKS = [
    ("NARMA-30", "narma30",
     lambda k: m(t20[f"narma30_k{k}"]), lambda k: m(n10[f"k{k}_x30"]), "NRMSE", (0.3, 1.2)),
    ("Silverbox", "silverbox",
     lambda k: m(t20[f"silverbox_k{k}"]), lambda k: m(p10[f"silverbox_k{k}"]), "NRMSE", (0.25, 1.15)),
    ("Wiener-Hammerstein", "wh",
     lambda k: m(t20[f"wh_k{k}"]), lambda k: m(p10[f"wh_k{k}"]), "NRMSE", (0.05, 1.05)),
    ("MNIST (accuracy)", "mnist",
     lambda k: m(t20[f"mnist_k{k}"]) * 100, lambda k: m(m10[f"mnist_k{k}"]) * 100, "acc (%)", (35, 90)),
]

fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.6))
for (title, key, f20, f10, ylab, ylim), ax in zip(TASKS, axes.flat):
    ks = list(range(1, 21))
    ks10 = list(range(1, 11))
    v20 = [f20(k) for k in ks]
    v10 = [f10(k) for k in ks10]
    ax.plot(ks, v20, "-o", color="#1a9850", lw=2, ms=4, label="bande jusqu'à 20 : τ={k₀..20}", zorder=3)
    ax.plot(ks10, v10, marker="s", ls="dashed", color="#9e9e9e", lw=1.6, ms=4, label="bande jusqu'à 10 : τ={k₀..10} (1.3quater)", zorder=2)
    kb = ks[int(np.argmin(v20))] if key != "mnist" else ks[int(np.argmax(v20))]
    vb = min(v20) if key != "mnist" else max(v20)
    ax.plot(kb, vb, "D", color="#ff8c00", ms=10, zorder=4, markeredgecolor="white")
    ax.axhline(1.0, color="#bbbbbb", lw=0.8, ls=":" )
    ax.set_title(title)
    ax.set_xlabel("première tranche conservée k₀")
    ax.set_ylabel(ylab)
    ax.set_ylim(*ylim)
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.grid(alpha=0.25)
    ax.legend(loc="best", frameon=False)
    # annotations specifiques
    if key == "narma30":
        ax.annotate("2e falaise : ≤5 tranches\nprofondes → pire que le hasard", xy=(16, 1.035), xytext=(8.2, 0.55),
                    fontsize=8, color="#555555", arrowprops=dict(arrowstyle="->", color="#555555"))
        ax.text(1.2, 0.40, "{1..20} ≈ {1..10} : τ=11..20 inertes", fontsize=8, color="#555555")
    if key == "silverbox":
        ax.annotate("la bande va AU-DELÀ de 20 :\n{2..20} = 0,306, nouveau record", xy=(2, 0.306), xytext=(4.5, 0.42),
                    fontsize=8, color="#555555", arrowprops=dict(arrowstyle="->", color="#555555"))
    if key == "wh":
        ax.annotate("falaise résolue : bande critique τ≈4..12,\ndégradation GRADUELLE ensuite (pas de saut)",
                    xy=(12, 0.314), xytext=(2.5, 0.72), fontsize=8, color="#555555",
                    arrowprops=dict(arrowstyle="->", color="#555555"))
    if key == "mnist":
        ax.annotate("étendre dilue (budget fragmenté) :\nmoins bon que le pic 84,2 % de {5..10}", xy=(1, 81.6), xytext=(2.5, 66),
                    fontsize=8, color="#555555", arrowprops=dict(arrowstyle="->", color="#555555"))
        ax.annotate("MAIS {10..20} = 82,4 % : la LARGEUR\ncompte, pas la position (cf. {10} seule : 65 %)",
                    xy=(10, 82.4), xytext=(7.5, 46), fontsize=8, color="#555555",
                    arrowprops=dict(arrowstyle="->", color="#555555"))

fig.suptitle("1.3.5 : Bande étendue τ={k₀..20} (budget constant 400k) : chaque tâche révèle sa bande critique et sa largeur minimale",
             fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(BASE / "tau20_slices.svg", bbox_inches="tight")
print("ok", (BASE / "tau20_slices.svg").stat().st_size, "octets")
