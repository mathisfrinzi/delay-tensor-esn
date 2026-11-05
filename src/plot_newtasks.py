"""Figure nouvelle 1.3 (nouvelles taches) : gauche = robustesse au bruit d'entree (NARMA-30),
droite = geometrie du parcours MNIST (lignes/colonnes/spirale). Convention maison :
svg.fonttype='none', annotations FR, ESN632 rouge, ESN200 gris, tenseur vert, log vert clair."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({"svg.fonttype": "none", "font.size": 9.5, "axes.titlesize": 11,
                     "axes.labelsize": 10, "legend.fontsize": 9})

BASE = Path(__file__).resolve().parent.parent / "results"
d = json.load(open(BASE / "results_newtasks.json"))

C = {"std632": "#d62728", "std200": "#9e9e9e", "tau10": "#1a9850", "log1248": "#66c2a5"}
NAMES = {"std632": "ESN N=632", "std200": "ESN N=200", "tau10": "Tenseur τ=1..10", "log1248": "Tenseur log {1,2,4,8}"}

fig, (ax, axm) = plt.subplots(1, 2, figsize=(12.4, 4.8))

# gauche : bruit
sigmas = [0.01, 0.05, 0.1, 0.2]
for cid in ["std632", "std200", "tau10", "log1248"]:
    v = [d[f"bruit_s{s}"][cid]["mean"] for s in sigmas]
    ax.plot(sigmas, v, "-o", color=C[cid], lw=2, ms=5, label=NAMES[cid])
ax.axhline(1.0, color="#777777", lw=0.9, ls="dashed")
ax.text(0.155, 1.02, "hasard (NRMSE=1)", fontsize=8.5, color="#777777", ha="right")
ax.annotate("ESN N=632 passe SOUS le hasard\n(le bruit recycle par la récurrence)",
            xy=(0.2, 1.053), xytext=(0.083, 0.87), fontsize=8.5, color="#555555",
            arrowprops=dict(arrowstyle="->", color="#555555"))
ax.set_xlabel("bruit d'entrée σ (u' = u + σ·η, u ∈ [0 ; 0,5])")
ax.set_ylabel("NRMSE (NARMA-30)")
ax.set_title("Robustesse au bruit d'entrée : l'hypothèse\nd'AMPLIFICATION par le tenseur est falsifiée (c'est l'inverse)")
ax.legend(frameon=False, loc="lower right")
ax.grid(alpha=0.25)

# droite : geometrie MNIST
geoms = ["lignes", "colonnes", "spirale"]
xs = np.arange(3)
for k, cid in enumerate(["std632", "std200", "tau10", "log1248"]):
    v = [d[f"mnist_{g}"][cid]["mean"] * 100 for g in geoms]
    axm.plot(xs, v, "-o", color=C[cid], lw=2, ms=6, label=NAMES[cid])
axm.set_xticks(xs, ["lignes\n(haut→bas)", "colonnes\n(gauche→droite)", "spirale\n(centre→extérieur)"])
axm.set_ylabel("accuracy MNIST (%)")
axm.set_title("Géométrie du parcours MNIST : le tenseur est plat,\nl'ESN passe de 64 % à 83 % : l'avantage suit le parcours")
axm.annotate("spirale = parcours spatialement local :\nl'ESN rattrape TOUT son retard", xy=(2, 83.2), xytext=(0.92, 88.5),
             fontsize=8.5, color="#555555", arrowprops=dict(arrowstyle="->", color="#555555"))
axm.legend(frameon=False, loc="lower right")
axm.grid(alpha=0.25)

fig.suptitle("1.3 : Nouvelles tâches : deux falsifications et une confirmation directionnelle (budget constant 400k)",
             fontsize=12, y=1.02)
fig.tight_layout()
fig.savefig(BASE / "newtasks.svg", bbox_inches="tight")
print("ok", (BASE / "newtasks.svg").stat().st_size, "octets")
