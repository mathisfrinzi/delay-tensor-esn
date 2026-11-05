"""Bilan des contrôles (rapport, partie 6) : relecture des JSON de 7.1, 7.2, 7.3.

Pour chaque tâche, meilleure valeur sur la grille de rayon spectral, 5 seeds
(meilleur rho choisi sur la validation, valeur rapportée = test au rho choisi) :
ESN sans fuite (référence), tenseur dense (meilleur de S = {1..10} et {1, 10}),
tenseur à tranches gratuites (budget honnête, meilleur de permutation et diagonale,
mêmes deux ensembles), ESN à entrée retardée (meilleur
ensemble de retards d'entrée), ESN à fuite ou DTS-ESN (meilleure variante).

Valeur tracée : erreur / erreur de l'ESN sans fuite (NRMSE ; taux d'erreur pour
MNIST ; 1 / capacité mémoire pour MC). Échelle log, plus bas = meilleur.

    python src/plot_controles.py      # écrit results/controles_bilan.svg
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt
import numpy as np

RES = Path(__file__).resolve().parents[1] / "results"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#d9d8d3", "#fcfcfb"


def best(agg, prefix, higher):
    """(14.2) Meilleur rho choisi sur la VALIDATION (mean_val) ; la valeur
    retournee est la metrique de test au rho choisi, jamais un extremum
    pris sur le test."""
    cands = [(v.get("mean_val", v["mean"]), v["mean"]) for k, v in agg.items()
             if k.startswith(prefix)]
    pick = max if higher else min
    return pick(cands)[1]


tr = json.load(open(RES / "results_p7_tranches.json"))["agg"]
en = json.load(open(RES / "results_p7_entree.json"))["agg"]
dt = json.load(open(RES / "results_p7_dts.json"))["agg"]

tasks = [("narma30", "NARMA-30", False), ("silverbox", "Silverbox", False), ("wh", "Wiener-Hammerstein", False),
         ("mnist", "MNIST lignes", True), ("mc", "capacité mémoire", True)]


def err(task, v):
    if task == "mnist":
        return 1.0 - v
    if task == "mc":
        return 1.0 / v
    return v


rows = []
for t, name, hib in tasks:
    esn = best(tr, f"{t}|esn|dense", hib)
    pick = max if hib else min
    dense = pick(best(tr, f"{t}|A10|dense", hib), best(tr, f"{t}|A110|dense", hib))
    free = pick(best(tr, f"{t}|{S}|{k}|honest", hib) for S in ("A10", "A110") for k in ("perm", "diag"))
    inp = best(en, f"{t}|esn632|d", hib) if t in ("narma30", "silverbox", "wh") else None
    leak = (max if hib else min)(best(dt, f"{t}|leak_0.", hib), best(dt, f"{t}|dts_", hib)) if any(
        k.startswith(f"{t}|") for k in dt) else None
    rows.append((name, err(t, esn), err(t, dense), err(t, free), None if inp is None else err(t, inp),
                 None if leak is None else err(t, leak)))

series = [("tenseur dense", BLUE), ("tenseur, tranches gratuites", AQUA), ("ESN à entrée retardée", YELLOW),
          ("ESN à fuite ou DTS-ESN", ORANGE)]
fig, ax = plt.subplots(figsize=(12.5, 4.8))
fig.patch.set_facecolor(SURF)
ax.set_facecolor(SURF)
w = 0.19
for i, (name, esn, *vals) in enumerate(rows):
    for j, ((lab, c), v) in enumerate(zip(series, vals)):
        if v is None:
            continue
        x = i + (j - 1.5) * w
        ax.bar(x, v / esn, width=w * 0.9, color=c, label=lab if i == 0 or (j == 2 and i == 0) else None)
for j, (lab, c) in enumerate(series):
    ax.bar([np.nan], [np.nan], color=c, label=lab)
ax.axhline(1, color=INK2, lw=1.2)
ax.text(len(rows) - 0.45, 1.06, "ESN sans fuite", ha="right", fontsize=9, color=INK2)
ax.set_yscale("log")
ax.set_ylim(4e-3, 3)
ax.set_xticks(range(len(rows)))
ax.set_xticklabels([r[0] for r in rows], fontsize=10, color=INK)
ax.set_ylabel("erreur relative à l'ESN sans fuite (log)", fontsize=10, color=INK)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.tick_params(colors=INK2, labelsize=9)
ax.grid(axis="y", color=GRID, lw=0.8)
ax.set_axisbelow(True)
handles, labels = ax.get_legend_handles_labels()
seen = {}
for h, l in zip(handles, labels):
    seen.setdefault(l, h)
ax.legend(seen.values(), seen.keys(), fontsize=9, frameon=False, loc="upper center", ncol=4,
          bbox_to_anchor=(0.5, -0.1))
ax.set_title("Même budget de 400 000 connexions, meilleur rayon spectral, 5 seeds (MC : 1 / capacité ; "
             "MNIST : taux d'erreur)", fontsize=10.5, color=INK, loc="left")
fig.tight_layout()
fig.savefig(RES / "controles_bilan.svg", facecolor=SURF)
for r in rows:
    print(r[0], [None if v is None else round(v / r[1], 3) for v in r[2:]])
