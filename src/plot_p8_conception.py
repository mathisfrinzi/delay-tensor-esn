"""Partie 8 : figure 8.3 : validation à budget egal (400k) sur 4 taches.

4 panneaux : NARMA-30 et NARMA-25 (echelle log) puis Silverbox et
Wiener-Hammerstein (lineaire). Barres = NRMSE test moyenne sur 5 seeds +- std,
a rho optimal par config. Bleu = calcule (8.2), orange = elus empiriques
(pruning / bande, Partie 2), aqua = uniforme {1..10}, gris = baselines
(ESN tau=0, variantes a gains min-i degenerees). Losange orange sur la
meilleure config de chaque tache.

Sortie : partie8_conception.svg (+ QA PNG).
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
DIAM = "#ff8c00"

ROOT = __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__))), "results")
conc = json.load(open(f"{ROOT}/results_p8_conception.json"))
calc = json.load(open(f"{ROOT}/results_p8_calcul.json"))

# elus empiriques (Partie 2) et supports calcules (8.2) pour les etiquettes
ELUS = {"narma30": {"prune_elu": "{1,10}", "band_elu": "[1 ; 12,4]"},
        "narma25": {"prune_elu": "{1,4}", "band_elu": "[1 ; 12,8]"},
        "silverbox": {"prune_elu": "{2,3,4,5,7,8,10}", "band_elu": "[2,0 ; 17,0]"},
        "wh": {"prune_elu": "{4,7}", "band_elu": "[3,2 ; 10,1]"}}


def fmt_supp(S):
    return "{" + ",".join(str(s) for s in S) + "}"


def style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8.5)
    ax.grid(color=GRID, lw=0.7, axis="y")
    ax.set_axisbelow(True)


TASKS = [("narma30", "NARMA-30 (échelle log)", True, "retards durs"),
         ("narma25", "NARMA-25 (échelle log)", True, "retards durs"),
         ("silverbox", "Silverbox", False, "trou mou"),
         ("wh", "Wiener-Hammerstein", False, "trou mou")]

fig, axs = plt.subplots(2, 2, figsize=(13.6, 9.0), facecolor=SURF)
for (task, ptitle, logscale, nature), ax in zip(TASKS, axs.ravel()):
    style(ax)
    a = conc["agg"][task]
    supp = {k: calc["tasks"][task]["support"][k]["best"]["S"] for k in ("2", "3", "4")}
    rows = [
        ("calculé k=2\n" + fmt_supp(supp["2"]), a["calc_k2"], BLUE, ""),
        ("calculé k=3\n" + fmt_supp(supp["3"]), a["calc_k3"], BLUE, ""),
        ("calculé k=4\n" + fmt_supp(supp["4"]), a["calc_k4"], BLUE, ""),
        ("k=2 + gains\nmin-ι dégénéré", a["calc_k2_g"], INK2, "//"),
        ("k=3 + gains\nmin-ι dégénéré", a["calc_k3_g"], INK2, "//"),
        ("k=4 + gains\nmin-ι dégénéré", a["calc_k4_g"], INK2, "//"),
        ("élu pruning\n" + ELUS[task]["prune_elu"], a["prune_elu"], ORANGE, ""),
        ("élu bande\n" + ELUS[task]["band_elu"], a["band_elu"], ORANGE, ""),
        ("uniforme {1..10}", a["unif_1_10"], AQUA, ""),
        ("ESN (τ = 0)", a["esn"], INK2, ""),
    ]
    labels, means, stds, colors, hatches = [], [], [], [], []
    for lab, d, col, h in rows:
        labels.append(lab)
        means.append(d["mean"])
        stds.append(d["std"])
        colors.append(col)
        hatches.append(h)
    x = np.arange(len(rows))
    bars = ax.bar(x, means, width=0.72, color=colors, edgecolor="none",
                  yerr=stds, ecolor=INK2, capsize=2.5, error_kw=dict(lw=0.9))
    for b, h in zip(bars, hatches):
        if h:
            b.set_hatch(h)
            b.set_alpha(0.55)
    # losange sur la meilleure config
    ibest = int(np.argmin(means))
    ax.plot(x[ibest], means[ibest], "D", color=DIAM, ms=8, mec=INK, mew=0.8,
            ls="none", zorder=5)
    # annotation ratio calcule k=2 / meilleur elu
    elu_best = min(a["prune_elu"]["mean"], a["band_elu"]["mean"])
    ratio = a["calc_k2"]["mean"] / elu_best
    ok = ratio <= 1.10
    txt = ("calc_k2 = " + f"{ratio:.2f}".replace(".", ",") + " × l'élu "
           + ("(validé)" if ok else "(échec)"))
    ax.set_title(ptitle + " : " + nature, fontsize=11, color=INK, loc="left")
    ax.text(0.985, 0.965, txt, transform=ax.transAxes, fontsize=9.5,
            ha="right", va="top", color=INK if ok else ORANGE, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.2, rotation=32, ha="right", rotation_mode="anchor")
    ax.set_ylabel("NRMSE test (budget 400k, 5 seeds)", fontsize=9, color=INK2)
    if logscale:
        ax.set_yscale("log")
        ax.set_ylim(min(means) * 0.55, max(max(means), 1.0) * 1.6)
        shown = [t for t in (0.001, 0.01, 0.1, 1.0)
                 if min(means) * 0.55 <= t <= max(means) * 1.6]
        names = {0.001: "0,001", 0.01: "0,01", 0.1: "0,1", 1.0: "1"}
        ax.set_yticks(shown)
        ax.set_yticklabels([names[t] for t in shown])
        ax.get_yaxis().set_minor_locator(matplotlib.ticker.NullLocator())
    else:
        ax.set_ylim(0, max(means) * 1.28)

fig.suptitle("8.3 : validation à budget égal : le calcul bat la force brute sur les retards durs (NARMA : ×9 et ×48, zéro entraînement)\n"
             "mais échoue sur les trous mous (Silverbox ×1,4, WH ×1,6) : critère « ≤ 1,1 × l'élu sur 3/4 tâches » : 2/4, échec instructif",
             fontsize=11.5, color=INK, x=0.01, ha="left", y=1.03)
fig.tight_layout(rect=[0, 0.02, 1, 0.95])
fig.savefig(f"{ROOT}/partie8_conception.svg", facecolor=SURF, bbox_inches="tight")
print("ok conception")
