"""Partie 4 : figure : le terme d'ordre 2 apparié contre tout le monde.

4 panneaux : parite2 / narma30 (échelle log : les terrains à produits
distants), parite6 / mc (linéaire : les terrains neutres). Barres = NRMSE
test (ou MC totale) moyenne sur 5 seeds à ρ optimal, ± std. Bleu = famille
4D (att = appariée, solide ; gen/ctrl pâles ; CP hachurées), aqua = tenseurs
3D (800k et empilé), gris = ESN 2×. Losange orange sur la meilleure config.

Sortie : ROOT/partie4_tenseur.svg (+ QA PNG).
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt
import os

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
DIAM = "#ff8c00"

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
conc = json.load(open(f"{ROOT}/results_p4_tenseur.json"))


def style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8.5)
    ax.grid(color=GRID, lw=0.7, axis="y")
    ax.set_axisbelow(True)


def rows_for(task):
    a = conc["agg"][task]
    r = [
        ("ESN 894", "esn800", INK2, ""),
        ("3D {1..10}", "3d800", AQUA, ""),
        ("2 couches", "3dstack", AQUA, "//"),
        ("4D (1,8)", "4d_gen", BLUE, ""),
        ("4D (1,1)", "4d_ctrl", BLUE, ""),
    ]
    if task == "parite2":
        r.insert(5, ("4D att.", "4d_att", BLUE, ""))
        r += [("CP r8", "4d_cp8", BLUE, "//"),
              ("CP r32", "4d_cp32", BLUE, "//")]
    elif task == "narma30":
        r.insert(5, ("4D att.", "4d_att", BLUE, ""))
        r += [("CP r8", "4d_cp8", BLUE, "//"),
              ("CP r32", "4d_cp32", BLUE, "//"),
              ("CP 16×6", "4d_cp16x6", BLUE, "//")]
    else:
        r.insert(5, ("4D att.", "4d_att", BLUE, ""))
    return [(lab, a[tag]["mean"], a[tag]["std"], col, h)
            for lab, tag, col, h in r if tag in a]


TASKS = [("parite2", "Parité L=2, D=10 (log) : paire appariée (10,11)", True),
         ("parite6", "Parité L=6, D=10 : personne ne sort du hasard", False),
         ("mc", "Capacité mémoire k=1..60 : paire (10,11)", False),
         ("narma30", "NARMA-30 (log) : paire appariée (1,30), CP-16×6 = 6 paires", True)]

fig, axs = plt.subplots(2, 2, figsize=(14.6, 9.2), facecolor=SURF)
for (task, ptitle, logscale), ax in zip(TASKS, axs.ravel()):
    style(ax)
    rows = rows_for(task)
    labels = [r[0] for r in rows]
    means = np.array([r[1] for r in rows])
    stds = np.array([r[2] for r in rows])
    x = np.arange(len(rows))
    bars = ax.bar(x, means, width=0.72, color=[r[3] for r in rows],
                  edgecolor="none", yerr=stds, ecolor=INK2, capsize=2.5,
                  error_kw=dict(lw=0.9))
    for b, r in zip(bars, rows):
        if r[4]:
            b.set_hatch(r[4])
            b.set_alpha(0.75)
        elif r[3] == BLUE and "appariée" not in r[0] and "CP" not in r[0]:
            b.set_alpha(0.45)
    ibest = int(np.argmin(means)) if task != "mc" else int(np.argmax(means))
    ax.plot(x[ibest], means[ibest], "D", color=DIAM, ms=8, mec=INK,
            mew=0.8, ls="none", zorder=5)
    base = conc["agg"][task]["3d800"]["mean"]
    if task == "mc":
        txt = ("4D appariée = " + f"{means[5]/base:.3f}".replace(".", ",")
               + " × la MC du 3D (aucun gain)")
        ax.set_ylim(0, 63)
        ax.axhline(60, color=INK2, lw=0.8, ls=":", alpha=0.7)
        ax.text(0.985, 0.965, "mémoire parfaite = 60", transform=ax.transAxes,
                fontsize=8.5, ha="right", va="top", color=INK2)
    else:
        ratio = means[5] / base
        ok = ratio < 0.9
        txt = ("appariée = " + f"{ratio:.3f}".replace(".", ",") + " × le 3D "
               + ("(prédiction P1/P4 : gagnée)" if ok else "(échec)"))
        ax.text(0.985, 0.965, txt, transform=ax.transAxes, fontsize=9.5,
                ha="right", va="top", color=INK if ok else ORANGE,
                fontweight="bold")
    ax.set_title(ptitle, fontsize=11, color=INK, loc="left")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.6, rotation=40, ha="right",
                       rotation_mode="anchor")
    ax.set_ylabel("NRMSE test (budget 2× = 800k, 5 seeds)" if task != "mc"
                  else "MC totale (max 60)", fontsize=9, color=INK2)
    if logscale:
        ax.set_yscale("log")
        ax.set_ylim(min(means) * 0.5, max(means) * 2.2)
    if task == "parite6":
        ax.axhline(1.0, color=ORANGE, lw=1.0, ls="dashed", alpha=0.8)
        ax.text(0.02, 1.012, "hasard = 1,0", transform=ax.get_yaxis_transform(),
                fontsize=8.5, color=ORANGE)
        ax.set_ylim(0.85, 1.10)

import matplotlib.patches as mpatches
leg = [mpatches.Patch(color=INK2, label="ESN 2× budget (N=894)"),
       mpatches.Patch(color=AQUA, label="tenseur 3D 800k (N=283)"),
       mpatches.Patch(facecolor=AQUA, hatch="//", label="2 couches 3D empilées"),
       mpatches.Patch(color=BLUE, label="4D paire appariée (dense)"),
       mpatches.Patch(facecolor=BLUE, alpha=0.45, label="4D générique (1,8) / contrôle carrés (1,1)"),
       mpatches.Patch(facecolor=BLUE, hatch="//", alpha=0.75, label="4D CP factorisé (rang r)")]
fig.legend(handles=leg, loc="lower center", ncol=3, fontsize=8.5,
           frameon=False, facecolor=SURF)
fig.suptitle("4.1-4.2 : le terme d'ordre 2 apparié écrase tout sur les terrains à produits distants : parité ×8,5, NARMA-30 ×15 contre le tenseur 3D : et la profondeur ne remplace pas l'ordre (2 couches : ×0,77 et ×0,99)\n"
             "4.3 : le rang compte : CP-8 perd plus de la moitié du gain, CP-32 en garde ~80 % ; six paires devinées font 3× pire qu'une paire bien choisie (falsifications P7/P8)",
             fontsize=11.5, color=INK, x=0.01, ha="left", y=1.03)
fig.tight_layout(rect=[0, 0.075, 1, 0.95])
fig.savefig(f"{ROOT}/partie4_tenseur.svg", facecolor=SURF, bbox_inches="tight")
print("ok p4")
