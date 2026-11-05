"""Partie 5 : figures de relecture « arithmétique des retards ».

Aucune simulation de réservoir : on relit les JSON déjà produits par les
parties 1 et 2, et on croise chaque configuration avec une quantité calculée
à la main : quels retards L sont ATTEIGNABLES depuis l'ensemble de tranches S
(L = somme d'éléments de S, répétitions permises ; L = 0 = entrée directe).

Figures produites :
  results/partie5_paliers_narma.svg  : les 104 configs NARMA en 3 paliers
  results/partie5_mnist_lignes.svg   : MNIST : précision vs lignes visibles

Mise à jour 6.7 (27/10/2026) : le panneau gauche de la figure des paliers
passe en échelle log et reçoit les trois PLANCHERS d'information mesurés
par le test 6.7 (oracle des retards sans réservoir, results_p6_planchers.json) :
le palier du milieu EST le plancher $L_2$, le palier du hasard EST le
plancher $L_3$ : seuls les ensembles qui voient u(t−1) restent au-dessus
de leur plancher.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"

# palette de référence (validée : 3 premiers slots, toutes paires)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SURF = "#fcfcfb"


def style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def reachable(S, Lmax):
    """r[L] = True si L s'écrit comme somme d'éléments de S (L=0 toujours vrai)."""
    r = np.zeros(Lmax + 1, bool)
    r[0] = True
    for L in range(1, Lmax + 1):
        r[L] = any(L >= s and r[L - s] for s in S)
    return r


def load_agg(name):
    return json.load(open(RES / name))["agg"]


# NARMA
pts = []  # (source, S, x, palier, nrmse)
for k, c in load_agg("results_narma_slices.json").items():          # 1.3ter : {k0..10} × x
    r = reachable(c["taus"], 60)
    pts.append(("1.3ter", c["taus"], c["x"], r[1], r[c["x"]], c["mean"]))
for f, src in (("results_tau20.json", "1.3.5"), ("results_ensembles.json", "2.1")):
    for k, c in load_agg(f).items():
        if "narma" not in str(c.get("task", k)):
            continue
        r = reachable(c["taus"], 60)
        pts.append((src, c["taus"], 30, r[1], r[30], c["mean"]))


def palier(l1, lx):
    if l1 and lx:
        return 0          # les deux retards utiles atteignables
    if lx:
        return 1          # retard x atteignable, retard 1 non
    return 2              # retard x inatteignable


# planchers d'information mesurés en 6.7 (oracle des retards, sans réservoir)
planchers = json.load(open(RES / "results_p6_planchers.json"))["agg"]
PL1 = planchers["prod/L1"]["mean"]     # u(t−1) ET u(t−30) connus
PL2 = planchers["prod/L2"]["mean"]     # sans u(t−1)
PL3 = planchers["prod/L3"]["mean"]     # sans u(t−1) ni u(t−30)

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.0), gridspec_kw={"width_ratios": [1.05, 1.25]})
fig.patch.set_facecolor(SURF)
style(ax)
rng = np.random.default_rng(0)
cols = [BLUE, ORANGE, AQUA]
labels = [
    "retards 1 ET x\natteignables",
    "retard x atteignable,\nretard 1 non",
    "retard x\ninatteignable",
]
for p in range(3):
    ys = np.array([q[5] for q in pts if palier(q[3], q[4]) == p])
    xs = p + rng.uniform(-0.22, 0.22, len(ys))
    ax.scatter(xs, ys, s=34, color=cols[p], edgecolor=SURF, linewidth=1.2, zorder=3)
    ax.text(p, ys.max() * 1.22, f"n = {len(ys)}\n[{ys.min():.2f} ; {ys.max():.2f}]",
            ha="center", va="bottom", fontsize=9, color=INK)
# les trois planchers d'information (6.7) : lignes horizontales
for pl, lab, ytxt in ((PL1, f"plancher $L_1$ = {PL1:.4f}", 0.0016),
                      (PL2, f"plancher $L_2$ = {PL2:.3f}  ← le palier du milieu EST le plancher", PL2 * 1.06),
                      (PL3, f"plancher $L_3$ = {PL3:.3f}  ← le « hasard » EST le plancher", PL3 * 1.06)):
    ax.axhline(pl, color=INK, lw=1.0, ls=(0, (5, 4)), alpha=0.65, zorder=2)
    ax.text(2.58, ytxt, lab, fontsize=8.4, color=INK, ha="right", va="bottom")
ax.set_xticks(range(3))
ax.set_xticklabels(labels, fontsize=9, color=INK)
ax.set_xlim(-0.6, 2.6)
ax.set_yscale("log")
ax.set_ylim(8e-4, 1.9)
ax.set_ylabel("NRMSE test (moyenne 3 seeds, échelle log)", fontsize=10, color=INK)
ax.set_title("104 configurations NARMA-x, aucun paramètre ajusté :\n"
             "trois paliers : et les paliers du haut sont des planchers d'information (6.7)",
             fontsize=11, color=INK, loc="left")

# panneau droit : la grille 1.3ter x × {k0..10}, cellule = NRMSE, symbole = palier prédit
agg = load_agg("results_narma_slices.json")
xs_ = sorted({c["x"] for c in agg.values()})
k0s = sorted({min(c["taus"]) for c in agg.values()})
Z = np.full((len(k0s), len(xs_)), np.nan)
P = np.zeros_like(Z, int)
for c in agg.values():
    i, j = k0s.index(min(c["taus"])), xs_.index(c["x"])
    r = reachable(c["taus"], 60)
    Z[i, j], P[i, j] = c["mean"], palier(r[1], r[c["x"]])
for i in range(len(k0s)):
    for j in range(len(xs_)):
        ax2.add_patch(plt.Rectangle((j - 0.46, i - 0.46), 0.92, 0.92, color=cols[P[i, j]],
                                    alpha=0.18 + 0.6 * min(1, (Z[i, j] - 0.3) / 0.8), lw=0))
        ax2.text(j, i, f"{Z[i, j]:.2f}", ha="center", va="center", fontsize=7.6, color=INK)
ax2.set_xlim(-0.5, len(xs_) - 0.5)
ax2.set_ylim(-0.5, len(k0s) - 0.5)
ax2.set_xticks(range(len(xs_)))
ax2.set_xticklabels([str(x) for x in xs_], fontsize=9, color=INK2)
ax2.set_yticks(range(len(k0s)))
ax2.set_yticklabels([f"{{{k}..10}}" if k < 10 else "{10}" for k in k0s], fontsize=9, color=INK2)
for s in ax2.spines.values():
    s.set_visible(False)
ax2.tick_params(length=0)
ax2.set_xlabel("ordre x de la NARMA (la cible a besoin de u(t−1) et u(t−x))", fontsize=10, color=INK)
ax2.set_title("Grille 1.3ter recolorée par le palier PRÉDIT : l'alternance pair/impair\n"
              "de {9,10} et {10} est de l'arithmétique (x = 9a + 10b ?)", fontsize=11, color=INK, loc="left")
fig.tight_layout()
fig.savefig(RES / "partie5_paliers_narma.svg", facecolor=SURF)
print("ok paliers", len(pts))

# MNIST
mp = []  # (source, S, nb lignes visibles, acc)
for f, src in (("results_tau20.json", "1.3.5 {k0..20}"), ("results_mnist_slices.json", "1.3quater {k0..10}"),
               ("results_ensembles.json", "2.1 pairs/impairs")):
    for k, c in load_agg(f).items():
        if "mnist" not in str(c.get("task", k)):
            continue
        rows = int(reachable(c["taus"], 27).sum())
        mp.append((src, c["taus"], rows, c["mean"] * 100, c["N"]))

fig, ax = plt.subplots(figsize=(8.6, 5.0))
fig.patch.set_facecolor(SURF)
style(ax)
marks = {"1.3.5 {k0..20}": (BLUE, "o"), "1.3quater {k0..10}": (ORANGE, "s"), "2.1 pairs/impairs": (AQUA, "D")}
for src, (c, m) in marks.items():
    q = [p for p in mp if p[0] == src]
    ax.scatter([p[2] for p in q], [p[3] for p in q], s=46, marker=m, color=c, edgecolor=SURF,
               linewidth=1.2, zorder=3, label=src)
offsets = {(20,): (10, -2), (10,): (-6, 10), (2, 4, 6, 8, 10): (-122, -3)}
for src, S, rows, acc, N in mp:
    if tuple(S) in offsets and not (src.startswith("2.1") and N == 200):
        name = "{" + ",".join(map(str, S)) + "}"
        ax.annotate(f"{name} : {rows} lignes", (rows, acc), xytext=offsets[tuple(S)], textcoords="offset points",
                    fontsize=8.5, color=INK2)
ax.axvline(14, color=INK2, lw=1, ls=(0, (4, 3)), zorder=1)
ax.text(14.4, 46, "au-delà de ~14 lignes :\nplateau 81-85 %, la position\net le N prennent le relais", fontsize=8.5,
        color=INK2, va="bottom")
rho = np.corrcoef(np.argsort(np.argsort([p[2] for p in mp])), np.argsort(np.argsort([p[3] for p in mp])))[0, 1]
ax.set_xlabel("nombre de lignes de l'image dont x(27) peut dépendre\n(distances 27 − ligne atteignables depuis S, sur 28)",
              fontsize=10, color=INK)
ax.set_ylabel("précision test MNIST (%)", fontsize=10, color=INK)
ax.set_title(f"MNIST ligne par ligne : la précision suit le nombre de lignes VISIBLES\n"
             f"(Spearman ρ = {rho:.2f} sur {len(mp)} configurations ; readout = état final seul)",
             fontsize=11, color=INK, loc="left")
ax.legend(fontsize=9, frameon=False, loc="lower right")
fig.tight_layout()
fig.savefig(RES / "partie5_mnist_lignes.svg", facecolor=SURF)
print("ok mnist", len(mp), round(rho, 3))
