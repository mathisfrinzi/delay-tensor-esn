"""Schémas pédagogiques du rapport (un par partie).

Aucune simulation de réservoir : dessins, petits calculs exacts (sommes de retards,
récurrence de renouvellement) et relecture de results/results_p8_spectre.json.

    python src/schemas.py            # écrit results/schemas/*.svg
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "schemas"
OUT.mkdir(parents=True, exist_ok=True)

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#d9d8d3", "#fcfcfb"
PALE_BLUE, PALE_ORANGE, PALE_AQUA = "#dbe8f8", "#fbe1d5", "#d6f0e5"
DASH = (0, (4, 3))


# outils
def blank(ax):
    ax.set_facecolor(SURF)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def styled(ax, grid="y"):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    if grid:
        ax.grid(axis=grid, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def box(ax, x, y, w, h, text, fc=PALE_BLUE, ec=BLUE, size=10, weight="normal", color=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ec, lw=1.4))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size,
            color=color, weight=weight)


def arrow(ax, p, q, color=INK2, rad=0.0, lw=1.6, style="-|>", ms=12, ls="solid"):
    ax.add_patch(FancyArrowPatch(p, q, connectionstyle=f"arc3,rad={rad}", arrowstyle=style,
                                 mutation_scale=ms, color=color, lw=lw, linestyle=ls))


def save(fig, name):
    fig.patch.set_facecolor(SURF)
    fig.savefig(OUT / f"{name}.svg", facecolor=SURF, bbox_inches="tight")
    plt.close(fig)
    print("écrit", name)


def reachable(S, lmax):
    r = np.zeros(lmax + 1, bool)
    r[0] = True
    for L in range(1, lmax + 1):
        r[L] = any(L >= s and r[L - s] for s in S)
    return r


def hops(S, lmax):
    h = [0] + [10 ** 6] * lmax
    for L in range(1, lmax + 1):
        h[L] = min((h[L - s] + 1 for s in S if L >= s), default=10 ** 6)
    return np.array(h)


def renewal(S, g, lmax, leak=1.0):
    e = np.zeros(lmax + 1)
    e[0] = 1.0
    for L in range(1, lmax + 1):
        v = sum(gi * e[L - s] for s, gi in zip(S, g) if L >= s) * leak ** 2
        if leak < 1.0:
            v += (1.0 - leak) ** 2 * e[L - 1]
        e[L] = v
    return e


def timeline(ax, y, n, active, labels_w=True, color=BLUE, pale=PALE_BLUE, highlight=None, arrows=None):
    """Cases x(t-n) ... x(t) sur une ligne ; flèches depuis les retards de `active`."""
    xs = {k: 1.2 * (n - k) for k in range(n + 1)}
    for k in range(n + 1):
        fc = pale if k in active or k == 0 else "#f1f0ec"
        ec = color if k in active or k == 0 else GRID
        if highlight and k in highlight:
            fc, ec = PALE_ORANGE, ORANGE
        ax.add_patch(FancyBboxPatch((xs[k] - 0.45, y - 0.35), 0.9, 0.7,
                                    boxstyle="round,pad=0.02,rounding_size=0.1", fc=fc, ec=ec, lw=1.2))
        lab = "x(t)" if k == 0 else f"x(t−{k})"
        ax.text(xs[k], y, lab, ha="center", va="center", fontsize=8.5, color=INK)
    for i, k in enumerate(sorted(active if arrows is None else arrows)):
        arrow(ax, (xs[k], y + 0.36), (xs[0] - 0.05, y + 0.36), color=color, rad=-0.4, lw=1.5)
        if labels_w:
            ax.text(xs[k] + 0.12, y + 0.55, f"W$_{{{k}}}$", ha="left", fontsize=10, color=color)
    return xs


# 1. le modèle et S
def schema_modele():
    fig = plt.figure(figsize=(14.5, 5.6))
    ax = fig.add_axes([0.0, 0.0, 0.66, 1.0])
    blank(ax)
    ax.set_xlim(-1.0, 10.6)
    ax.set_ylim(-3.1, 3.9)
    ax.text(-0.9, 3.55, "ESN : l'état ne lit que l'instant précédent", fontsize=12, color=INK, weight="bold")
    timeline(ax, 2.0, 8, {1}, color=INK2, pale="#ecebe6")
    ax.text(-0.9, 0.95, "Réservoir à retards, S = {1, 3, 7} : une matrice par retard", fontsize=12,
            color=INK, weight="bold", zorder=5, bbox=dict(fc=SURF, ec="none", pad=1.5))
    timeline(ax, -1.4, 8, {1, 3, 7})
    ax.text(-0.9, -2.35, "x(t) = tanh( W$_{in}$ u(t) + Σ$_{τ \\in S}$ W$_τ$ x(t−τ) )", fontsize=10.5, color=INK)
    ax.text(-0.9, -2.85, "S = l'ensemble des retards ; chaque W$_τ$ est une matrice N×N tirée au hasard puis figée ; "
            "seule la lecture linéaire est apprise", fontsize=9.5, color=INK2)

    ax2 = fig.add_axes([0.69, 0.08, 0.30, 0.84])
    blank(ax2)
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    for i, (tau, col) in enumerate([(1, BLUE), (3, BLUE), (7, BLUE)]):
        x0, y0 = 1.2 + 1.0 * i, 1.6 + 1.0 * i
        ax2.add_patch(Rectangle((x0, y0), 4.6, 4.6, fc=PALE_BLUE, ec=col, lw=1.4, alpha=0.95))
        ax2.text(x0 + 4.75, y0 + 0.2, f"W$_{tau}$", fontsize=11, color=col)
    ax2.text(3.2 + 2.3, 3.6 + 2.3, "tranche\nN × N", ha="center", va="center", fontsize=9, color=INK2)
    ax2.text(5.0, 9.4, "Le tenseur W(i, j, τ)", ha="center", fontsize=12, weight="bold", color=INK)
    ax2.text(5.0, 0.6, "coût : |S| × N² connexions\nbudget fixe : N = √(400 000 / |S|)\n"
             "|S| = 1 (ESN) : N = 632\n|S| = 10 : N = 200",
             ha="center", fontsize=9.5, color=INK2, va="top")
    save(fig, "schema_modele")


# 2. structure de S
def schema_structure():
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.6), sharey=True)
    taus = np.arange(1, 11)
    panels = [
        ("tous les retards\nS = {1, …, 10}", np.ones(10), BLUE),
        ("un sous-ensemble\n(élagage) : S = {1, 10}", np.where(np.isin(taus, [1, 10]), 1.0, 0.0), ORANGE),
        ("un gain par retard\nα$_τ$ parmi 0 ; 0,5 ; 1 ; 2", np.array([2, 2, .5, 0, 1, 1, 1, 1, 1, 1.0]), AQUA),
        ("une bande continue\n[τ$_0$ ; τ$_1$] = [2,3 ; 8,5]", None, BLUE),
    ]
    for ax, (title, h, c) in zip(axs, panels):
        styled(ax)
        if h is None:
            t0, t1 = 2.3, 8.5
            h = np.clip(taus - t0 + 1, 0, 1) * np.clip(t1 - taus + 1, 0, 1)
            xx = np.linspace(0.5, 10.5, 400)
            ax.plot(xx, np.clip(xx - t0 + 1, 0, 1) * np.clip(t1 - xx + 1, 0, 1), color=INK2, lw=1, ls=DASH)
        ax.bar(taus, h, color=c, width=0.7)
        ax.set_xticks(taus)
        ax.set_title(title, fontsize=10.5, color=INK, loc="left")
        ax.set_xlabel("retard τ", fontsize=9.5, color=INK2)
    axs[0].set_ylabel("poids de la tranche W$_τ$", fontsize=9.5, color=INK2)
    save(fig, "schema_structure")


# 3. retards atteignables
def schema_atteignables():
    S = [3, 7]
    lmax = 24
    r = reachable(S, lmax)
    h = hops(S, lmax)
    fig = plt.figure(figsize=(14.5, 5.2))
    ax = fig.add_axes([0.03, 0.42, 0.94, 0.5])
    blank(ax)
    ax.set_xlim(-0.8, lmax + 0.8)
    ax.set_ylim(-1.3, 3.2)
    ax.axhline(0, color=GRID, lw=1.2, zorder=0)
    for L in range(lmax + 1):
        if L == 0:
            ax.add_patch(Circle((L, 0), 0.28, fc=INK2, ec=INK2))
            ax.text(L, -0.75, "0", ha="center", fontsize=9, color=INK2)
            continue
        if r[L]:
            ax.add_patch(Circle((L, 0), 0.28, fc=BLUE, ec=BLUE))
            ax.text(L, 0, str(int(h[L])), ha="center", va="center", fontsize=8, color="white", weight="bold")
        else:
            ax.add_patch(Circle((L, 0), 0.28, fc=SURF, ec=ORANGE, lw=1.6))
        ax.text(L, -0.75, str(L), ha="center", fontsize=9, color=INK2)
    # deux chemins illustratifs
    for path, col, yb in (([3, 7], AQUA, 0.4), ([3, 3, 7], BLUE, 0.4)):
        pos = 0
        for i, s in enumerate(path):
            arrow(ax, (pos, 0.3), (pos + s, 0.3), color=col, rad=-0.55 if col == AQUA else -0.38, lw=1.4)
            pos += s
    ax.text(5, 2.35, "10 = 3 + 7 (2 sauts)", fontsize=10, color=AQUA, ha="center")
    ax.text(9.8, 1.55, "13 = 3 + 3 + 7 (3 sauts)", fontsize=10, color=BLUE, ha="left")
    ax.text(-0.6, 2.95, "S = {3, 7} : quels âges L de l'entrée l'état x(t) peut-il contenir ?",
            fontsize=12, weight="bold", color=INK)
    ax.text(lmax + 0.6, -1.25, "âge L de l'entrée u(t−L)", ha="right", fontsize=9.5, color=INK2)

    ax2 = fig.add_axes([0.03, 0.0, 0.94, 0.34])
    blank(ax2)
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 3)
    ax2.add_patch(Circle((0.25, 2.3), 0.12, fc=BLUE, ec=BLUE))
    ax2.text(0.5, 2.3, "atteignable : L est une somme d'éléments de S (le chiffre = nombre minimal de sauts)",
             va="center", fontsize=10, color=INK)
    ax2.add_patch(Circle((0.25, 1.55), 0.12, fc=SURF, ec=ORANGE, lw=1.6))
    ax2.text(0.5, 1.55, "trou : aucune somme ne vaut L, donc x(t) ne contient aucune information sur u(t−L), "
             "quelles que soient les matrices", va="center", fontsize=10, color=INK)
    ax2.text(0.15, 0.6, "Ici les trous sont 1, 2, 4, 5, 8 et 11 ; à partir de 12, tout est atteignable. "
             "Si 1 est dans S, tout âge est atteignable et seul le nombre de sauts change.",
             fontsize=10, color=INK2)
    save(fig, "schema_atteignables")


# 4. conception par le calcul
def schema_conception():
    spec = json.load(open(ROOT / "results" / "results_p8_spectre.json"))["agg"]["narma30"]["I_single_mean"]
    spec = np.clip(np.array(spec), 0, None)
    fig = plt.figure(figsize=(15, 4.4))
    a1 = fig.add_axes([0.04, 0.18, 0.27, 0.66])
    styled(a1)
    a1.bar(np.arange(len(spec)), spec, color=BLUE, width=0.8)
    a1.set_xlabel("retard L", fontsize=9.5, color=INK2)
    a1.set_ylabel("perte si L est retiré", fontsize=9.5, color=INK2)
    a1.set_title("1. Spectre de retards de la tâche\n(oracle sur les entrées retardées, NARMA-30)",
                 fontsize=10.5, loc="left", color=INK)
    for L in (1, 30):
        a1.text(L, spec[L] + 0.02, str(L), ha="center", fontsize=9, color=INK)

    a2 = fig.add_axes([0.41, 0.18, 0.25, 0.66])
    styled(a2)
    ds = np.arange(2, 41)
    cost = np.array([hops([1, d], 31)[30] for d in ds])
    a2.bar(ds, cost, color=[ORANGE if d == 30 else AQUA for d in ds], width=0.8)
    a2.set_yscale("log")
    a2.set_xlabel("second retard d de S = {1, d}", fontsize=9.5, color=INK2)
    a2.set_ylabel("sauts pour atteindre L = 30", fontsize=9.5, color=INK2)
    a2.set_title("2. Choisir S qui atteint les retards utiles\nen un minimum de sauts (calcul exact)",
                 fontsize=10.5, loc="left", color=INK)

    a3 = fig.add_axes([0.72, 0.12, 0.27, 0.76])
    blank(a3)
    a3.set_xlim(0, 10)
    a3.set_ylim(0, 10)
    box(a3, 0.6, 6.2, 8.8, 2.4, "S = {1, 30}\nobtenu sans entraîner de réservoir", fc=PALE_ORANGE, ec=ORANGE)
    box(a3, 0.6, 1.4, 8.8, 3.2, "3. Réservoir construit avec ce S\nNARMA-30 : 0,006\n"
        "(élagage glouton, 54 réservoirs : 0,056)", fc=PALE_BLUE, ec=BLUE)
    arrow(a3, (5, 6.15), (5, 4.7), color=INK2)
    for x0 in (0.325, 0.67):
        fig.add_artist(FancyArrowPatch((x0, 0.52), (x0 + 0.04, 0.52), transform=fig.transFigure,
                                       arrowstyle="-|>", mutation_scale=16, color=INK2, lw=1.8))
    save(fig, "schema_conception")


# 5. contrôles
def schema_controles():
    rng = np.random.default_rng(3)
    N = 12
    mats = [
        ("dense (N² poids)", rng.normal(0, 1, (N, N))),
        ("permutation (gratuite)", np.eye(N)[rng.permutation(N)]),
        ("diagonale ±1 (gratuite)", np.diag(rng.choice((-1.0, 1.0), N))),
        ("identité (gratuite)", np.eye(N)),
    ]
    fig = plt.figure(figsize=(15, 5.6))
    for i, (name, M) in enumerate(mats):
        ax = fig.add_axes([0.03 + 0.125 * i, 0.5, 0.11, 0.38])
        ax.imshow(M, cmap="RdBu_r", vmin=-2, vmax=2, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(GRID)
        ax.set_title(name, fontsize=9.5, color=INK)
    fig.text(0.03, 0.94, "Quelle matrice dans les tranches τ ≥ 2 ?", fontsize=12, weight="bold", color=INK)
    fig.text(0.03, 0.42, "W$_1$ reste dense ; seules les tranches de retard changent.\n"
             "Budget honnête : les tranches gratuites libèrent les N² connexions, W$_1$ grandit (N = 632).",
             fontsize=9.5, color=INK2)

    ax = fig.add_axes([0.55, 0.08, 0.44, 0.84])
    blank(ax)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.text(0.1, 9.55, "Les autres façons d'obtenir de la mémoire", fontsize=12, weight="bold", color=INK)
    rows = [
        ("retard dans l'état", "x(t) lit x(t−30)", "tenseur", PALE_BLUE, BLUE),
        ("retard dans l'entrée", "x(t) lit u(t) et u(t−30)", "ESN à entrée retardée", PALE_AQUA, AQUA),
        ("fuite", "x(t) = (1−a) x(t−1) + a tanh(…)", "ESN à fuite, DTS-ESN", PALE_ORANGE, ORANGE),
        ("aucun réservoir", "u(t−L) et leurs produits", "NG-RC", "#ecebe6", INK2),
    ]
    for i, (a, b, c, fc, ec) in enumerate(rows):
        y = 7.4 - 2.0 * i
        box(ax, 0.1, y, 2.9, 1.4, a, fc=fc, ec=ec, size=10, weight="bold")
        box(ax, 3.3, y, 3.9, 1.4, b, fc=SURF, ec=GRID, size=9.5)
        box(ax, 7.5, y, 2.4, 1.4, c, fc=SURF, ec=GRID, size=9.5)
    save(fig, "schema_controles")


# 6. ordre 2
def schema_ordre2():
    fig = plt.figure(figsize=(14.5, 4.4))
    ax = fig.add_axes([0.0, 0.0, 0.64, 1.0])
    blank(ax)
    ax.set_xlim(-1.0, 13.6)
    ax.set_ylim(-2.3, 3.6)
    xs = timeline(ax, 1.4, 11, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, labels_w=False, highlight={10, 11}, arrows=[])
    ax.text(xs[5], 2.15, "tranches W$_1$ … W$_{10}$ du tenseur 3D (S = {1, …, 10})", ha="center", fontsize=10, color=BLUE)
    ax.text(-0.9, 3.25, "Tenseur 3D + un seul terme produit sur la paire (10, 11)", fontsize=12,
            weight="bold", color=INK)
    box(ax, 0.6, -1.6, 4.3, 1.1, "x(t−10) ⊙ x(t−11)", fc=PALE_ORANGE, ec=ORANGE, size=11)
    arrow(ax, (xs[10], 1.0), (2.2, -0.45), color=ORANGE)
    arrow(ax, (xs[11], 1.0), (3.3, -0.45), color=ORANGE)
    arrow(ax, (4.95, -1.05), (xs[0], 1.0), color=ORANGE, rad=0.25)
    ax.text(8.3, -1.25, "W$_2$ (une matrice N×N de plus)", fontsize=10.5, color=ORANGE)
    ax2 = fig.add_axes([0.67, 0.12, 0.31, 0.76])
    blank(ax2)
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.text(0.2, 9.0, "Pourquoi un produit ?", fontsize=12, weight="bold", color=INK)
    ax2.text(0.2, 7.4, "XOR(a, b) = a + b − 2ab", fontsize=12, color=INK)
    ax2.text(0.2, 5.6, "Une lecture linéaire de l'état sait faire a + b.\nLe terme ab demande de multiplier\n"
             "deux bits anciens : c'est ce que fait\nla paire (10, 11).", fontsize=10, color=INK2, va="top")
    ax2.text(0.2, 1.4, "NARMA-30 contient 1,5·u(t−30)·u(t−1) :\nla paire utile y est (1, 30).",
             fontsize=10, color=INK2)
    save(fig, "schema_ordre2")


# 7. flèche du temps
def schema_bidir():
    fig = plt.figure(figsize=(14, 3.8))
    ax = fig.add_axes([0, 0, 1, 1])
    blank(ax)
    ax.set_xlim(-0.5, 14)
    ax.set_ylim(-2.2, 3.2)
    for k in range(-5, 6):
        x = 6.5 + 1.15 * k
        lab = "x(t)" if k == 0 else (f"x(t{k:+d})")
        fc, ec = (PALE_BLUE, BLUE) if k < 0 else ((PALE_ORANGE, ORANGE) if k > 0 else ("#ecebe6", INK2))
        ax.add_patch(FancyBboxPatch((x - 0.45, -0.35), 0.9, 0.7, boxstyle="round,pad=0.02,rounding_size=0.1",
                                    fc=fc, ec=ec, lw=1.2))
        ax.text(x, 0, lab, ha="center", va="center", fontsize=8.5, color=INK)
    for k, c in ((-1, BLUE), (-3, BLUE), (1, ORANGE), (3, ORANGE)):
        x = 6.5 + 1.15 * k
        arrow(ax, (x, 0.4 if k < 0 else -0.4), (6.5, 0.4 if k < 0 else -0.4), color=c,
              rad=-0.45 if k < 0 else -0.45, lw=1.4)
    ax.text(1.0, 1.7, "passé : retards τ > 0", fontsize=10.5, color=BLUE)
    ax.text(9.6, -1.75, "futur : retards τ < 0 (hors ligne seulement)", fontsize=10.5, color=ORANGE)
    ax.text(-0.3, 2.85, "Tenseur signé : S contient des retards positifs et négatifs", fontsize=12,
            weight="bold", color=INK)
    save(fig, "schema_bidir")


# 8. réservoir physique à retards
def schema_tdr():
    N = 12
    fig = plt.figure(figsize=(15, 5.8))
    ax = fig.add_axes([0.0, 0.04, 0.36, 0.9])
    blank(ax)
    ax.set_aspect("equal")
    ax.set_xlim(-1.9, 1.9)
    ax.set_ylim(-1.9, 1.9)
    ang = np.pi / 2 - 2 * np.pi * np.arange(N) / N
    R = 1.25
    ax.add_patch(Circle((0, 0), R, fc="none", ec=GRID, lw=6))
    for i in range(N):
        ax.add_patch(Circle((R * np.cos(ang[i]), R * np.sin(ang[i])), 0.12, fc=BLUE, ec=BLUE))
        ax.text(1.5 * np.cos(ang[i]), 1.5 * np.sin(ang[i]), f"{i}", ha="center", va="center", fontsize=8.5,
                color=INK2)
    ax.add_patch(Rectangle((-0.36, -0.24), 0.72, 0.48, fc=PALE_ORANGE, ec=ORANGE, lw=1.3))
    ax.text(0, 0, "nœud\nnon linéaire", ha="center", va="center", fontsize=8.5, color=INK)
    ax.text(0, -1.85, "un seul nœud physique, N nœuds virtuels ;\nun tour = un pas de la tâche = N nœuds",
            ha="center", va="top", fontsize=9.5, color=INK2)

    ax2 = fig.add_axes([0.39, 0.06, 0.6, 0.86])
    blank(ax2)
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.text(0.1, 9.55, "Une boucle de longueur D = m·N + s relie le nœud i au nœud i − s, m tours plus tôt",
             fontsize=11.5, weight="bold", color=INK)
    for row, (s, title, c, fc, nom) in enumerate([
            (0, "D = m·N (s = 0) : chaque nœud se relit lui-même", ORANGE, PALE_ORANGE, "tranche identité :\nles traces s'empilent"),
            (1, "D = m·N + 1 (s = 1) : chaque nœud lit son voisin", BLUE, PALE_BLUE, "tranche permutation :\nles traces se séparent")]):
        y = 6.5 - 3.9 * row
        ax2.text(0.1, y + 1.65, title, fontsize=10.5, color=c, weight="bold")
        for i in range(8):
            for yy in (y + 0.6, y - 0.6):
                ax2.add_patch(FancyBboxPatch((0.35 + 0.8 * i, yy), 0.55, 0.5, boxstyle="round,pad=0.01,rounding_size=0.06",
                                             fc=fc, ec=c, lw=1.1))
                ax2.text(0.625 + 0.8 * i, yy + 0.25, str(i), ha="center", va="center", fontsize=8.5, color=INK)
            j = i + s
            if j < 8:
                arrow(ax2, (0.625 + 0.8 * i, y - 0.08), (0.625 + 0.8 * j, y + 0.58), color=c, lw=1.2, ms=8)
        ax2.text(6.95, y + 0.85, "tour n", fontsize=9, color=INK2, va="center")
        ax2.text(6.95, y - 0.35, "tour n − m", fontsize=9, color=INK2, va="center")
        ax2.text(8.35, y + 0.25, nom, fontsize=9.5, color=c, va="center")
    ax2.text(0.1, 0.25, "Une boucle à s = 0 se comporte comme la tranche identité de la partie 4 (mémoire effondrée) ;\n"
             "avec s ≠ 0, comme une permutation. Les âges atteignables se comptent en tours, comme en partie 2.",
             fontsize=9.5, color=INK2, va="bottom")
    save(fig, "schema_reservoir_physique")


def schema_familles():
    """Où chaque famille de la partie 4 met son retard."""
    fig = plt.figure(figsize=(15, 6.4))
    ax = fig.add_axes([0, 0, 1, 1])
    blank(ax)
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 13)
    ax.text(0.4, 12.35, "Où mettre un retard : les familles comparées au même budget et au même réglage",
            fontsize=12.5, weight="bold", color=INK)
    cartes = [
        ("recette (ce travail)", "x(t) lit x(t − τ), τ ∈ S calculé,\nune matrice aléatoire par retard\n(+ un terme produit si le spectre le dit)",
         "S calculé sur les données, sans réservoir", PALE_BLUE, BLUE),
        ("Iacob et Dambre (2024)", "chaque connexion j → i a son délai,\nproportionnel à la distance ‖p_i − p_j‖\nentre neurones placés dans le plan",
         "positions au hasard, ou optimisées par CMA-ES", PALE_ORANGE, ORANGE),
        ("A, B, C : les mélanges", "A : leur matrice, délais tirés dans S\nB : leurs délais près, sauts de S loin\nC : leurs délais, portée = max S",
         "chaque mélange teste une moitié de l'idée", "#ece4f7", "#9b6bd6"),
        ("lecture retardée (Holzmann et Hauser, 2010)", "ESN ordinaire ; la lecture reçoit\n[x(t), x(t − τ)] pour τ ∈ S calculé",
         "le retard est dans la lecture, pas dans l'état", PALE_AQUA, AQUA),
        ("ESN + entrées retardées", "ESN ordinaire ; l'entrée reçoit\n[u(t), u(t − τ)] pour τ ∈ S calculé",
         "le retard est dans l'entrée", PALE_AQUA, AQUA),
        ("self-ESN (Xu et al., 2024)", "x(t) lit x(t − 1) par W\net x(t − m) par γ·I",
         "un seul retard, une seule copie de l'état", "#ecebe6", INK2),
        ("cycle (Rodan et Tiňo, 2011)\net orthogonal", "W = r·(permutation circulaire)\nou W = ρ·Q, Q orthogonale",
         "un seul retard, une matrice structurée", "#ecebe6", INK2),
        ("ESN à fuite, DTS-ESN", "x(t) = (1 − a)·x(t − 1) + a·tanh(…)\nDTS-ESN : une fuite par neurone",
         "la mémoire vient de la lenteur, pas d'un retard", "#ecebe6", INK2),
    ]
    for k, (titre, corps, note, fc, ec) in enumerate(cartes):
        col, lig = k % 4, k // 4
        x, y = 0.4 + 7.4 * col, 6.4 - 5.9 * lig
        ax.add_patch(FancyBboxPatch((x, y), 7.0, 5.2, boxstyle="round,pad=0.02,rounding_size=0.15", fc=fc, ec=ec, lw=1.4))
        ax.text(x + 0.3, y + 4.75, titre, fontsize=10, weight="bold", color=INK, va="top")
        ax.text(x + 0.3, y + 3.3, corps, fontsize=9.2, color=INK, va="top", linespacing=1.5)
        ax.text(x + 0.3, y + 0.35, note, fontsize=8.6, color=ec if ec != INK2 else INK2, va="bottom", style="italic")
    save(fig, "schema_familles")


# 9. trous mous
def schema_trous_mous():
    fig, axs = plt.subplots(1, 3, figsize=(15.5, 4.3))
    rng = np.random.default_rng(1)
    T = 120
    ax = axs[0]
    styled(ax, grid=None)
    w = rng.normal(size=T + 200)
    for phi, c, lab in ((0.0, INK2, "i.i.d. (φ = 0)"), (0.9, BLUE, "lisse, AR(1) φ = 0,9")):
        u = np.zeros_like(w)
        for t in range(1, len(w)):
            u[t] = phi * u[t - 1] + w[t]
        u = u[200:] / u[200:].std()
        ax.plot(np.arange(T), u + (3.2 if phi == 0 else 0), color=c, lw=1.2, label=lab)
    ax.set_yticks([])
    ax.set_xlabel("temps", fontsize=9.5, color=INK2)
    ax.set_title("Entrée i.i.d. ou lisse", fontsize=10.5, loc="left", color=INK)
    ax.legend(fontsize=8.5, frameon=False, loc="upper right")

    ax = axs[1]
    styled(ax)
    L = np.arange(0, 21)
    for phi, c in ((0.0, INK2), (0.5, AQUA), (0.9, BLUE)):
        # variance conditionnelle de u(t-1) connaissant u(t) et u(t-2) pour un AR(1) gaussien
        cond = (1 - phi ** 2) / (1 + phi ** 2)
        ax.bar(phi * 10, cond, width=3.5, color=c)
        ax.text(phi * 10, cond + 0.03, f"φ = {phi:g}", ha="center", fontsize=9, color=INK)
    ax.set_xticks([])
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("part de u(t−1) introuvable", fontsize=9.5, color=INK2)
    ax.set_title("Trou en L = 1 avec S = {2, …} :\nplus l'entrée est lisse, moins il coûte",
                 fontsize=10.5, loc="left", color=INK)

    ax = axs[2]
    styled(ax)
    Lm = 40
    Ls = np.arange(1, Lm + 1)
    kernel = Ls * np.exp(-Ls / 6.0)
    ax.fill_between(Ls, kernel / kernel.sum(), color=PALE_ORANGE, label="enveloppe utile de la tâche")
    for S, c, lab in (([1], INK2, "ESN, S = {1}"), ([1, 4, 7], BLUE, "S = {1, 4, 7}"),
                      ([1, 12], AQUA, "S = {1, 12}")):
        e = renewal(S, [0.81 / len(S)] * len(S), Lm)[1:]
        ax.plot(Ls, e / e.sum(), color=c, lw=2, label=lab)
    ax.set_xlabel("âge L", fontsize=9.5, color=INK2)
    ax.set_ylabel("part de la mémoire à l'âge L", fontsize=9.5, color=INK2)
    ax.set_title("La masse de mémoire est fixe : il s'agit\nde la placer là où la tâche en a besoin",
                 fontsize=10.5, loc="left", color=INK)
    ax.legend(fontsize=8.3, frameon=False, loc="upper right")
    fig.tight_layout()
    save(fig, "schema_trous_mous")


# recette
def schema_recette():
    fig = plt.figure(figsize=(15, 3.4))
    ax = fig.add_axes([0, 0, 1, 1])
    blank(ax)
    ax.set_xlim(0, 30)
    ax.set_ylim(0.2, 8.0)
    steps = [
        ("1. spectre de retards\nridge sur u(t−L), L = 0..lmax\nimportance = perte si L est retiré", PALE_AQUA, AQUA),
        ("2. S par les sauts\nmin Σ$_L$ w$_L$ · h$_S$(L)\nK = 2, 3, 5, plus {1, …, 10}", PALE_ORANGE, ORANGE),
        ("3. paire d'ordre 2\nW$_2$ (x(t−a) · x(t−b)), terme à terme\nsi le spectre la désigne", PALE_ORANGE, ORANGE),
        ("4. réglage sur la validation\nρ, fuite, échelle d'entrée, λ\nmêmes tirages que les autres", PALE_BLUE, BLUE),
    ]
    for i, (t, fc, ec) in enumerate(steps):
        x = 0.4 + 7.4 * i
        box(ax, x, 3.0, 6.2, 3.0, t, fc=fc, ec=ec, size=10)
        if i < len(steps) - 1:
            arrow(ax, (x + 6.25, 4.5), (x + 7.35, 4.5), color=INK2, lw=1.8)
    ax.text(0.4, 7.3, "La recette : calculer S à partir des données, puis le régler comme n'importe quel réservoir",
            fontsize=12, weight="bold", color=INK)
    ax.text(0.4, 1.6, "h$_S$(L) = nombre minimal de sauts de longueurs prises dans S pour écrire L (rendu de monnaie) ; "
            "w$_L$ = importance normalisée de l'âge L.", fontsize=10, color=INK2)
    ax.text(0.4, 0.7, "Aucun réservoir n'est simulé aux étapes 1 à 3. Tranches τ ≥ 2 permutées (gratuites), "
            "W$_1$ dense reçoit tout le budget.", fontsize=10, color=INK2)
    save(fig, "schema_recette")


# séries réelles : le protocole
def schema_reel():
    fig = plt.figure(figsize=(15, 5.2))
    ax = fig.add_axes([0, 0, 1, 1])
    blank(ax)
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 11)
    ax.text(0.4, 10.3, "Séries neuves : la prévision est écrite et poussée avant tout calcul de réservoir",
            fontsize=12, weight="bold", color=INK)
    y0 = 7.6
    ax.add_patch(Rectangle((0.4, y0), 18.0, 1.0, fc=PALE_AQUA, ec=AQUA, lw=1.2))
    ax.add_patch(Rectangle((18.4, y0), 4.6, 1.0, fc=PALE_BLUE, ec=BLUE, lw=1.2))
    ax.add_patch(Rectangle((23.0, y0), 6.4, 1.0, fc=PALE_ORANGE, ec=ORANGE, lw=1.2))
    ax.text(9.4, y0 + 0.5, "ajustement (80 % de l'entraînement)", ha="center", va="center", fontsize=10, color=INK)
    ax.text(20.7, y0 + 0.5, "validation (20 %)", ha="center", va="center", fontsize=10, color=INK)
    ax.text(26.2, y0 + 0.5, "test : fin de la série", ha="center", va="center", fontsize=10, color=INK)
    ax.text(11.7, y0 + 1.3, "entraînement : sert au diagnostic, au réglage et à la lecture",
            ha="center", fontsize=9, color=INK2)
    ax.text(26.2, y0 + 1.3, "ouvert une seule fois, à la fin", ha="center", fontsize=9, color=ORANGE)
    steps = [
        ("1. diagnostic sans réservoir\nsur l'entraînement seul :\nP*, rapport, S calculé", PALE_AQUA, AQUA),
        ("2. prévision par une règle fixe\nrapport ≤ 0,95 : « la recette gagne »\nsinon : « pas de victoire »", "#ecebe6", INK2),
        ("3. commit + push\nde la prévision\n(date et empreinte)", "#ecebe6", INK2),
        ("4. réservoirs et références\nréglés sur la validation,\n10 seeds", PALE_BLUE, BLUE),
        ("5. test et verdict\nWilcoxon p < 0,05 :\nprévision tenue ou non", PALE_ORANGE, ORANGE),
    ]
    for i, (t, fc, ec) in enumerate(steps):
        x = 0.4 + 5.9 * i
        box(ax, x, 2.2, 5.1, 3.4, t, fc=fc, ec=ec, size=9.5)
        if i < len(steps) - 1:
            arrow(ax, (x + 5.15, 3.9), (x + 5.85, 3.9), color=INK2, lw=1.6)
    ax.text(0.4, 1.0, "P* = la période p dont les multiples {p, 2p, 3p, …} (jusqu'à 200) ajoutés à {0, 1, 2} font le plus "
            "baisser l'erreur de validation d'un ridge ; rapport = val(P*) / val({0, 1, 2}).", fontsize=9.5, color=INK2)
    save(fig, "schema_reel")


if __name__ == "__main__":
    schema_modele()
    schema_structure()
    schema_atteignables()
    schema_conception()
    schema_controles()
    schema_ordre2()
    schema_bidir()
    schema_tdr()
    schema_familles()
    schema_trous_mous()
    schema_recette()
    schema_reel()
