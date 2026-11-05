"""Figures et chiffres de la comparaison à Iacob et Dambre (2024) et aux trois mélanges.

    python src/plot_iacob.py          # écrit results/iacob_*.svg et imprime le tableau du duel
Relit seulement results/results_iacob.json et les fichiers de référence ; aucune simulation.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iacob_dambre as I        # noqa: E402
import reservoir as R           # noqa: E402
from schemas import AQUA, BLUE, GRID, INK, INK2, ORANGE, PALE_BLUE, SURF, styled  # noqa: E402

RES = R.RES
VIOLET, GRIS = "#9b6bd6", "#8d8c87"
NOMS = {"narma30": "NARMA-30", "silverbox": "Silverbox", "wh": "Wiener-Hammerstein", "mnist": "MNIST lignes",
        "mc": "capacité mémoire", "narma10": "NARMA-10", "bande|0.5|0": "bande", "ilots|0.5|0": "îlots",
        "electricite_h24": "électricité", "trafic_h24": "trafic", "energie_charge_h24": "charge (Espagne)",
        "energie_prix_h24": "prix (Espagne)", "solaire_h24": "solaire (Alabama)",
        "meteo_T_h24": "température (Jena)", "meteo_rayonnement_h24": "rayonnement (Jena)"}
FAMS = [("recette", "recette", BLUE, "o"), ("iacob", "Iacob-Dambre", ORANGE, "s"),
        ("iacob_S", "A : délais tirés dans S", AQUA, "D"), ("hybride", "B : mélange", VIOLET, "^"),
        ("iacob_spectre", "C : dmax = max S", GRIS, "v")]


def erreurs(t):
    """{famille: erreurs de test par seed} ; pour mc, erreur = 300 − capacité."""
    out = R.lire(I.FICHIER)["cells"]
    ss = I.seeds(t)
    ref = I.references(t)
    e = {"esn": np.array(ref["ESN à fuite"]), "recette": np.array(ref["recette"])}
    for f in I.FAMILLES:
        e[f] = np.array([out[f"{t}|{f}|s{s}"]["test"] for s in ss])
    return e


def duel():
    print("tâche | recette | Iacob | recette/Iacob | p | gagnant")
    lignes = []
    for t in I.TOUTES:
        e = erreurs(t)
        r, p = e["recette"].mean() / e["iacob"].mean(), R.wilcoxon(e["recette"], e["iacob"])
        g = "égalité" if p >= 0.05 else ("recette" if r < 1 else "Iacob")
        lignes.append((t, r, p, g))
        print(f"{t} | {e['recette'].mean():.4g} | {e['iacob'].mean():.4g} | ×{r:.3f} | {p:.3g} | {g}")
    return lignes


def fig_points(taches, nom, titre, log, xlim):
    fig, ax = plt.subplots(figsize=(7.6, 0.5 * len(taches) + 1.3))
    styled(ax, grid="x")
    ax.axvline(1.0, color=INK2, lw=1)
    for k, t in enumerate(taches):
        e = erreurs(t)
        y = len(taches) - 1 - k
        for j, (f, lab, c, mk) in enumerate(FAMS):
            ax.plot(e[f].mean() / e["esn"].mean(), y + (j - 2) * 0.13, mk, color=c, ms=6.5,
                    mec="white", mew=0.6, label=lab if k == 0 else None)
    if log:
        ax.set_xscale("log")
    ax.set_xlim(*xlim)
    ax.set_ylim(-0.6, len(taches) - 0.4)
    ax.set_yticks(range(len(taches)))
    ax.set_yticklabels([NOMS[t] for t in taches][::-1], fontsize=9, color=INK)
    ax.set_xlabel("erreur de test / erreur de l'ESN à fuite (moyenne sur les seeds ; sous 1 = mieux)",
                  color=INK2, fontsize=9)
    ax.legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.set_title(titre, fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(RES / nom, facecolor=SURF, bbox_inches="tight")
    plt.close(fig)
    print("écrit", nom)


def fig_duel(lignes):
    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    styled(ax, grid="x")
    ax.axvline(1.0, color=INK2, lw=1)
    n = len(lignes)
    for k, (t, r, p, g) in enumerate(lignes):
        y = n - 1 - k
        c = {"recette": BLUE, "Iacob": ORANGE, "égalité": GRID}[g]
        ax.barh(y, np.log10(r), left=0, height=0.62, color=c)
        txt = f"×{r:.2f}" if r >= 0.1 else f"×{r:.3f}"
        ax.text(np.log10(r) + (0.04 if r >= 1 else -0.04), y, txt, va="center",
                ha="left" if r >= 1 else "right", fontsize=8, color=INK2)
    ax.axvline(0, color=INK2, lw=1)
    for yb in (n - 5.5, n - 8.5):
        ax.axhline(yb, color=GRID, lw=0.8, ls=(0, (4, 3)))
    ax.set_yticks(range(n))
    ax.set_yticklabels([NOMS[l[0]] for l in lignes][::-1], fontsize=9, color=INK)
    ticks = [0.01, 0.03, 0.1, 0.3, 1, 3]
    ax.set_xticks(np.log10(ticks))
    ax.set_xticklabels([f"×{v:g}" for v in ticks])
    ax.set_xlim(np.log10(0.006), np.log10(4))
    ax.set_xlabel("erreur de la recette / erreur d'Iacob-Dambre (échelle log ; à gauche, la recette est meilleure)",
                  color=INK2, fontsize=9)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=BLUE, label="recette devant (p < 0,05)"),
                       Patch(color=ORANGE, label="Iacob-Dambre devant (p < 0,05)"),
                       Patch(color=GRID, label="pas de différence significative")],
              fontsize=8, frameon=False, loc="lower left")
    ax.text(np.log10(0.0065), n - 0.45, "développement", fontsize=8, color=INK2, va="bottom")
    ax.text(np.log10(0.0065), n - 5.45, "noyaux plantés (5 seeds)", fontsize=8, color=INK2, va="bottom")
    ax.text(np.log10(0.0065), n - 8.45, "séries neuves", fontsize=8, color=INK2, va="bottom")
    ax.set_title("La recette contre Iacob-Dambre, tâche par tâche", fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(RES / "iacob_duel.svg", facecolor=SURF)
    plt.close(fig)
    print("écrit iacob_duel.svg")


def fig_delais():
    """Part de Σ W² (énergie récurrente) par retard, pour un réservoir typique de chaque famille sur NARMA-30.

    Pour une matrice gaussienne ramenée au rayon sc, Σ W² ≈ n·sc², la même chose qu'une tranche
    permutée de valeur sc : les parts se comparent directement. Valeurs typiques retenues sur NARMA-30
    (médianes des réglages) : S = {1, 2, 30}, paire (1, 30), dmax = 38 (Iacob), 3 (B), 30 (C).
    """
    n, L = 632, 40
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, (n, 2))
    dist = np.sqrt(((p[:, None, :] - p[None, :, :]) ** 2).sum(-1)) / np.sqrt(2)

    def geo(dmax):
        D = 1 + np.rint((dmax - 1) * dist).astype(int)
        return np.bincount(D.ravel(), minlength=L + 2)[1:L + 1] / D.size

    S = [1, 2, 30]
    rec = np.zeros(L)
    for t in S:
        rec[t - 1] = 1 / 4                  # trois tranches + la paire : K = 4
    A = np.zeros(L)
    for t in S:
        A[t - 1] = 1 / 3
    B = np.zeros(L)
    B[:3] = geo(3)[:3] / 3                  # bloc dense 1..3, tranche permutée 30, paire : K = 3
    B[29] = 1 / 3
    lignes = [("recette : S = {1, 2, 30} + paire", rec, BLUE, "+ paire x(t−1)⊙x(t−30)"),
              ("Iacob-Dambre : délais de propagation, dmax = 38", geo(38), ORANGE, ""),
              ("A : délais tirés dans S", A, AQUA, ""),
              ("B : propagation 1..3 + saut 30 + paire", B, VIOLET, "+ paire x(t−1)⊙x(t−30)"),
              ("C : propagation, dmax = max S = 30", geo(30), GRIS, "")]
    fig, axs = plt.subplots(len(lignes), 1, figsize=(7.6, 6.4), sharex=True)
    for ax, (lab, h, c, note) in zip(axs, lignes):
        styled(ax, grid="y")
        ax.axvspan(29.5, 30.5, color=PALE_BLUE, zorder=0)
        ax.bar(np.arange(1, L + 1), h, width=0.8, color=c)
        ax.set_ylim(0, max(0.36, h.max() * 1.15))
        ax.set_yticks([0, 0.1, 0.2, 0.3])
        ax.set_yticklabels(["0", "", "", "30 %"], fontsize=8)
        ax.text(0.15, 0.92, lab, transform=ax.transAxes, fontsize=9, color=INK, va="top")
        if note:
            ax.text(0.99, 0.92, note, transform=ax.transAxes, fontsize=8, color=INK2, va="top", ha="right")
        v30 = h[29]
        ax.text(30, max(v30, 0.0) + 0.02, f"{100 * v30:.1f} %" if v30 >= 0.001 else "< 0,1 %",
                ha="center", fontsize=8, color=INK2)
    axs[-1].set_xlabel("retard de la connexion (pas)", color=INK2, fontsize=9)
    axs[-1].set_xticks([1, 5, 10, 15, 20, 25, 30, 35, 40])
    axs[0].set_title("NARMA-30 : part de la récurrence (Σ W²) portée par chaque retard, un réservoir typique par famille",
                     fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(RES / "iacob_delais.svg", facecolor=SURF)
    plt.close(fig)
    print("écrit iacob_delais.svg")


if __name__ == "__main__":
    lignes = duel()
    fig_delais()
    fig_points(I.DEV + I.SYNTH, "iacob_dev.svg",
               "Tâches de développement et noyaux plantés : toutes les familles réglées pareil",
               log=True, xlim=(0.007, 3))
    fig_points(I.REEL_V2 + I.REEL_V3, "iacob_series.svg",
               "Sept séries neuves : toutes les familles réglées pareil", log=False, xlim=(0.86, 1.1))
    fig_duel(lignes)
