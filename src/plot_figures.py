"""Figures de la théorie (partie 2), du test 4 (partie 5), des références (partie 4) et du bilan.

    python src/plot_figures.py
Ne relit que les JSON de results/ ; aucune simulation.
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from schemas import AQUA, BLUE, GRID, INK, INK2, ORANGE, PALE_BLUE, SURF, styled  # noqa: E402
import plot_reel as PR  # noqa: E402

RES = Path(__file__).resolve().parents[1] / "results"
VIOLET, GRIS = "#9b6bd6", "#8d8c87"
SEEDS = range(10)
NOMS4 = {"pekin_pm25_h24": "PM2.5 (Pékin)", "pekin_T_h24": "température (Pékin)",
         "pekin_pression_h24": "pression (Pékin)", "taxi_h24": "taxis (New York)",
         "pjm_charge_h24": "charge (PJM)", "change_h24": "taux de change"}


def lire(nom):
    return json.load(open(RES / nom, encoding="utf-8-sig"))


def save(fig, nom):
    fig.savefig(RES / nom, facecolor=SURF, bbox_inches="tight")
    plt.close(fig)
    print("écrit", nom)


# partie 2 : la borne par les sauts
def fig_theorie():
    d = lire("results_theorie.json")
    fig, axs = plt.subplots(1, 3, figsize=(13.2, 4.0), gridspec_kw={"width_ratios": [1.15, 1.15, 0.9]})
    for ax, k, titre in ((axs[0], "37_typique", "a. S = {3, 7}, ρ = 0,9, sans fuite"),
                         (axs[1], "130_typique", "b. S = {1, 30} (tranche 30 permutée), fuite 0,5")):
        r = d[k]
        J, B, h = np.array(r["J"]), np.array(r["B"]), np.array(r["h"])
        L = np.arange(len(J))
        styled(ax, grid="y")
        ax.semilogy(L, B, color=INK2, lw=1.2, ls=(0, (4, 3)), label="borne a·‖w‖·c(L)")
        ok = (h >= 0) & (J > 0)
        sc = ax.scatter(L[ok], J[ok], c=h[ok], cmap="viridis_r", s=14, zorder=3, label="‖∂x(t)/∂u(t − L)‖ mesurée")
        trou = h < 0
        ax.scatter(L[trou], np.full(trou.sum(), B[B > 0].min() * 0.3), marker="x", color=ORANGE, s=22,
                   zorder=3, label="L hors de ⟨S⟩ : exactement 0")
        ax.set_xlabel("âge L de l'entrée", color=INK2, fontsize=9)
        ax.set_title(titre, fontsize=9.5, color=INK, loc="left")
        cb = fig.colorbar(sc, ax=ax, pad=0.01)
        cb.set_label("nombre minimal de sauts h(L)", fontsize=8, color=INK2)
        cb.ax.tick_params(labelsize=8)
    axs[0].set_ylabel("sensibilité (échelle log)", color=INK2, fontsize=9)
    axs[0].set_xlim(-1, 45)
    axs[1].set_xlim(-1, 95)
    axs[0].legend(fontsize=7.5, frameon=False, loc="upper left")
    ax = axs[2]
    styled(ax, grid="both")
    for k, c, lab in (("37_typique", BLUE, "{3, 7}"), ("130_typique", ORANGE, "{1, 30}"),
                      ("130_contractant", AQUA, "{1, 30}, ρ = 0,4")):
        r = d[k]
        J, h = np.array(r["J"]), np.array(r["h"])
        ok = (h > 0) & (J > 0)
        ax.scatter(h[ok], np.log10(J[ok] / J[ok].max()), s=10, color=c, alpha=0.7,
                   label=f"{lab} (rang {r['spearman_logJ_h']:+.2f})")
    ax.set_xlabel("nombre minimal de sauts h(L)", color=INK2, fontsize=9)
    ax.set_ylabel("log₁₀ sensibilité relative", color=INK2, fontsize=9)
    ax.set_title("c. La sensibilité suit les sauts", fontsize=9.5, color=INK, loc="left")
    ax.legend(fontsize=7.5, frameon=False, loc="lower left")
    fig.tight_layout()
    save(fig, "theorie_borne.svg")


# partie 5 : test 4
def fig_test4():
    pre = lire("prereg_v4.json")["taches"]
    out = lire("results_reel_v4.json")["cells"]
    g = lambda t, f: np.array([out[f"{t}|{f}|s{s}"]["test"] for s in SEEDS])
    fig, axs = plt.subplots(1, 2, figsize=(11.6, 4.0))
    regles = ((axs[0], "esn_fuite", "prevision", "retards_devant",
               "a. Règle 1 : recette contre ESN à fuite", "erreur recette / ESN à fuite"),
              (axs[1], "iacob", "prevision_iacob", "recette_devant",
               "b. Règle 2 : recette contre Iacob-Dambre", "erreur recette / Iacob-Dambre"))
    taches = list(NOMS4)
    for ax, ref, cle, gagne, titre, xl in regles:
        styled(ax, grid="x")
        ax.axvline(1.0, color=INK2, lw=1)
        for k, t in enumerate(taches):
            y = len(taches) - 1 - k
            a, b = g(t, "recette"), g(t, ref)
            m, lo, hi = PR.ic_rapport(a, b)
            p = PR.wilcoxon(a, b)
            prev = pre[t][cle] == gagne
            obs = m < 1 and p < 0.05
            col = BLUE if prev else ORANGE
            ax.errorbar(m, y, xerr=[[m - lo], [hi - m]], fmt="o", ms=7, color=col,
                        mfc=col if prev == obs else SURF, capsize=2.5, elinewidth=1.1)
            ax.text(1.06, y, f"×{m:.3f}, p = {p:.2g}", fontsize=7.5, color=INK2, va="center")
        ax.set_yticks(range(len(taches)))
        ax.set_yticklabels([NOMS4[t] for t in taches][::-1], fontsize=8.5, color=INK)
        ax.set_xlabel(xl + " (10 seeds, IC 95 %)", color=INK2, fontsize=9)
        ax.set_title(titre, fontsize=9.5, color=INK, loc="left")
        ax.set_xlim(0.4, 1.3)
    axs[1].set_yticklabels([])
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.text(0.01, 0.01, "bleu : victoire prévue ; orange : pas de victoire prévue ; plein : prévision tenue ; vide : ratée",
             fontsize=8, color=INK2)
    save(fig, "reel_test4.svg")


# partie 4 : toutes les familles
def _erreurs_toutes():
    import iacob_dambre as I
    import references2 as F2
    ia = lire("results_iacob.json")["cells"]
    rf = lire("results_references2.json")["cells"]
    lignes = []
    for t in I.TOUTES:
        ss = I.seeds(t)
        ref = I.references(t)
        e = {"ESN à fuite": np.array(ref["ESN à fuite"]), "recette": np.array(ref["recette"])}
        e["Iacob-Dambre"] = np.array([ia[f"{t}|iacob|s{s}"]["test"] for s in ss])
        for f, nom in (("iacob_S", "A"), ("hybride", "B"), ("iacob_spectre", "C")):
            e[nom] = np.array([ia[f"{t}|{f}|s{s}"]["test"] for s in ss])
        for f in F2.familles(t):
            e[F2.NOMS[f]] = np.array([rf[f"{t}|{f}|s{s}"]["test"] for s in ss])
        lignes.append((t, len(ss), e))
    o4 = lire("results_reel_v4.json")["cells"]
    for t in NOMS4:
        e = {}
        for f, nom in (("esn_fuite", "ESN à fuite"), ("recette", "recette"), ("iacob", "Iacob-Dambre"),
                       ("iacob_S", "A"), ("hybride", "B"), ("iacob_spectre", "C"), ("self_esn", "self-ESN"),
                       ("lecture_retardee", "lecture retardée"), ("cycle", "cycle (SCR)"),
                       ("orthogonal", "orthogonal"), ("iacob_cma", "Iacob-Dambre + CMA-ES")):
            e[nom] = np.array([o4[f"{t}|{f}|s{s}"]["test"] for s in SEEDS])
        lignes.append((t, 10, e))
    return lignes


NOMS_T = {"narma30": "NARMA-30", "silverbox": "Silverbox", "wh": "Wiener-Hammerstein", "mnist": "MNIST lignes",
          "mc": "capacité mémoire", "narma10": "NARMA-10 (5 seeds)", "bande|0.5|0": "bande (5 seeds)",
          "ilots|0.5|0": "îlots (5 seeds)", **PR.NOMS, **NOMS4}
COLS = ["recette", "Iacob-Dambre", "Iacob-Dambre + CMA-ES", "A", "B", "C", "lecture retardée", "self-ESN",
        "cycle (SCR)", "orthogonal"]


def fig_references():
    lignes = _erreurs_toutes()
    M = np.full((len(lignes), len(COLS)), np.nan)
    sig = np.zeros_like(M, bool)
    for i, (t, n, e) in enumerate(lignes):
        esn = e["ESN à fuite"]
        for j, c in enumerate(COLS):
            if c in e:
                M[i, j] = np.log2(e[c].mean() / esn.mean())
                sig[i, j] = PR.wilcoxon(e[c], esn) < 0.05 if n >= 10 else False
    fig, ax = plt.subplots(figsize=(9.6, 0.36 * len(lignes) + 1.6))
    cmap = LinearSegmentedColormap.from_list("d", [BLUE, "#f4f3ef", ORANGE])
    Mc = np.clip(M, -2, 2)
    im = ax.imshow(np.ma.masked_invalid(Mc), cmap=cmap, norm=TwoSlopeNorm(0, -2, 2), aspect="auto")
    for i in range(M.shape[0]):
        best = np.nanargmin(M[i])
        for j in range(M.shape[1]):
            if np.isnan(M[i, j]):
                ax.text(j, i, "·", ha="center", va="center", fontsize=8, color=GRID)
                continue
            v = 2 ** M[i, j]
            txt = f"×{v:.2f}" if v >= 0.1 else f"×{v:.3f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=6.8,
                    color=INK if abs(Mc[i, j]) < 1.3 else "white",
                    weight="bold" if j == best else "normal")
            if sig[i, j]:
                ax.plot(j + 0.38, i - 0.3, "o", ms=2.2, color=INK)
    ax.set_xticks(range(len(COLS)))
    ax.set_xticklabels(COLS, rotation=35, ha="right", fontsize=8.5, color=INK)
    ax.set_yticks(range(len(lignes)))
    ax.set_yticklabels([NOMS_T[t] for t, _, _ in lignes], fontsize=8.5, color=INK)
    for yb in (4.5, 7.5, 14.5):
        ax.axhline(yb, color=SURF, lw=3)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, pad=0.01, ticks=[-2, -1, 0, 1, 2])
    cb.ax.set_yticklabels(["×0,25", "×0,5", "×1", "×2", "×4"], fontsize=8)
    cb.set_label("erreur / erreur de l'ESN à fuite (bleu : mieux)", fontsize=8, color=INK2)
    ax.set_title("Toutes les familles, rapportées à l'ESN à fuite réglé pareil (gras : meilleure de la ligne ; "
                 "point : Wilcoxon p < 0,05 ; capacité mémoire : 300 − MC)", fontsize=9, color=INK, loc="left")
    fig.tight_layout()
    save(fig, "references_carte.svg")
    return lignes


# bilan
def fig_bilan(lignes):
    """Recette contre l'ESN à fuite et contre la meilleure autre famille, sur toutes les tâches à 10 seeds."""
    rows = []
    for t, n, e in lignes:
        if n < 10:
            continue
        rec, esn = e["recette"], e["ESN à fuite"]
        autres = {k: v for k, v in e.items() if k not in ("recette", "ESN à fuite")}
        kb = min(autres, key=lambda k: autres[k].mean())
        rows.append((t, rec.mean() / esn.mean(), PR.wilcoxon(rec, esn), rec.mean() / autres[kb].mean(),
                     PR.wilcoxon(rec, autres[kb]), kb))
    fig, ax = plt.subplots(figsize=(8.6, 0.34 * len(rows) + 1.4))
    styled(ax, grid="x")
    ax.axvline(0, color=INK2, lw=1)
    for k, (t, r1, p1, r2, p2, kb) in enumerate(rows):
        y = len(rows) - 1 - k
        ax.plot(np.log2(r1), y + 0.12, "o", ms=6.5, color=BLUE, mfc=BLUE if p1 < 0.05 else SURF,
                label="contre l'ESN à fuite" if k == 0 else None)
        x2 = min(np.log2(r2), 2.0)
        ax.plot(x2, y - 0.12, "D" if x2 < 2.0 else ">", ms=5.5, color=ORANGE, mfc=ORANGE if p2 < 0.05 else SURF,
                label="contre la meilleure autre famille" if k == 0 else None)
        if x2 >= 2.0:
            ax.text(1.9, y - 0.42, f"×{r2:.0f}", fontsize=7, color=ORANGE, ha="right")
        ax.text(2.3, y - 0.12, kb, fontsize=7.5, color=INK2, va="center")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([NOMS_T[r[0]] for r in rows][::-1], fontsize=8.5, color=INK)
    tk = [1 / 64, 1 / 16, 1 / 4, 1 / 2, 1, 2, 4]
    ax.set_xticks(np.log2(tk))
    ax.set_xticklabels(["×1/64", "×1/16", "×1/4", "×1/2", "×1", "×2", "×4"], fontsize=8)
    ax.set_xlim(np.log2(0.006), 2.2)
    for yb in (len(rows) - 5.5, len(rows) - 12.5):
        ax.axhline(yb, color=GRID, lw=0.8, ls=(0, (4, 3)))
    ax.text(np.log2(0.0065), len(rows) - 1 + 0.32, "développement", fontsize=7.5, color=INK2)
    ax.text(np.log2(0.0065), len(rows) - 6 + 0.32, "tests 2 et 3", fontsize=7.5, color=INK2)
    ax.text(np.log2(0.0065), len(rows) - 13 + 0.32, "test 4", fontsize=7.5, color=INK2)
    ax.set_xlabel("erreur de la recette / erreur de la famille comparée (échelle log ; à gauche, la recette est meilleure ; "
                  "plein : p < 0,05)", color=INK2, fontsize=8.5)
    ax.legend(fontsize=8, frameon=False, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2)
    fig.tight_layout()
    save(fig, "bilan_final.svg")
    for r in rows:
        print(r[0], f"×{r[1]:.3f} p={r[2]:.3g} | ×{r[3]:.3f} p={r[4]:.3g} ({r[5]})")


if __name__ == "__main__":
    fig_theorie()
    fig_test4()
    fig_bilan(fig_references())
