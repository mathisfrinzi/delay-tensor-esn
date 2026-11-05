"""Figures des séries réelles (partie 5) et du bilan (partie 8).

    python src/plot_reel.py            # écrit results/reel_*.svg et results/bilan_recette.svg
Le panneau de diagnostic relit ETTh1 (REEL_DATA, défaut /tmp/datasets_reel) ; sans les données,
il est sauté. Les autres panneaux ne relisent que les JSON de results/.
"""
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from schemas import AQUA, BLUE, GRID, INK, INK2, ORANGE, SURF, styled  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
SEEDS = range(10)
NOMS = {"electricite_h24": "électricité", "trafic_h24": "trafic", "energie_charge_h24": "charge (Espagne)",
        "energie_prix_h24": "prix (Espagne)", "solaire_h24": "solaire (Alabama)",
        "meteo_T_h24": "température (Jena)", "meteo_rayonnement_h24": "rayonnement (Jena)",
        "etth1_h24": "ETTh1", "etth2_h24": "ETTh2", "sunspots_h12": "taches solaires",
        "melbourne_h7": "Melbourne"}


def lire(nom):
    return json.load(open(RES / nom, encoding="utf-8-sig"))


def serie(out, t, fam):
    return np.array([out["cells"][f"{t}|{fam}|s{s}"]["test"] for s in SEEDS])


def ic_rapport(a, b, n=4000, seed=0):
    """Rapport des moyennes a/b et IC bootstrap à 95 % (seeds appariées)."""
    rng = np.random.default_rng(seed)
    i = rng.integers(0, len(a), (n, len(a)))
    r = a[i].mean(1) / b[i].mean(1)
    return a.mean() / b.mean(), np.percentile(r, 2.5), np.percentile(r, 97.5)


def wilcoxon(a, b):
    from scipy.stats import wilcoxon as w
    return float(w(a, b).pvalue)


def pre_enregistrees():
    """(tâche, rapport du diagnostic, prévision, résultats) pour les tests 2 et 3."""
    rows = []
    for pre, res in (("prereg_v2.json", "results_reel_v2.json"), ("prereg_v3.json", "results_reel_v3.json")):
        p, o = lire(pre)["taches"], lire(res)
        for t, d in p.items():
            rows.append((t, d["rapport"], d["prevision"], o))
    return rows


def fig_prevision():
    rows = pre_enregistrees()
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    styled(ax, grid="both")
    ax.axvline(0.95, color=INK2, ls=(0, (4, 3)), lw=1)
    ax.axhline(1.0, color=INK2, lw=1)
    ax.text(0.945, 0.865, "seuil 0,95", color=INK2, fontsize=8, ha="right")
    ax.text(0.43, 1.085, "à gauche du seuil : prévu « la recette gagne » (bleu) ; à droite : « pas de victoire » (orange)",
            color=INK2, fontsize=8, va="top")
    for t, x, prev, o in rows:
        rec, esn = serie(o, t, "recette"), serie(o, t, "esn_fuite")
        m, lo, hi = ic_rapport(rec, esn)
        p = wilcoxon(rec, esn)
        col = BLUE if prev == "retards_devant" else ORANGE
        tenue = (m < 1 and p < 0.05) == (prev == "retards_devant")
        ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], fmt="o" if p < 0.05 else "s", ms=7,
                    mfc=col if tenue else SURF, mec=col, ecolor=col, elinewidth=1.2, capsize=3)
        dx, dy, ha = {"meteo_T_h24": (-0.008, 0.0, "right"), "electricite_h24": (-0.008, 0.0, "right"),
                      "meteo_rayonnement_h24": (-0.008, -0.012, "right"),
                      "solaire_h24": (-0.008, 0.0, "right")}.get(t, (0.008, 0.0, "left"))
        ax.text(x + dx, m + dy, NOMS[t], fontsize=8, color=INK, ha=ha, va="center")
    ax.set_xlim(0.42, 1.0)
    ax.set_ylim(0.86, 1.095)
    ax.set_xlabel("diagnostic sans réservoir : val(P*) / val({0, 1, 2})  (plus bas = périodes plus utiles)",
                  color=INK2, fontsize=9)
    ax.set_ylabel("erreur recette / erreur ESN à fuite\n(test, moyenne sur 10 seeds, IC 95 %)", color=INK2, fontsize=9)
    ax.set_title("Prévision écrite avant le calcul contre résultat : plein = prévision tenue, vide = ratée",
                 fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(RES / "reel_prevision.svg", facecolor=SURF)
    plt.close(fig)
    print("écrit reel_prevision.svg")


def fig_familles():
    rows = pre_enregistrees()
    fams = [("dts", "DTS-ESN", INK2, "v"), ("tenseur_fuite", "retards avec fuite, S cherché", AQUA, "D"),
            ("esn_entree", "ESN + entrées retardées", "#9b6bd6", "^"), ("recette", "recette (S calculé)", BLUE, "o")]
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    styled(ax, grid="x")
    ax.axvline(1.0, color=INK2, lw=1)
    for k, (t, _, _, o) in enumerate(rows):
        y = len(rows) - 1 - k
        esn = serie(o, t, "esn_fuite").mean()
        for j, (f, lab, c, mk) in enumerate(fams):
            ax.plot(serie(o, t, f).mean() / esn, y + (j - 1.5) * 0.12, mk, color=c, ms=6,
                    label=lab if k == 0 else None)
        ax.plot(o["cells"][f"{t}|ridge_lags"]["test"] / esn, y + 0.3, "x", color=ORANGE, ms=6,
                label="ridge sur 200 retards (sans réservoir)" if k == 0 else None)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([NOMS[r[0]] for r in rows][::-1], fontsize=9, color=INK)
    ax.set_xlabel("erreur de test / erreur de l'ESN à fuite (moyenne sur 10 seeds ; sous 1 = mieux)",
                  color=INK2, fontsize=9)
    ax.set_xlim(0.84, 1.13)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.set_title("Sept séries neuves, toutes les familles réglées pareil", fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(RES / "reel_familles.svg", facecolor=SURF)
    plt.close(fig)
    print("écrit reel_familles.svg")


def fig_diagnostic():
    data = Path(os.environ.get("REEL_DATA", "/tmp/datasets_reel"))
    if not (data / "ETTh1.csv").exists():
        print("ETTh1 absent : panneau de diagnostic sauté")
        return
    os.environ["REEL_DATA"] = str(data)
    import conception as C
    import p7_common as P7
    import reel_nouveau as RN
    d = RN.charger_reel("etth1_h24")
    s, lmax = d["splits"], d["lmax"]
    u, y = d["U"][:s["tr_end"], 0].astype(np.float64), d["y"][:s["tr_end"]]
    spec = C.spectre(u, y, lmax=lmax, wash=s["wash"], produits=False)
    I = np.clip(np.array(spec["I_lag"], float), 0, None)
    T = len(u)
    v0 = s["wash"] + int(0.8 * (T - s["wash"]))

    def val(lags):
        F = np.zeros((T, len(lags)), np.float32)
        for j, k in enumerate(lags):
            F[k:, j] = u[:T - k]
        return P7.ridge_readout(F, y, s["wash"], v0, v0, T)[0]
    ens = [("{0}", [0]), ("{0, 1, 22}\n(test 1)", [0, 1, 22]), ("0 à 24", list(range(25))),
           ("{0, 1, 2}\n+ multiples de 24", [0, 1, 2] + list(range(24, 193, 24))), ("0 à 200", list(range(201)))]
    vals = [val(l) for _, l in ens]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.5, 3.6), gridspec_kw={"width_ratios": [1.4, 1]})
    styled(a)
    a.bar(range(lmax + 1), I / I.max(), color=BLUE, width=1.0)
    for k in range(24, lmax + 1, 24):
        a.axvline(k, color=ORANGE, lw=0.6, ls=(0, (2, 3)))
    a.set_yscale("log")
    a.set_ylim(1e-5, 2)
    a.axhline(0.05, color=INK2, lw=1, ls=(0, (4, 3)))
    a.text(lmax, 0.065, "seuil 5 % du test 1", ha="right", fontsize=8, color=INK2)
    a.set_xlabel("âge L (heures)", color=INK2, fontsize=9)
    a.set_ylabel("importance de l'âge L seul\n(normalisée au max)", color=INK2, fontsize=9)
    a.set_title("a. Retirer un âge à la fois : seul L = 0 compte (orange : multiples de 24)",
                fontsize=9.5, color=INK, loc="left")
    styled(b)
    cols = [INK2, INK2, INK2, BLUE, INK2]
    b.bar(range(len(ens)), vals, color=cols, width=0.65)
    for i, v in enumerate(vals):
        b.text(i, v + 0.004, f"{v:.3f}".replace(".", ","), ha="center", fontsize=8, color=INK)
    b.set_xticks(range(len(ens)))
    b.set_xticklabels([e[0] for e in ens], fontsize=7.5, color=INK)
    b.set_ylim(0.6, 0.77)
    b.set_ylabel("erreur de validation d'un ridge", color=INK2, fontsize=9)
    b.set_title("b. Les multiples de 24 ensemble font tout le gain", fontsize=9.5, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(RES / "reel_diagnostic.svg", facecolor=SURF)
    plt.close(fig)
    print("écrit reel_diagnostic.svg", [round(v, 4) for v in vals])


def fig_bilan():
    """Recette / ESN à fuite sur toutes les tâches où les deux ont été réglés pareil (10 seeds)."""
    e, r = lire("results_p12_equite.json"), lire("results_p12_recette.json")
    rows = []
    for t, nom in (("narma30", "NARMA-30"), ("silverbox", "Silverbox"), ("wh", "Wiener-Hammerstein"),
                   ("mnist", "MNIST lignes (erreur)"), ("mc", "capacité mémoire (300 − MC)")):
        a = np.array([r["cells"][f"{t}|s{s}"]["test"] for s in SEEDS])
        b = np.array([e["cells"][f"{t}|esn_fuite|s{s}"]["test"] for s in SEEDS])
        rows.append(("dev", nom, *ic_rapport(a, b), wilcoxon(a, b)))
    o1 = lire("results_reel_nouveau.json")
    for t in ("etth1_h24", "etth2_h24", "sunspots_h12", "melbourne_h7"):
        a, b = serie(o1, t, "recette"), serie(o1, t, "esn_fuite")
        rows.append(("t1", NOMS[t], *ic_rapport(a, b), wilcoxon(a, b)))
    for t, _, _, o in pre_enregistrees():
        a, b = serie(o, t, "recette"), serie(o, t, "esn_fuite")
        rows.append(("t23", NOMS[t], *ic_rapport(a, b), wilcoxon(a, b)))
    col = {"dev": INK2, "t1": ORANGE, "t23": BLUE}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.6), gridspec_kw={"width_ratios": [1, 1.25]})
    for ax, groupe, log, titre in ((a1, ("dev",), True, "a. Tâches de développement (vues pendant le projet)"),
                                   (a2, ("t1", "t23"), False, "b. Séries réelles")):
        styled(ax, grid="x")
        ax.axvline(1.0, color=INK2, lw=1)
        sub = [r for r in rows if r[0] in groupe]
        n = len(sub)
        for k, (g, nom, m, lo, hi, p) in enumerate(sub):
            y = n - 1 - k
            ax.errorbar(m, y, xerr=[[m - lo], [hi - m]], fmt="o", ms=6, color=col[g],
                        mfc=col[g] if p < 0.05 else SURF, capsize=2.5, elinewidth=1.1)
        ax.set_yticks(range(n))
        ax.set_yticklabels([r[1] for r in sub][::-1], fontsize=8.5, color=INK)
        if log:
            ax.set_xscale("log")
        ax.set_title(titre, fontsize=9.5, color=INK, loc="left")
    a2.set_xlim(0.8, 1.6)
    a2.text(1.57, 6.5, "test 1 : ancien diagnostic", color=ORANGE, fontsize=8, ha="right")
    a2.text(1.57, 5.5, "tests 2 et 3 : pré-enregistrés", color=BLUE, fontsize=8, ha="right")
    a2.axhline(6.5, color=GRID, lw=0.8)
    fig.supxlabel("erreur recette / erreur ESN à fuite, même réglage, 10 seeds, IC 95 % (plein : Wilcoxon p < 0,05 ; sous 1 : la recette gagne)",
                  color=INK2, fontsize=9)
    fig.tight_layout()
    fig.savefig(RES / "bilan_recette.svg", facecolor=SURF)
    plt.close(fig)
    print("écrit bilan_recette.svg")
    for row in rows:
        print(row[0], row[1], round(row[2], 3), round(row[3], 3), round(row[4], 3), round(row[5], 4))


if __name__ == "__main__":
    fig_prevision()
    fig_familles()
    fig_diagnostic()
    fig_bilan()
