"""Figure 3.1 : Bidirectionnalite : bi-ESN directionnel, tenseur D aux
retards negatifs (miroir temporel), variantes COUPLEes acausales
v1 (proposition M. Frinzi : x+/- mono-pas + etat fusionne x) et
v2 (multi-retards DANS la recurrence : dir 3 etats / flat tenseur signe),
a budget de connexions total 400k.
Panneau B : famille ELU (uniforme dans les tables du doc).
Panneau C : MNIST avec controle de parite de readout (miroir relu a
4/8 points, bidir_pts4.py).
Style maison : svg.fonttype='none', FR, tenseur vert #1a9850, ESN rouge
#d62728, gris #9e9e9e, record losange orange #ff8c00, hachures = couple.
Sortie : results/partie3_bidir.svg."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({"svg.fonttype": "none", "font.size": 9,
                     "axes.titlesize": 10.5, "axes.labelsize": 9.5})
BASE = Path(__file__).resolve().parent.parent / "results"
bidir = json.load(open(BASE / "results_bidir.json", encoding="utf-8-sig"))
cpl = json.load(open(BASE / "results_bidir_cpl.json", encoding="utf-8-sig"))
cpl2 = json.load(open(BASE / "results_bidir_cpl2.json", encoding="utf-8-sig"))
prune = json.load(open(BASE / "results_prune_final.json", encoding="utf-8-sig"))["cells"]
cprank = json.load(open(BASE / "results_cprank.json", encoding="utf-8-sig"))
part2 = json.load(open(BASE / "results_part2.json", encoding="utf-8-sig"))
esnref = json.load(open(BASE / "results_esn_ref.json", encoding="utf-8-sig"))

GREY, RED, GREEN, ORANGE = "#9e9e9e", "#d62728", "#1a9850", "#ff8c00"
TN = {"narma30": "NARMA-30", "silverbox": "Silverbox", "wh": "WH"}
REG = ["narma30", "silverbox", "wh"]
B = lambda cfg, t: bidir[cfg][t]["test_mean"]
C = lambda cfg, t, v: cpl[cfg][t][f"test_{v}_mean"]
C2 = lambda cfg, t, v: cpl2[cfg][t][f"test_{v}_mean"]
mean = lambda d: sum((v["test"] if isinstance(v, dict) else v) for v in d.values()) / len(d)

causal632 = {t: esnref[t]["std"]["test_mean"] for t in REG}
elu = {t: mean(prune[t]["seeds"]) for t in prune}
uni_causal = {t: part2[t]["tau_1_10"]["mean"] for t in ["narma30", "silverbox", "wh", "mnist"]}
cprank_mnist = max(mean(cprank["mnist"]["ranks"][r]["seeds"]) for r in cprank["mnist"]["ranks"])

fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(16.5, 5.2),
                                    gridspec_kw={"width_ratios": [1.0, 1.25, 1.45]})

# A : bi-ESN sur taches causales
x = np.arange(3)
w = 0.19
groups = [("causal N=632 (400k)", causal632, RED, ""),
          ("causal N=446 (200k)", {t: B("esn_half", t) for t in REG}, GREY, "//"),
          ("bi-ESN β=0,5 (446+446)", {t: B("esn_b50", t) for t in REG}, "#e8968a", ""),
          ("anti-causal β=1 (0+632)", {t: B("esn_b100", t) for t in REG}, "#7a1616", "")]
for ci, (lab, vals, col, hatch) in enumerate(groups):
    for ti, t in enumerate(REG):
        axA.bar(ti + (ci - 1.5) * w, vals[t], w * 0.92, color=col,
                hatch=hatch, edgecolor="#555555" if hatch else "none",
                linewidth=0.6, label=lab if ti == 0 else None)
        axA.text(ti + (ci - 1.5) * w, vals[t] * 1.06, f"{vals[t]:.2f}",
                 ha="center", fontsize=7.3)
axA.axhline(1.0, color="#555555", linestyle="dashed", linewidth=1)
axA.text(1.02, 1.012, "NRMSE = 1 : prédire la moyenne", fontsize=7.5, color="#555555")
axA.set_xticks(x, [TN[t] for t in REG])
axA.set_yscale("log")
axA.set_ylim(0.35, 1.45)
axA.set_ylabel("NRMSE (log)")
axA.set_title("A : bi-ESN sur tâches causales :\nle miroir ne rapporte rien, il coûte")
axA.legend(fontsize=7.3, loc="upper right")
axA.grid(axis="y", alpha=0.3)

# B : famille ELU : causal > miroir indep > couple
x = np.arange(4)
w = 0.15
TASKS4 = ["narma30", "silverbox", "wh", "mnist"]
def val_b(lab, t):
    if lab == "elu":        return elu[t]
    if lab == "miroir":     return B("tens_elected", t)
    if lab == "v1":         return C("cpl_elected", t, "xpm")
    if lab == "v2dir":      return C2("cpl2_dir_elected", t, "xpm")
    return C2("cpl2_flat_elected", t, "xpm" if t == "mnist" else "x")
labels = [
    ("élu causal (2.2)", "elu", GREEN, ""),
    ("élu ± miroir indép.", "miroir", "#5f9c7d", ""),
    ("élu ± couplé v1 [x⁺;x⁻;x]", "v1", GREEN, "///"),
    ("élu ± couplé v2 dir [x⁺;x⁻;x]", "v2dir", GREEN, "xxx"),
    ("couplé v2 flat (x seul ; MNIST : 4 pts)", "v2flat", "#a6d8b8", "..."),
]
for ci, (lab, key, col, hatch) in enumerate(labels):
    for ti, t in enumerate(TASKS4):
        v = val_b(key, t)
        h10 = t == "mnist"
        vv = v * 10 if h10 else v
        axB.bar(ti + (ci - 2) * w, vv, w * 0.92, color=col,
                alpha=0.55 if h10 else 1.0, hatch=hatch,
                edgecolor="#333333" if hatch else "none",
                linewidth=0.8, label=lab if ti == 0 else None)
        txt = f"{v*100:.1f} %" if h10 else f"{v:.3f}"
        axB.text(ti + (ci - 2) * w, vv * 1.07, txt, ha="center", va="bottom",
                 fontsize=6.3, rotation=90)
axB.axhline(1.0, color="#555555", linestyle="dashed", linewidth=1)
axB.set_xticks(x, ["NARMA-30", "Silverbox", "WH", "MNIST (acc. ×10)"])
axB.set_yscale("log")
axB.set_ylim(0.045, 15)
axB.set_title("B : famille élu, retards négatifs : causal > miroir indép. > couplé\n"
              "v2 (multi-retards dans la récurrence) répare v1 mais ne bat pas le miroir")
axB.legend(fontsize=6.4, loc="upper left", ncol=1, borderpad=0.4, labelspacing=0.4)
axB.grid(axis="y", alpha=0.3)
axB.annotate("v1 : récurrence monopas →\nperd la mémoire multi-retards",
             xy=(0 + 0 * w, 0.422), xytext=(0.52, 1.9), fontsize=7.2,
             arrowprops=dict(arrowstyle="->", color=GREY))
axB.annotate("v2 répare v1 (×2,4 NARMA)\nmais reste derrière le miroir\n(0,179 > 0,096)",
             xy=(0 + 1 * w, 0.179), xytext=(1.25, 0.052), fontsize=7.2,
             arrowprops=dict(arrowstyle="->", color=GREY))
axB.plot(3 + 2 * w, C2("cpl2_flat_elected", "mnist", "xpm") * 10, marker="D",
         color="black", markersize=7, zorder=6, markerfacecolor="none", markeredgewidth=1.2)
axB.annotate("record MNIST 89,3 %", xy=(3 + 2 * w, 9.4), xytext=(2.1, 13.2),
             fontsize=8.5, color=ORANGE, fontweight="bold",
             arrowprops=dict(arrowstyle="->", color=ORANGE))

# C : MNIST : parite de readout
cfgs = [("ESN causal (400k)", esnref["mnist"]["std"]["test_mean"], RED, ""),
        ("bi-ESN β=0,5", B("esn_b50", "mnist"), "#e8968a", ""),
        ("ESN anti-causal", B("esn_b100", "mnist"), "#7a1616", ""),
        ("uniforme causal", uni_causal["mnist"], "#77c29a", ""),
        ("élu causal (2.2)", elu["mnist"], GREEN, ""),
        ("CP-rang (2.7)", cprank_mnist, "#bbbbbb", ""),
        ("miroir indép. (446 feat.)", B("tens_elected", "mnist"), "#5f9c7d", ""),
        ("miroir indép. 4 pts (891)", bidir["tens_elected_pts4"]["mnist"]["test_mean"], "#5f9c7d", ".."),
        ("miroir indép. 8 pts (1783)", bidir["tens_elected_pts8"]["mnist"]["test_mean"], "#5f9c7d", "//"),
        ("couplé v1 (624 feat.)", C("cpl_elected", "mnist", "xpm"), GREEN, "///"),
        ("couplé v2 dir (474)", C2("cpl2_dir_elected", "mnist", "xpm"), GREEN, "xxx"),
        ("couplé v2 flat (892)", C2("cpl2_flat_elected", "mnist", "xpm"), "#a6d8b8", "...")]
xp = np.arange(len(cfgs))
for i, (lab, v, col, hatch) in enumerate(cfgs):
    axC.bar(i, v * 100, 0.72, color=col, hatch=hatch or None,
            edgecolor="#333333" if (hatch or col == "#bbbbbb") else "none",
            linewidth=1.0)
    axC.text(i, v * 100 + 0.8, f"{v*100:.1f}", ha="center", fontsize=7.2)
axC.plot(11, C2("cpl2_flat_elected", "mnist", "xpm") * 100, marker="D", color="black",
         markersize=7, zorder=6, markerfacecolor="none", markeredgewidth=1.2)
axC.set_xticks(xp, [c[0] for c in cfgs], rotation=42, ha="right", fontsize=7.0)
axC.axhline(esnref["mnist"]["std"]["test_mean"] * 100, color=RED, linestyle="dashed",
            linewidth=1, alpha=0.7)
axC.text(0.1, esnref["mnist"]["std"]["test_mean"] * 100 + 0.6, "ESN causal",
         fontsize=7, color=RED)
axC.set_ylabel("accuracy test (%)")
axC.set_ylim(58, 94)
axC.set_title("C : MNIST, sans flèche du temps : à parité de readout le couplé v2 prend la tête\n"
              "(89,3 % vs 86,9 % à ~892 feat.) ; le miroir ne reprend l'avantage qu'à 2× features (90,0 %)")
axC.grid(axis="y", alpha=0.3)
axC.annotate("record à parité\nde readout : 89,3 %", xy=(11, 90.1), xytext=(8.7, 92.6),
             fontsize=8, color=ORANGE, fontweight="bold",
             arrowprops=dict(arrowstyle="->", color=ORANGE))
axC.annotate("2× features", xy=(8, 90.6), xytext=(6.1, 92.7), fontsize=7.2, color=GREY,
             arrowprops=dict(arrowstyle="->", color=GREY))

fig.suptitle("2.10 : Bidirectionnalité : le miroir temporel ne paie que là où la tâche n'a pas de flèche du temps "
             "(budget total 400k constant, protocole std, 3 seeds)", fontsize=11, y=0.99)
fig.text(0.005, 0.010,
         "Direction arrière = réservoir miroir sur la séquence inversée, washout sur le futur au-delà de la fenêtre (200 pts NARMA / 500 pts phys) ; MNIST : lignes en ordre inverse. "
         "β = part du budget récurrent allouée à la direction arrière.\n"
         "Couplé v1 (proposition de M. Frinzi) : x± mono-pas, état fusionné x par tranches signées ; hachures larges ///. "
         "Couplé v2 : les retards multiples DANS la récurrence : dir : x⁺(t)=tanh(Wᵢₙu(t)+Σ W⁺_τ x(t−τ)), x⁻ miroir, x(t)=tanh(Wᵢₙu+Σ W_a,τ x⁺(t−τ)+W_b,τ x⁻(t+τ)) (hachures xxx) ;\n"
         "flat : tenseur D signé nu x(t)=tanh(Wᵢₙu(t)+Σ W₋_τ x(t−τ)+W₊_τ x(t+τ)) (hachures fines ...) ; point fixe global par balayages Jacobi relaxés (damp 0,7, warm start tenseur indépendant).\n"
         "MNIST : readout du flat = 4 points de lecture [x(0);x(9);x(18);x(27)] (892 features) ; contrôle de parité : miroir indépendant relu à 4 points (891 feat., bidir_pts4.py) et 8 points (1783 feat.).\n"
         "Couplés : meilleur ρ par seed sur le readout concaténé ; point fixe approché (δ ≤ 1e-4 à ρ=0,5 ; δ ~ 1e-2 à ρ=0,9-0,99, verdicts inchangés).",
         fontsize=7.3, color="#555555")
fig.subplots_adjust(left=0.05, right=0.99, top=0.78, bottom=0.32, wspace=0.24)
fig.savefig(BASE / "partie3_bidir.svg", bbox_inches="tight")
print("ok")
