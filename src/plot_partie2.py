"""Figures Partie 2, campagne 2 : UNE FIGURE PAR SECTION (demande M. Frinzi) :
  partie2_21_ensembles.svg (2.1) ; partie2_22_pruning.svg (2.2) ;
  partie2_23_alpha.svg (2.3) ; partie2_24_bandes.svg (2.4).
Style maison : svg.fonttype='none', FR, tenseur vert #1a9850, baseline rouge #d62728,
gris #9e9e9e, elu losange orange #ff8c00."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({"svg.fonttype": "none", "font.size": 9,
                     "axes.titlesize": 10.5, "axes.labelsize": 9.5})

BASE = Path(__file__).resolve().parent.parent / "results"
ens = json.load(open(BASE / "results_ensembles.json"))["agg"]
prune = json.load(open(BASE / "results_prune.json"))
prune_final = json.load(open(BASE / "results_prune_final.json"))["cells"]
alpha = json.load(open(BASE / "results_alpha10.json"))
band = json.load(open(BASE / "results_band_refine.json"))
band_esn = json.load(open(BASE / "results_band_esn.json"))

TASKS = ["narma30", "silverbox", "wh", "mnist"]
TN = {"narma30": "NARMA-30", "silverbox": "Silverbox", "wh": "WH", "mnist": "MNIST"}
BASELINE = {"narma30": 0.387, "silverbox": 0.350, "wh": 0.098, "mnist": 0.826}
GREY, RED, GREEN, ORANGE = "#9e9e9e", "#d62728", "#1a9850", "#ff8c00"

# 2.1 ensembles echantillonnes
fig, ax = plt.subplots(figsize=(7.4, 4.6))
reg = ["narma30", "silverbox", "wh"]
x = np.arange(4)
w = 0.26
for ci, (cfg, lab, col) in enumerate([
        ("baseline", "{1..10} N=200 (baseline)", RED),
        ("even_N283", "pairs {2,4,6,8,10} N=283", GREEN),
        ("odd_N283", "impairs {1,3,5,7,9} N=283", "#77c29a")]):
    for ti, t in enumerate(reg):
        v = BASELINE[t] if cfg == "baseline" else ens[f"{t}_{cfg}"]["mean"]
        ax.bar(ti + (ci - 1) * w, v, w, color=col,
               label=lab if ti == 0 else None)
        ax.text(ti + (ci - 1) * w, v * 1.05, f"{v:.2f}", ha="center", fontsize=8)
    v = BASELINE["mnist"] if cfg == "baseline" else ens[f"mnist_{cfg}"]["mean"]
    ax.bar(3 + (ci - 1) * w, v * 10, w * 1.6, color=col, alpha=0.35,
           hatch="//", edgecolor=col,
           label="MNIST : acc. ×10" if ci == 0 else None)
    ax.text(3 + (ci - 1) * w, v * 10 + 0.12, f"{v*100:.1f} %", ha="center", fontsize=8)
ax.set_xticks(list(range(4)))
ax.set_xticklabels([TN[t] for t in reg] + ["MNIST\n(acc. ×10, hachuré)"])
ax.set_yscale("log")
ax.set_ylabel("NRMSE (log)")
ax.set_title("2.1 : Échantillonner 1 tranche sur 2 est quasi gratuit\n"
             "pairs/impairs à budget égal (400k) vs baseline {1..10}")
ax.legend(fontsize=7.5, loc="upper left")
ax.grid(axis="y", alpha=0.3)
ax.annotate("τ=1 indispensable :\nles pairs s'effondrent", xy=(w, 0.789),
            xytext=(0.32, 1.6), fontsize=8,
            arrowprops=dict(arrowstyle="->", color=GREY))
ax.annotate("records projet", xy=(2 - w, 0.078), xytext=(1.45, 0.045),
            fontsize=8, color=GREEN,
            arrowprops=dict(arrowstyle="->", color=GREEN))
fig.tight_layout()
fig.savefig(BASE / "partie2_21_ensembles.svg")
plt.close(fig)
print("2.1 OK")

# 2.2 pruning
fig, ax = plt.subplots(figsize=(7.4, 4.6))
cols = {"narma30": RED, "silverbox": GREEN, "wh": "#54278f", "mnist": ORANGE}
for t in TASKS:
    path = prune[t]["path"]
    Ks = [p["K"] for p in path]
    vals = [p["val"] for p in path]
    if t == "mnist":
        vals = [v * 10 for v in vals]
    ax.plot(Ks, vals, "o-", color=cols[t], ms=4,
            label=f"{TN[t]} (acc. ×10)" if t == "mnist" else TN[t], lw=1.3)
    el = prune_final[t]
    kEl = len(el["taus"])
    vEl = [p["val"] for p in path if p["K"] == kEl][0]
    if t == "mnist":
        vEl *= 10
    ax.plot(kEl, vEl, "D", ms=9, color=cols[t], mec="black", mew=0.8, zorder=5)
    ax.annotate("{" + ",".join(map(str, el["taus"])) + "}", xy=(kEl, vEl),
                xytext=(kEl - 1.2, vEl * (1.3 if t != "mnist" else 1.07)),
                fontsize=8, color=cols[t])
ax.set_yscale("log")
ax.invert_xaxis()
ax.set_xlabel("K tranches restantes (élimination gloutonne 10 → 1, budget redistribué N=√(400k/K))")
ax.set_ylabel("NRMSE validation (log ; MNIST acc. ×10)")
ax.set_title("2.2 : Pruning glouton : chaque tâche survit avec 2-7 tranches\n"
             "losange = ensemble élu (évaluation finale 3 seeds × ρ)")
ax.legend(fontsize=8, loc="lower left")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(BASE / "partie2_22_pruning.svg")
plt.close(fig)
print("2.2 OK")

# 2.3 alpha_tau
fig, ax = plt.subplots(figsize=(7.4, 4.6))
taus = list(range(1, 11))
mk = {"narma30": ("o", RED), "silverbox": ("s", GREEN),
      "wh": ("^", "#54278f"), "mnist": ("D", ORANGE)}
for t in TASKS:
    a = alpha[t]["learned"]["alphas"]
    m, c = mk[t]
    ax.plot(taus, a, m + "-", color=c, ms=5, lw=1.2, label=TN[t])
    u = alpha[t]["uniform"]["test"]
    l = alpha[t]["learned"]["test"]
    ax.text(10.4, a[-1] + (0.14 if t not in ("wh", "silverbox") else -0.35),
            f"{u:.3f} → {l:.3f}", color=c, fontsize=8, va="center")
ax.axhline(1.0, color=GREY, ls="dashed", lw=0.9)
ax.text(1.0, 1.07, "α uniforme = baseline", fontsize=8, color=GREY)
ax.set_xticks(taus)
ax.set_xlabel("τ")
ax.set_ylabel(r"$\alpha_\tau$ appris")
ax.set_title(r"2.3 : Profils α_τ appris par descente de coordonnées (10 tranches, init α = 1 uniforme)"
             "\nWH tue τ=1,2 ; NARMA booste τ=1,2 : test avant → après à droite")
ax.set_xlim(0.5, 13.6)
ax.legend(fontsize=8, loc="upper right")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(BASE / "partie2_23_alpha.svg")
plt.close(fig)
print("2.3 OK")

# 2.4 bandes continues
fig, ax = plt.subplots(figsize=(7.4, 4.6))
ks = np.arange(1, 21)
def gains(t0, t1):
    return np.clip(ks - t0 + 1, 0, 1) * np.clip(t1 - ks + 1, 0, 1)
for t in TASKS:
    lb = band[t]["final_band"]
    g = gains(lb["tau0"], lb["tau1"])
    m, c = mk[t]
    ax.fill_between(ks, g, alpha=0.18, color=c)
    ax.plot(ks, g, color=c, lw=1.5,
            label=f"{TN[t]} : [{lb['tau0']:g}, {lb['tau1']:g}] → {lb['test_mean']:.3f}±{lb['test_std']:.3f}")
    ax.text(lb["tau1"], 1.05, f"τ₁={lb['tau1']:g}", ha="center", fontsize=7.5, color=c)
    ax.text(lb["tau0"], -0.1, f"τ₀={lb['tau0']:g}", ha="center", fontsize=7.5, color=c)
# variante : meme recherche continue mais depart ESN (1,1) : en pointilles
for t in TASKS:
    eb = band_esn[t]["learned_band"]
    g = gains(eb["tau0"], eb["tau1"])
    m, c = mk[t]
    ax.plot(ks, g, color=c, lw=1.1, ls="dashed", alpha=0.85,
            label="même recherche, départ ESN (1,1)" if t == "narma30" else None)
ax.set_xticks(ks)
ax.set_xlabel("couche k (Tmax = 20)")
ax.set_ylabel("gain g(k)")
ax.set_ylim(-0.2, 1.2)
ax.set_title("2.4 : Bandes continues apprises [τ₀, τ₁] : recherche continue (précision ~0,05)\n"
             "plein = départ tenseur (1,10) ; pointillé = départ ESN (1,1) : couches 1 seules")
ax.legend(fontsize=8, loc="lower right")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(BASE / "partie2_24_bandes.svg")
plt.close(fig)
print("2.4 OK")

# 2.5 profils rho_tau
rho = json.load(open(BASE / "results_rho_tau.json"))
fig, ax = plt.subplots(figsize=(7.4, 4.6))
profs = ["uniform", "croissant", "decroissant", "en-tente", "en-V"]
plab = {"uniform": "uniforme (=ρ/√K)", "croissant": "croissant ρ_τ∝k",
        "decroissant": "décroissant ρ_τ∝1/k", "en-tente": "en-tente (milieu fort)",
        "en-V": "en-V (extrêmes forts)"}
pcols = {"uniform": RED, "croissant": GREEN, "decroissant": GREY,
         "en-tente": ORANGE, "en-V": "#54278f"}
x = np.arange(4)
w = 0.15
for ci, pr in enumerate(profs):
    for ti, t in enumerate(["narma30", "silverbox", "wh"]):
        v = rho[t][pr]["test_mean"]
        ax.bar(ti + (ci - 2) * w, v, w, color=pcols[pr],
               label=plab[pr] if ti == 0 else None)
        ax.text(ti + (ci - 2) * w, v * 1.04, f"{v:.2f}", ha="center", fontsize=5.6)
    v = rho["mnist"][pr]["test_mean"]
    ax.bar(3 + (ci - 2) * w, v * 10, w * 1.1, color=pcols[pr], alpha=0.35,
           hatch="//", edgecolor=pcols[pr])
    ax.text(3 + (ci - 2) * w, v * 10 + 0.08, f"{v*100:.0f}", ha="center", fontsize=5.6)
ax.set_xticks(x)
ax.set_xticklabels(["NARMA-30", "Silverbox", "WH", "MNIST (acc.×10)"])
ax.set_yscale("log")
ax.set_ylabel("NRMSE test (log)")
ax.set_title("2.5 : Rayons individuels ρ_τ : profils imposés à budget spectral total constant (ρ²)\n"
             "décroissant détruit partout (prédiction « tranches longues faibles » falsifiée) ; aucun profil imposé ne bat le ρ_τ appris (2.3)")
ax.legend(fontsize=6.5, ncol=2, loc="upper left")
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(BASE / "partie2_25_rho.svg")
plt.close(fig)
print("2.5 OK")

# 2.6 sparsite p(tau)
ptau = json.load(open(BASE / "results_p_tau.json"))
fig, ax = plt.subplots(figsize=(7.4, 4.6))
configs = [("dense", "dense p=1 (réf.)"), ("uni02", "uniforme p=0,2 (5× moins de connexions)"),
           ("uni05", "uniforme p=0,5"), ("learned", "p(τ) appris par tranche")]
ccols = {"dense": RED, "uni02": GREEN, "uni05": "#77c29a", "learned": ORANGE}
x = np.arange(4)
w = 0.19
for ci, (cfg, lab) in enumerate(configs):
    for ti, t in enumerate(["narma30", "silverbox", "wh"]):
        v = rho[t]["uniform"]["test_mean"] if cfg == "dense" else ptau[t][cfg]["test_mean"]
        ax.bar(ti + (ci - 1.5) * w, v, w, color=ccols[cfg], label=lab if ti == 0 else None)
        ax.text(ti + (ci - 1.5) * w, v * 1.03, f"{v:.2f}", ha="center", fontsize=6)
    v = rho["mnist"]["uniform"]["test_mean"] if cfg == "dense" else ptau["mnist"][cfg]["test_mean"]
    ax.bar(3 + (ci - 1.5) * w, v * 10, w * 1.1, color=ccols[cfg], alpha=0.35,
           hatch="//", edgecolor=ccols[cfg])
    ax.text(3 + (ci - 1.5) * w, v * 10 + 0.06, f"{v*100:.0f}", ha="center", fontsize=6)
ax.set_xticks(x)
ax.set_xticklabels(["NARMA-30", "Silverbox", "WH", "MNIST (acc.×10)"])
ax.set_yscale("log")
ax.set_ylabel("NRMSE test (log)")
ax.set_title("2.6 : Sparsité par tranche : densité p(τ) apprise vs uniforme vs dense\n"
             "la sparsité est quasi gratuite (5× moins de connexions) ; le profil appris n'apporte rien de plus : réglage de second ordre")
ax.legend(fontsize=6.5, loc="upper left")
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(BASE / "partie2_26_sparsite.svg")
plt.close(fig)
print("2.6 OK")

# 2.7 CP-rang
cpr = json.load(open(BASE / "results_cprank.json"))
prf = json.load(open(BASE / "results_prune_final.json"))["cells"]
fig, ax = plt.subplots(figsize=(7.4, 4.6))
tk = {"narma30": (RED, "NARMA-30 {1,10}"), "silverbox": (GREEN, "Silverbox {2,3,4,5,7,8,10}"),
      "wh": ("#54278f", "WH {4,7}"), "mnist": (ORANGE, "MNIST {2,5,7,10}")}
for t, (c, lab) in tk.items():
    N = cpr[t]["N"]
    fr = [0.25, 0.5, 1.0]
    vals = []
    for r in sorted(int(k) for k in cpr[t]["ranks"]):
        s = cpr[t]["ranks"][str(r)]["seeds"]
        vals.append(np.mean([s[str(i)] for i in range(3)]))
    ref = prf[t]["seeds"]
    refv = np.mean(list(ref.values()))
    vals = [v * 10 for v in vals] if t == "mnist" else vals
    refv = refv * 10 if t == "mnist" else refv
    ax.plot(fr, vals, "o-", color=c, ms=5, lw=1.3,
            label=lab + (" (acc.×10)" if t == "mnist" else ""))
    ax.plot(1.18, refv, "s", ms=7, color=c, mec="black", mew=0.7)
    ax.annotate(f"gaussien plein {refv/(10 if t=='mnist' else 1):.3f}",
                xy=(1.18, refv), xytext=(0.98, refv * (0.75 if t != "mnist" else 0.82)),
                fontsize=6.5, color=c)
ax.axvline(1.0, color=GREY, ls=":", lw=0.9)
ax.text(0.26, 0.0015, "rang r=N/4\n(budget ÷ 2)", fontsize=7, color=GREY)
ax.set_xticks([0.25, 0.5, 1.0])
ax.set_xticklabels(["r = N/4", "r = N/2", "r = N"])
ax.set_yscale("log")
ax.invert_xaxis()
ax.set_xlabel("rang de la factorisation W_τ = A_τB_τᵀ (fraction de N)")
ax.set_ylabel("NRMSE test (log ; MNIST acc.×10)")
ax.set_title("2.7 : Factorisation CP par tranche sur les ensembles élus (carré = gaussien plein, réf. 2.2)\n"
             "NARMA : r=N/4 → 0,035, record projet (×11 vs baseline) à moitié budget ; MNIST : r=N/2 → 85,7 % ; WH préfère le plein rang")
ax.legend(fontsize=7, loc="lower right")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(BASE / "partie2_27_cprang.svg")
plt.close(fig)
print("2.7 OK")

# 2.8 clusters
clu = json.load(open(BASE / "results_cluster.json"))
fig, ax = plt.subplots(figsize=(7.4, 4.6))
x = np.arange(4)
w = 0.3
for ti, t in enumerate(TASKS):
    single = band[t]["final_band"]["test_mean"]
    mix = clu[t]["test_mean"]
    v1, v2 = (single, mix) if t != "mnist" else (single * 10, mix * 10)
    ax.bar(ti - w / 2, v1, w, color=GREEN, label="bande seule (2.4)" if ti == 0 else None)
    ax.bar(ti + w / 2, v2, w, color=ORANGE, label="mélange 2 clusters (2.8)" if ti == 0 else None)
    ax.text(ti - w / 2, v1 * 1.03, f"{v1/(10 if t=='mnist' else 1):.3f}", ha="center", fontsize=6.5)
    ax.text(ti + w / 2, v2 * 1.03, f"{v2/(10 if t=='mnist' else 1):.3f}", ha="center", fontsize=6.5)
ax.set_xticks(x)
ax.set_xticklabels([TN[t] + ("\n(acc.×10)" if t == "mnist" else "") for t in TASKS])
ax.set_yscale("log")
ax.set_ylabel("NRMSE test (log ; MNIST acc.×10)")
ax.set_title("2.8 : τ par cluster : mélange des 2 bassins multi-départ (N/2 chaque, budget total constant)\n"
             "verdict négatif : la bande seule gagne partout : diviser N par cluster coûte plus que la diversité n'apporte")
ax.legend(fontsize=7, loc="upper left")
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(BASE / "partie2_28_cluster.svg")
plt.close(fig)
print("2.8 OK")

# 2.9 NG-RC
ngr = json.load(open(BASE / "results_ngrc.json"))
fig, ax = plt.subplots(figsize=(7.4, 4.6))
rows = [("D10_u", "NG-RC D=10, u seul (équitable)", GREY),
        ("D30_u", "NG-RC D=30, u seul (équitable)", "#8c6bb1"),
        ("D10_u+y", "NG-RC D=10, oracle u+y", "#dfc27d"),
        ("D30_u+y", "NG-RC D=30, oracle u+y", "#bf812d")]
x = np.arange(3)
w = 0.28
for ci, (k, lab, col) in enumerate(rows):
    for ti, t in enumerate(["narma30", "silverbox", "wh"]):
        v = ngr[t][k]["test_mean"]
        ax.bar(ti + (ci - 1.5) * w, v, w, color=col, label=lab if ti == 0 else None)
        ax.text(ti + (ci - 1.5) * w, v * 1.06, f"{v:.2f}", ha="center", fontsize=5.8)
# reperes tenseur
for ti, (t, cfg, lab) in enumerate([("narma30", 0.035, "tenseur optimisé (2.7, r=N/4)"),
                                    ("silverbox", 0.336, "tenseur optimisé (2.7, r=N/4)"),
                                    ("wh", 0.057, "tenseur optimisé (2.7, r=N)")]):
    ax.plot([ti - 0.42, ti + 0.42], [cfg, cfg], color=GREEN, lw=1.8)
    ax.text(ti, cfg * 0.8, lab, fontsize=6.2, color=GREEN, ha="center")
for ti, (t, cfg) in enumerate([("narma30", 0.387), ("silverbox", 0.350), ("wh", 0.098)]):
    ax.plot([ti - 0.42, ti + 0.42], [cfg, cfg], color=RED, lw=1.4, ls="dashed")
    ax.text(ti, cfg * 1.12, "tenseur uniforme", fontsize=6.2, color=RED, ha="center")
ax.set_xticks(x)
ax.set_xticklabels(["NARMA-30", "Silverbox", "WH"])
ax.set_yscale("log")
ax.set_ylabel("NRMSE test (log)")
ax.set_title("2.9 : Baseline NG-RC (features = retards D + monômes d'ordre 2, ~10³ paramètres vs 400k connexions)\n"
             "u seul D=30 bat le tenseur uniforme sur NARMA ; l'oracle u+y est quasi parfait ; le tenseur optimisé reste devant partout")
ax.legend(fontsize=6.5, ncol=2, loc="upper left")
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(BASE / "partie2_29_ngrc.svg")
plt.close(fig)
print("2.9 OK")

for f in ["partie2_21_ensembles.svg", "partie2_22_pruning.svg",
          "partie2_23_alpha.svg", "partie2_24_bandes.svg",
          "partie2_25_rho.svg", "partie2_26_sparsite.svg", "partie2_27_cprang.svg",
          "partie2_28_cluster.svg", "partie2_29_ngrc.svg"]:
    d = (BASE / f).read_text()
    print(f, len(d), d.strip().endswith("</svg>"))
