"""Les 400 000 poids servent-ils vraiment ? (rapport/4_comparaisons.md)

Tranches structurees contre tranches denses gaussiennes, tau = 1 TOUJOURS
dense (coeur recurrent), seules les tranches tau >= 2 changent :
  (a) dense gaussienne (reference) ; (b) rho_tau * P (permutation) ;
  (c) rho_tau * I (identite) ; (d) rho_tau * D (diagonale ±1).
Rayon spectral exact rho_tau = rho/sqrt(K) pour toutes.

Deux comptabilites :
  - Nref    : N de la reference (200 pour {1..10}, 447 pour {1,10}) ;
  - honest  : budget honnete : W1 dense de taille 632 (400 000 poids) +
              tranches structurees pour tau >= 2 (cout ~ 0), meme regle
              rho/sqrt(K) par tranche pour rester comparable a Nref.

Ensembles {1..10} et {1,10} ; taches NARMA-30, Silverbox, WH, MNIST ligne
par ligne, profil MC_k (k = 1..40) de 6.1 ; rho dans {0.5, 0.9} ; 5 seeds
(appariees : meme graine -> meme W_in quel que soit le type de tranche).

Diagnostic d'alignement (reseau linearise) : correlation entre l'empreinte
h(30) du chemin 10+10+10 (W_10^3 @ Win) et h(0) = Win : proche de 1 pour
ident (resonance de Stelzer), ~0 pour dense (orthogonalite des chemins).

Sorties : results_p7_tranches.json + partie7_tranches.svg.
Usage : python p7_1_tranches.py <narma30|silverbox|wh|mnist|mc|diag|figure>
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p7_common as C

KINDS = ["dense", "perm", "ident", "diag"]
ENSEMBLES = {"A10": list(range(1, 11)), "A110": [1, 10]}
RADII = [0.5, 0.9]
N_ESN = 632 if not C.SMOKE else 200
RJ = C.OUT / "results_p7_tranches.json"
FIG = C.OUT / "partie7_tranches.svg"
SEEDS = C.SEEDS5


def build_configs():
    """Liste des configs : (cle, ens, kind, acct, N, taus, kinds)."""
    out = []
    for ens, taus in ENSEMBLES.items():
        Nref = int(round((C.BUDGET / len(taus)) ** 0.5))
        for kind in KINDS:
            kinds = {t: ("dense" if t == 1 else kind) for t in taus}
            key = f"{ens}|{kind}|Nref"
            out.append((key, ens, kind, "Nref", Nref, taus, kinds))
            if kind != "dense":
                key = f"{ens}|{kind}|honest"
                out.append((key, ens, kind, "honest", N_ESN, taus, kinds))
    out.append(("esn|dense|Nref", "esn", "dense", "Nref", N_ESN, [1], {1: "dense"}))
    return out


def eval_seq(cfg, task, rho, seed, data):
    """Taches NARMA / Silverbox / WH. data = (U, y) pour la seed."""
    taus, N, kinds = cfg[5], cfg[4], cfg[6]
    if task.startswith("narma"):
        U, y, _ = C.narma30(seed)
    else:
        U, y = data
    Win, W, taus_s, N = C.make_reservoir(seed, taus, rho, N=N, kinds=kinds)
    F = C.run_states(U[:len(y)], Win, W, taus_s)
    sp = C.task_splits(task, taus_s)
    nrmse, lam, val = C.ridge_readout(F, y, sp["wash"], sp["tr_end"], sp["te0"], sp["te1"])
    return {"test": nrmse, "lam": lam, "val": val, "rho": rho}


def eval_mnist(cfg, rho, seed, mnist):
    Xtr, ytr, Xte, yte = mnist
    taus, N, kinds = cfg[5], cfg[4], cfg[6]
    rngseed = seed
    # ordre des tirages : Win (28,N) puis les tranches par tau croissant
    rng = np.random.default_rng(rngseed)
    Win = rng.uniform(-0.5, 0.5, size=(28, N)).astype(np.float32)
    W = np.empty((len(taus), N, N), np.float32)
    for i, tau in enumerate(sorted(taus)):
        kind = kinds.get(tau, "dense")
        W[i] = (C.slice_matrix(kind, rng, N) * (rho / np.sqrt(len(taus)))).astype(np.float32)
    taus_s = sorted(taus)
    Ftr = C.states_mnist(Xtr, Win, W, taus_s)
    Fte = C.states_mnist(Xte, Win, W, taus_s)
    acc, lam, val = C.mnist_readout(Ftr, ytr, Fte, yte)
    return {"test": acc, "lam": lam, "val": val, "rho": rho}


def eval_mc(cfg, rho, seed):
    taus, N, kinds = cfg[5], cfg[4], cfg[6]
    U, u = C.mc_input(seed)
    Win, W, taus_s, N = C.make_reservoir(seed, taus, rho, N=N, kinds=kinds)
    F = C.run_states(U, Win, W, taus_s)
    mc, mc_val = C.mc_profile(F, u, with_val=True)
    return {"mc": mc,
            "mc_sum": float(sum(mc.values())),
            "mc30": mc.get("30", 0.0),
            "val": float(sum(mc_val.values())),
            "rho": rho}


def run_diag(results):
    """Correlation empreinte h(30) (chemin 10+10+10) vs h(0) = Win.
    Le tenseur linearise donne h(30) contient W_10^3 @ Win[:,0] : pour ident
    les chemins s'alignent (corr ~ 1), pour dense ils sont quasi orthogonaux."""
    diag = results.setdefault("diag", {})
    for rho in RADII:
        for kind in KINDS:
            key = f"A10|{kind}|rho{rho}"
            cell = diag.setdefault(key, {"seeds": {}})
            for s in [x for x in SEEDS if str(x) not in cell["seeds"]]:
                taus = ENSEMBLES["A10"]
                kinds = {t: ("dense" if t == 1 else kind) for t in taus}
                Win, W, taus_s, N = C.make_reservoir(s, taus, rho, kinds=kinds)
                i10 = taus_s.index(10)
                W10 = W[i10].astype(np.float64)
                Win0 = Win[0].astype(np.float64)  # h(0) = W_in applique au canal u
                h30 = np.linalg.matrix_power(W10, 3) @ Win0
                if np.std(h30) < 1e-12:
                    corr = 0.0
                else:
                    corr = float(np.corrcoef(h30, Win0)[0, 1])
                cell["seeds"][str(s)] = corr
            cell["mean"] = float(np.mean(list(cell["seeds"].values())))
            print(f"  diag {key}: corr = {cell['mean']:.3f}", flush=True)
    C.save_json(results, RJ)


def figure(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    agg = results["agg"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    colors = {"dense": "#2a7de1", "perm": "#e1822a", "ident": "#444444", "diag": "#8bc34a"}
    titles = {"narma30": "NARMA-30 (NRMSE, log)", "silverbox": "Silverbox (NRMSE, log)",
              "wh": "Wiener-Hammerstein (NRMSE, log)", "mnist": "MNIST lignes (acc)"}
    # rang 0 : narma30, silverbox, wh ; rang 1 : mnist, MC, diag
    panels = [(0, 0, "narma30"), (0, 1, "silverbox"), (0, 2, "wh"),
              (1, 0, "mnist")]
    for row, col, task in panels:
        ax = axes[row, col]
        higher = task == "mnist"
        keys = [k for k in agg if k.startswith(f"{task}|")]
        if not keys:
            ax.set_visible(False)
            continue
        bars, labels = [], []
        for ki, kind in enumerate(KINDS):
            accts = [("Nref", "")] if kind == "dense" else [("Nref", ""), ("honest", "///")]
            for ai, (acct, hatch) in enumerate(accts):
                for j, rho in enumerate(RADII):
                    key = f"{task}|A10|{kind}|{acct}|rho{rho}"
                    if key in agg:
                        bars.append((ki + 0.2 * (2 * ai + j) - 0.3, agg[key]["mean"],
                                     agg[key]["std"], colors[kind], hatch, rho))
            labels.append(kind)
        for x, m, sd, col_, hatch, rho in bars:
            ax.bar(x, m, yerr=sd, color=col_, alpha=0.9 if rho == 0.9 else 0.45,
                   hatch=hatch, width=0.18, capsize=2)
        esn_keys = [k for k in agg if k.startswith(f"{task}|esn|")]
        if esn_keys:
            vals = [agg[k]["mean"] for k in esn_keys]
            ax.axhline(max(vals) if higher else min(vals),
                       color="#c62828", ls="dashed", lw=1.2,
                       label="ESN N=632 (meilleur ρ)")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        if not higher:
            ax.set_yscale("log")
        ax.set_title(titles[task])
        ax.grid(alpha=0.25, axis="y")
        if row == 0 and col == 0:
            handles = [plt.Rectangle((0, 0), 1, 1, fc="#2a7de1"),
                       plt.Rectangle((0, 0), 1, 1, fc="#2a7de1", hatch="///"),
                       plt.Rectangle((0, 0), 1, 1, fc="#2a7de1", alpha=0.45)]
            ax.legend(handles, ["N réf.", "budget honnête", "ρ=0.5 (pâle)"],
                      fontsize=8, loc="best")

    # panneau MC
    ax = axes[1, 1]
    for kind in KINDS:
        key = f"mc|A10|{kind}|Nref|rho0.9"
        cell = results["cells"].get(key)
        if not cell or "mc_profiles" not in cell:
            continue
        profs = list(cell["mc_profiles"].values())
        mcs = [np.mean([p[str(k)] for p in profs]) for k in range(1, 41)]
        ax.plot(range(1, 41), mcs, label=kind, color=colors[kind])
    ax.set_xlabel("k (retard)")
    ax.set_ylabel("MC_k")
    ax.set_title("Profil MC_k, {1..10}, ρ=0.9")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    # panneau diag
    ax = axes[1, 2]
    diag = results.get("diag", {})
    if diag:
        kinds_ = KINDS
        xs = np.arange(len(kinds_))
        for j, rho in enumerate(RADII):
            vals = [diag.get(f"A10|{k}|rho{rho}", {}).get("mean", np.nan) for k in kinds_]
            ax.bar(xs + 0.25 * j - 0.25, vals, width=0.22,
                   alpha=0.9 if rho == 0.9 else 0.45, color="#5e35b1")
        ax.set_xticks(xs)
        ax.set_xticklabels(kinds_)
        ax.set_ylabel("corr(h(30) chemin 10+10+10, h(0))")
        ax.set_title("Alignement des empreintes (réseau linéarisé)")
        ax.grid(alpha=0.25, axis="y")

    fig.suptitle("Tranches structurées contre denses (± é.-t., 5 seeds)")
    fig.tight_layout()
    fig.savefig(FIG)
    print(f"figure -> {FIG}")


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "narma30"
    results = C.load_json(RJ)
    C.meta_init(results, "p7_1_tranches.py", kinds=KINDS,
                ensembles={k: v for k, v in ENSEMBLES.items()},
                radii=RADII, seeds=SEEDS, budget=C.BUDGET)

    if which == "figure":
        C.aggregate(results)
        C.save_json(results, RJ)
        figure(results)
        return
    if which == "diag":
        run_diag(results)
        return

    task = which
    cfgs = build_configs()
    data = None
    mnist = None
    if task == "silverbox" or task == "wh":
        data = C.load_phys(task)
    elif task == "mnist":
        mnist = C.get_mnist()

    t0 = time.time()
    stop = False
    for cfg in cfgs:
        if stop:
            break
        key0, ens, kind, acct, N, taus, kinds = cfg
        for rho in RADII:
            if stop:
                break
            key = f"{task}|{key0}|rho{rho}"
            missing = C.cell_missing(results, key, SEEDS)
            if not missing:
                continue
            for s in missing:
                if time.time() - t0 > C.TIME_BUDGET:
                    stop = True
                    break
                if task == "mnist":
                    val = eval_mnist(cfg, rho, s, mnist)
                elif task == "mc":
                    val = eval_mc(cfg, rho, s)
                else:
                    val = eval_seq(cfg, task, rho, s, data)
                scalar = val["mc_sum"] if task == "mc" else val["test"]
                cell = C.cell_add(results, key, s, scalar, ens=ens, kind=kind,
                                 acct=acct, N=N, taus=taus, rho=rho, task=task,
                                 lam=val.get("lam", 0), val=val.get("val"))
                if task == "mc":
                    cell.setdefault("mc_profiles", {})[str(s)] = val["mc"]
                print(f"{key} seed{s}: test={scalar:.4f} lam={val.get('lam', 0):.0e} "
                      f"({time.time()-t0:.0f}s)", flush=True)
                C.save_json(results, RJ)
    C.aggregate(results)
    C.save_json(results, RJ)
    done = all(not C.cell_missing(results, f"{task}|{c[0]}|rho{r}", SEEDS)
               for c in cfgs for r in RADII)
    print("COMPLETE" if done and not stop else ("TIME_STOP" if stop else "PARTIEL : relancer"))


if __name__ == "__main__":
    main()
