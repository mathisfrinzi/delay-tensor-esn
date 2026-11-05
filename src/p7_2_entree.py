"""Le retard doit-il être dans l'état ou suffit-il dans
l'entrée ? (rapport/4_comparaisons.md, cf. Picco et al. 2025)

ESN N = 632 nourri avec [u(t), u(t-d1), ..., u(t-dm), 1] (W_in (m+2) x N,
meme loi U[-0.5, 0.5] que 6.0), contre le tenseur :
  - ESN632  : ensembles d'entree {1}, {10}, {30}, {1,30}, {10,20,30}, {1..10}
              + ESN plain (reference budget egal) ;
  - tenseur {1,10} N=447 et {1..10} N=200 : SANS les entrees retardees
              (references) et AVEC chacune des memes ensembles d'entree.
Taches : NARMA-30, Silverbox, WH. rho dans {0.5, 0.9}, 5 seeds appariees.
Les retards d'entree valent 0 avant t = 0 (comme l'etat du reservoir).

Prediction pre-enregistree (7.2) : l'ESN a entree retardee ecrase NARMA-30
(NRMSE < 0.05) ; le tenseur garde l'avantage sur Silverbox et WH.

Sorties : results_p7_entree.json + partie7_entree.svg.
Usage : python p7_2_entree.py <narma30|silverbox|wh|figure>
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p7_common as C

DELAY_SETS = {
    "d1": [1],
    "d10": [10],
    "d30": [30],
    "d1_30": [1, 30],
    "d10_20_30": [10, 20, 30],
    "d1_10": list(range(1, 11)),
}
RADII = [0.5, 0.9]
SEEDS = C.SEEDS5
RJ = C.OUT / "results_p7_entree.json"
FIG = C.OUT / "partie7_entree.svg"


def delayed_input(u, delays):
    """[u(t), u(t-d) pour d dans delays (0 avant t=0), 1] -> (T, m+2)."""
    cols = [u]
    for d in delays:
        c = np.zeros_like(u)
        c[d:] = u[:-d]
        cols.append(c)
    cols.append(np.ones_like(u))
    return np.stack(cols, 1).astype(np.float32)


def build_configs():
    """(cle_config, famille, taus, delais)."""
    out = []
    for dname, delays in DELAY_SETS.items():
        out.append((f"esn632|{dname}", "esn", [1], delays))
    out.append(("esn632|plain", "esn", [1], []))
    for tens, taus in (("A10", list(range(1, 11))), ("A110", [1, 10])):
        out.append((f"{tens}|plain", "tens", taus, []))
        for dname, delays in DELAY_SETS.items():
            out.append((f"{tens}|{dname}", "tens", taus, delays))
    return out


def eval_config(cfg, task, rho, seed, data_phys=None):
    _, fam, taus, delays = cfg
    K = len(taus)
    N = int(round((C.BUDGET / K) ** 0.5))
    if task.startswith("narma"):
        U2, y, reg = C.narma30(seed)
        u = U2[:, 0]
    else:
        U2, y = data_phys
        u = U2[:, 0]
    U = delayed_input(u, delays)
    Win, W, taus_s, N = C.make_reservoir(seed, taus, rho, N=N, d=U.shape[1])
    F = C.run_states(U, Win, W, taus_s)
    sp = C.task_splits(task, taus_s)
    nrmse, lam, val = C.ridge_readout(F, y, sp["wash"], sp["tr_end"], sp["te0"], sp["te1"])
    return {"test": nrmse, "lam": lam, "val": val}


def figure(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    agg = results["agg"]
    tasks = ["narma30", "silverbox", "wh"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    labels = ["plain"] + list(DELAY_SETS)
    for r, rho in enumerate(RADII):
        for col, task in enumerate(tasks):
            ax = axes[r, col]
            keys = [k for k in agg if k.startswith(f"{task}|") and f"rho{rho}" in k]
            if not keys:
                ax.set_visible(False)
                continue
            # barres : ESN+retards, tenseurs plain et +retards
            xs, hs, cs, ls_ = [], [], [], []
            x = 0.0
            for dname in labels:
                key = f"{task}|esn632|{dname}|rho{rho}"
                if key in agg:
                    xs.append(x); hs.append(agg[key]["mean"])
                    cs.append("#e1822a"); ls_.append(dname); x += 1
            x += 0.5
            for tens in ("A10", "A110"):
                key = f"{task}|{tens}|plain|rho{rho}"
                if key in agg:
                    xs.append(x); hs.append(agg[key]["mean"])
                    cs.append("#2a7de1"); ls_.append(f"{tens} plain"); x += 1
                for dname in labels[1:]:
                    key = f"{task}|{tens}|{dname}|rho{rho}"
                    if key in agg:
                        xs.append(x); hs.append(agg[key]["mean"])
                        cs.append("#7e57c2"); ls_.append(f"{tens}+{dname}"); x += 1
                x += 0.5
            ax.bar(xs, hs, color=cs, width=0.8)
            ax.set_xticks(xs)
            ax.set_xticklabels(ls_, rotation=60, fontsize=7)
            ax.set_title(f"{task} : ρ={rho} (NRMSE)")
            ax.set_yscale("log")
            ax.grid(alpha=0.25, axis="y")
    fig.suptitle("Retards dans l'entrée vs retards dans l'état (5 seeds)")
    fig.tight_layout()
    fig.savefig(FIG)
    print(f"figure -> {FIG}")


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "narma30"
    results = C.load_json(RJ)
    C.meta_init(results, "p7_2_entree.py", delay_sets=DELAY_SETS,
                radii=RADII, seeds=SEEDS, budget=C.BUDGET)

    if which == "figure":
        C.aggregate(results)
        C.save_json(results, RJ)
        figure(results)
        return

    task = which
    cfgs = build_configs()
    data_phys = C.load_phys(task) if task in ("silverbox", "wh") else None
    t0 = time.time()
    stop = False
    for cfg in cfgs:
        if stop:
            break
        for rho in RADII:
            if stop:
                break
            key = f"{task}|{cfg[0]}|rho{rho}"
            missing = C.cell_missing(results, key, SEEDS)
            if not missing:
                continue
            for s in missing:
                if time.time() - t0 > C.TIME_BUDGET:
                    stop = True
                    break
                val = eval_config(cfg, task, rho, s, data_phys)
                C.cell_add(results, key, s, val["test"], family=cfg[1],
                            taus=cfg[2], delays=cfg[3], rho=rho, task=task,
                            lam=val["lam"], val=val["val"])
                print(f"{key} seed{s}: NRMSE={val['test']:.4f} ({time.time()-t0:.0f}s)",
                      flush=True)
                C.save_json(results, RJ)
    C.aggregate(results)
    C.save_json(results, RJ)
    done = all(not C.cell_missing(results, f"{task}|{c[0]}|rho{r}", SEEDS)
               for c in cfgs for r in RADII)
    print("COMPLETE" if done and not stop else ("TIME_STOP" if stop else "PARTIEL : relancer"))


if __name__ == "__main__":
    main()
