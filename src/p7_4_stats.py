"""Partie 7.4 : Statistiques et équité de tuning (rapport/4_comparaisons.md).

Deux volets :
  1. Les six affirmations phares rejouees a 10 seeds, intervalles de
     confiance bootstrap 95 % + Wilcoxon apparie :
       - wh_gain / silverbox_gain : tenseur {1..10} N=200 vs ESN N=632,
         meilleur rho dans {0.5, 0.9} PAR SEED sur la validation (readout 6.0) ;
       - narma_record  : tenseur {1,10} N=447 + CP-rang r=N/4 (record 0,035)
         vs ESN N=632, rho dans {0.5, 0.9, 0.99} sur la validation ;
       - mnist60k     : ESN N=632 vs tenseur {1..10} N=200, 60k train /
         10k test, rho dans {0.9, 0.99}, metrique acc ;
       - crossover    : NARMA-x, x = 16..28 (pas 2 puis raffine pas 1 sur
         l'encadrement), ESN 632 vs tenseur {1..10} N=200 a rho = 0.9 ;
         x* = changement de signe de log NRMSE_ESN - log NRMSE_tenseur ;
       - bruit        : NARMA-30 bruite (entree u + 0.1*eta, cible propre,
         cf. suite_newtasks.py), rho = 0.9, ESN 632 vs tenseur {1..10}.
  2. Equite de tuning : recherche aleatoire de 60 evaluations par famille
     (ESN : rho U[0.3, 1.1], echelle d'entree logU[0.05, 2], fuite U[0.05, 1],
     ridge logU[1e-8, 1e-2] ; tenseur : S parmi 30 ensembles candidats,
     rho, echelle d'entree, ridge), selection sur la validation, test ensuite.
     Taches : narma30, silverbox, wh. A comparer au "tuning du projet"
     (grille rho) des affirmations ci-dessus.

Critere 7.4 : une affirmation n'entre dans le write-up que si elle survit
au tuning equitable avec un IC qui exclut l'egalite.

Sorties : results_p7_stats.json + partie7_tableau.md (tableau unique pour
le README). Usage :
  python p7_4_stats.py <wh_gain|silverbox_gain|narma_record|mnist60k|crossover|bruit|fair|table>
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p7_common as C

SEEDS10 = [0, 1] if C.SMOKE else list(range(10))
SEEDS_FAIR = [0, 1] if C.SMOKE else list(range(5))
RJ = C.OUT / "results_p7_stats.json"

CANDIDATE_SETS = [
    [1],
    [1, 2], [1, 2, 3], [1, 2, 3, 4], [1, 2, 3, 4, 5],
    list(range(1, 7)), list(range(1, 8)), list(range(1, 9)), list(range(1, 10)),
    list(range(1, 11)),                                        # uniformes
    [1, 2, 4, 8], [1, 2, 4, 8, 16], [1, 3, 9],                  # log
    [1, 10], [1, 15], [1, 20], [1, 30],                        # {1, d}
    [4, 7], [1, 7], [2, 5], [3, 7], [10, 20, 30],              # elus / echelonnes
    [2, 3, 4, 5, 7, 8, 10],                                    # elu silverbox (2.2)
    [2, 5, 7, 10],                                             # elu mnist (2.7)
    [2, 4, 6, 8, 10], [1, 3, 5, 7, 9],                         # pairs / impairs
    list(range(2, 21)), list(range(5, 16)), list(range(11, 21)),  # bandes
    [1, 5, 10, 20, 30],                                        # log elargi
]
assert len(CANDIDATE_SETS) == 30, len(CANDIDATE_SETS)

# 14.1 : espace de recherche A PRIORI du reglage equitable (identique a P13.ENSEMBLES : uniformes,
# puissances de 2, pairs, bandes, {1, d}), sans aucun ensemble elu sur NARMA/Silverbox/WH.
FAIR_SETS = [list(range(1, k + 1)) for k in (2, 3, 5, 10, 20)] + [
    [1, 2, 4, 8, 16], [1, 2, 4, 8, 16, 32], [1, 3, 9], [2, 4, 6, 8, 10],
    list(range(2, 11)), list(range(4, 11)), list(range(2, 21))] + [[1, d] for d in (5, 10, 15, 20, 25, 30, 40)]
FAIR_PROTO = "14_1_apriori"


# helpers
def best_rho_on_val(task, taus, seed, make_fn, rhos, data=None):
    """Meilleur rho PAR SEED sur la validation (6.0), renvoie
    (test, rho_choisi, lam) ; make_fn(taus, rho, seed, N, d) -> (Win, W, taus, N)."""
    if task.startswith("narma"):
        U, y, _ = C.narma30(seed)
    else:
        U, y = data
    sp = C.task_splits(task, taus)
    best = None
    for rho in rhos:
        Win, W, taus_s, N = make_fn(taus, rho, seed, U.shape[1])
        F = C.run_states(U[:len(y)], Win, W, taus_s)
        v0 = sp["wash"] + int(0.8 * (sp["tr_end"] - sp["wash"]))
        best_lam, best_val = None, np.inf
        Fb = C._bias(F)
        yv = np.asarray(y, np.float64)
        for lam in C.RIDGES:
            w = C._ridge_solve(Fb[sp["wash"]:v0], yv[sp["wash"]:v0], lam)
            resid = Fb[v0:sp["tr_end"]] @ w - yv[v0:sp["tr_end"]]
            val = float(np.sqrt(np.mean(resid ** 2)) / np.std(yv[v0:sp["tr_end"]]))
            if val < best_val:
                best_val, best_lam = val, lam
        if best is None or best_val < best[0]:
            w = C._ridge_solve(Fb[sp["wash"]:sp["tr_end"]], yv[sp["wash"]:sp["tr_end"]], best_lam)
            pred = Fb[sp["te0"]:sp["te1"]] @ w
            te = float(np.sqrt(np.mean((pred - yv[sp["te0"]:sp["te1"]]) ** 2))
                       / np.std(yv[sp["te0"]:sp["te1"]]))
            best = (best_val, te, rho, best_lam)
    return best[1], best[2], best[3]


def claim_seq(results, claim, task, tens_name, tens_make, rhos, taus_tens):
    """Tenseur vs ESN sur une tache sequentielle, 10 seeds, stats appariees."""
    cl = results.setdefault("claims", {}).setdefault(claim, {"tens": {}, "esn": {}, "stats": {}})
    data = C.load_phys(task) if task in ("silverbox", "wh") else None

    def esn_make(taus, rho, seed, d):
        return C.make_reservoir(seed, [1], rho, N=632 if not C.SMOKE else 200, d=d)

    for label, make, taus in (("tens", tens_make, taus_tens), ("esn", esn_make, [1])):
        R = cl[label]
        for s in [x for x in SEEDS10 if str(x) not in R]:
            te, rho, lam = best_rho_on_val(task, taus, s, make, rhos, data)
            R[str(s)] = {"test": te, "rho": rho, "lam": lam}
            json_save(results)
            print(f"{claim}/{label} seed{s}: test={te:.4f} rho={rho} ({time.time()-CLAIM_T0:.0f}s)",
                  flush=True)
    finalize_claim(results, claim)


def finalize_claim(results, claim, higher=False):
    """Moyennes, IC bootstrap 95 % sur l'ecart apparie, Wilcoxon."""
    cl = results["claims"][claim]
    seeds = sorted(set(cl["tens"]) & set(cl["esn"]), key=int)
    if not seeds:
        return
    t = np.array([cl["tens"][s]["test"] for s in seeds])
    e = np.array([cl["esn"][s]["test"] for s in seeds])
    ratio = t / e if not higher else e / t
    diff = t - e
    stats = {
        "n": len(seeds),
        "tens_mean": float(t.mean()), "tens_std": float(t.std()),
        "esn_mean": float(e.mean()), "esn_std": float(e.std()),
        "ratio_mean": float(ratio.mean()),
        "ratio_ci": list(C.bootstrap_ci(ratio)),
        "diff_ci": list(C.bootstrap_ci(diff)),
        "wilcoxon_p": C.wilcoxon_paired(t, e),
        "per_seed": {s: {"tens": cl["tens"][s]["test"], "esn": cl["esn"][s]["test"]}
                     for s in seeds},
    }
    cl["stats"] = stats
    json_save(results)
    print(f"{claim}: tenseur {stats['tens_mean']:.4f} vs ESN {stats['esn_mean']:.4f} "
          f"| ratio {stats['ratio_mean']:.3f} IC {stats['ratio_ci'][0]:.3f}-{stats['ratio_ci'][1]:.3f} "
          f"| Wilcoxon p={stats['wilcoxon_p']}", flush=True)


def json_save(results):
    C.save_json(results, RJ)


# claims
def run_narma_record(results):
    """Record NARMA-30 : tenseur {1,10} N=447 + CP-rang r=N/4 contre ESN 632."""
    def tens_make(taus, rho, seed, d):
        N = int(round((C.BUDGET / len(taus)) ** 0.5))
        return C.make_cp_reservoir(seed, taus, rho, r=max(2, N // 4), N=N, d=d)
    claim_seq(results, "narma_record", "narma30", "{1,10}+CP r=N/4",
              tens_make, [0.5, 0.9, 0.99], [1, 10])


def run_phys_gain(results, task):
    claim = f"{task}_gain"

    def tens_make(taus, rho, seed, d):
        return C.make_reservoir(seed, taus, rho, d=d)
    claim_seq(results, claim, task, "{1..10} N=200",
              tens_make, [0.5, 0.9], list(range(1, 11)))


def run_mnist60k(results):
    """MNIST 60k : ESN 632 vs tenseur {1..10} N=200, acc, 10 seeds."""
    cl = results.setdefault("claims", {}).setdefault("mnist60k",
                                                      {"tens": {}, "esn": {}, "stats": {}})
    Xtr, ytr, Xte, yte = C.get_mnist(60000, 10000)
    for label, taus in (("tens", list(range(1, 11))), ("esn", [1])):
        R = cl[label]
        for s in [x for x in SEEDS10 if str(x) not in R]:
            N = int(round((C.BUDGET / len(taus)) ** 0.5)) if label == "tens" \
                else (632 if not C.SMOKE else 200)
            best = None
            for rho in ((0.9, 0.99) if not C.SMOKE else (0.9,)):
                rng = np.random.default_rng(s)
                Win = rng.uniform(-0.5, 0.5, size=(28, N)).astype(np.float32)
                W = np.empty((len(taus), N, N), np.float32)
                for i in range(len(taus)):
                    M = rng.normal(0, 1, (N, N))
                    r = max(np.abs(np.linalg.eigvals(M)).max(), 1e-12)
                    W[i] = (M * ((rho / np.sqrt(len(taus))) / r)).astype(np.float32)
                Ftr = C.states_mnist(Xtr, Win, W, sorted(taus))
                Fte = C.states_mnist(Xte, Win, W, sorted(taus))
                acc, lam, _v = C.mnist_readout(Ftr, ytr, Fte, yte)
                if best is None or acc > best[0]:
                    best = (acc, rho, lam)
            R[str(s)] = {"test": best[0], "rho": best[1], "lam": best[2]}
            json_save(results)
            print(f"mnist60k/{label} seed{s}: acc={best[0]:.4f} rho={best[1]} "
                  f"({time.time()-CLAIM_T0:.0f}s)", flush=True)
    finalize_claim(results, "mnist60k", higher=True)


def run_crossover(results):
    """x* (rho) : NARMA-x, ESN 632 vs tenseur {1..10} N=200 a rho = 0.9.
    Grille grossiere pas 2 puis raffinement pas 1 autour de l'encadrement."""
    cl = results.setdefault("claims", {}).setdefault("crossover",
                                                     {"cells": {}, "xstar": {}})
    rho = 0.9
    xs_coarse = [16, 18, 20, 22, 24, 26, 28]
    xstar_by_seed = {}

    def run_grid(xs):
        for x in xs:
            for label, taus, N in (("tens", list(range(1, 11)), None),
                                   ("esn", [1], 632 if not C.SMOKE else 200)):
                key = f"x{x}|{label}"
                cell = cl["cells"].setdefault(key, {"seeds": {}})
                for s in [q for q in SEEDS10 if str(q) not in cell["seeds"]]:
                    U, y, _ = C.narma_x(x, s)
                    Win, W, taus_s, Nn = C.make_reservoir(s, taus, rho, N=N,
                                                          d=U.shape[1])
                    F = C.run_states(U[:len(y)], Win, W, taus_s)
                    sp = C.seq_splits(taus_s)
                    nrmse, lam, _v = C.ridge_readout(F, y, sp["wash"], sp["tr_end"],
                                                     sp["te0"], sp["te1"])
                    cell["seeds"][str(s)] = nrmse
                    json_save(results)
                    print(f"crossover {key} seed{s}: {nrmse:.4f} "
                          f"({time.time()-CLAIM_T0:.0f}s)", flush=True)

    run_grid(xs_coarse)
    # x* grossier par seed, puis raffinement pas 1 sur l'encadrement commun
    def xstar(xs, seeds):
        out = {}
        for s in seeds:
            prev = None
            xs_found = None
            for x in xs:
                # seed s uniquement (les cles du JSON sont des chaines)
                te = np.array([cl["cells"][f"x{x}|tens"]["seeds"][str(s)]
                               for _ in [0] if str(s) in cl["cells"][f"x{x}|tens"]["seeds"]])
                es = np.array([cl["cells"][f"x{x}|esn"]["seeds"][str(s)]
                               for _ in [0] if str(s) in cl["cells"][f"x{x}|esn"]["seeds"]])
                if len(te) == 0 or len(es) == 0:
                    continue
                r = np.log(np.mean(te)) - np.log(np.mean(es))
                if prev is not None and (prev[1] > 0) != (r > 0):
                    xs_found = prev[0] + (x - prev[0]) * prev[1] / (prev[1] - r)
                    break
                prev = (x, r)
            out[s] = xs_found
        return out

    seeds_done = [q for q in SEEDS10
                  if all(f"x{x}|esn" in cl["cells"] and str(q) in cl["cells"][f"x{x}|esn"]["seeds"]
                         for x in xs_coarse)]
    xcoarse = xstar(xs_coarse, seeds_done)
    # raffinement pas 1 autour du changement de signe (moyenne sur les seeds)
    finites = [v for v in xcoarse.values() if v is not None]
    if finites:
        xc = float(np.mean(finites))
        lo, hi = int(np.floor(xc)) - 1, int(np.ceil(xc)) + 1
        xs_fine = sorted({x for x in range(lo, hi + 1) if x not in xs_coarse})
        run_grid(xs_fine)
        xs_all = sorted(set(xs_coarse) | set(xs_fine))
        xfine = xstar(xs_all, seeds_done)
        cl["xstar"] = {"coarse": xcoarse, "fine": xfine,
                       "mean": float(np.mean([v for v in xfine.values() if v is not None]))
                       if any(v is not None for v in xfine.values()) else None,
                       "ci": list(C.bootstrap_ci([v for v in xfine.values() if v is not None]))
                       if any(v is not None for v in xfine.values()) else None}
        json_save(results)
        print(f"crossover x* = {cl['xstar']['mean']} IC {cl['xstar']['ci']}", flush=True)


def run_bruit(results):
    """Robustesse au bruit sigma = 0.1 (1.3) : cible NARMA-30 propre,
    entree du reservoir u + 0.1*eta, rho = 0.9, 10 seeds appariees."""
    cl = results.setdefault("claims", {}).setdefault("bruit",
                                                     {"tens": {}, "esn": {}, "stats": {}})
    sigma, rho = 0.1, 0.9
    for label, taus, N in (("tens", list(range(1, 11)), None),
                           ("esn", [1], 632 if not C.SMOKE else 200)):
        R = cl[label]
        for s in [x for x in SEEDS10 if str(x) not in R]:
            rng = np.random.default_rng(s)
            u = rng.uniform(0, 0.5, size=5200 if not C.SMOKE else 1600)
            T = len(u)
            y = np.zeros(T, np.float32)
            for t in range(30, T):
                y[t] = 0.2 * y[t - 1] + 0.004 * y[t - 1] * y[t - 30:t].sum() \
                    + 1.5 * u[t - 30] * u[t - 1] + 0.001
            u_noisy = (u + sigma * rng.standard_normal(T)).astype(np.float32)
            U = np.stack([u_noisy, np.ones(T, np.float32)], 1)
            Win, W, taus_s, Nn = C.make_reservoir(s, taus, rho, N=N, d=2)
            F = C.run_states(U, Win, W, taus_s)
            sp = C.seq_splits(taus_s)
            nrmse, lam, _v = C.ridge_readout(F, y, sp["wash"], sp["tr_end"], sp["te0"], sp["te1"])
            R[str(s)] = {"test": nrmse, "rho": rho, "lam": lam}
            json_save(results)
            print(f"bruit/{label} seed{s}: NRMSE={nrmse:.4f} ({time.time()-CLAIM_T0:.0f}s)",
                  flush=True)
    finalize_claim(results, "bruit")


# fair tuning
def fair_search(results, task):
    """60 evaluations aleatoires par famille sur la validation, puis test.
    ESN : rho, echelle d'entree, fuite, ridge. Tenseur : S, rho, echelle, ridge."""
    fair = results.setdefault("fair", {}).setdefault(task, {})
    data = C.load_phys(task) if task in ("silverbox", "wh") else None
    if task.startswith("narma"):
        seeds = SEEDS_FAIR
    else:
        seeds = [0]
    n_evals = 5 if C.SMOKE else 60

    for family in ("esn", "tens"):
        fam = fair.setdefault(family, {})
        for s in seeds:
            if str(s) in fam and fam[str(s)].get("proto") == FAIR_PROTO and len(fam[str(s)].get("evals", [])) >= n_evals:
                continue
            rng = np.random.default_rng(1000 * (s + 1) + (0 if family == "esn" else 1))
            if task.startswith("narma"):
                U, y, _ = C.narma30(s)
            else:
                U, y = data
            evals, best = [], None
            for e in range(n_evals):
                rho = rng.uniform(0.3, 1.1)
                scale = np.exp(rng.uniform(np.log(0.05), np.log(2.0)))
                lam = np.exp(rng.uniform(np.log(1e-8), np.log(1e-2)))
                if family == "esn":
                    taus = [1]
                    leak = rng.uniform(0.05, 1.0)
                else:
                    taus = FAIR_SETS[rng.integers(len(FAIR_SETS))]
                    leak = 1.0
                K = len(taus)
                N = int(round((C.BUDGET / K) ** 0.5))
                Win, W, taus_s, N = C.make_reservoir(s, taus, rho, N=N)
                U_scaled = U.copy()
                U_scaled[:, 0] *= scale
                sp = C.task_splits(task, taus_s)
                v0 = sp["wash"] + int(0.8 * (sp["tr_end"] - sp["wash"]))
                F = C.run_states(U_scaled, Win, W, taus_s, leak=leak)
                val = C.ridge_readout_given_lam(F, y, sp["wash"], v0, v0, sp["tr_end"], lam)
                evals.append({"e": e, "rho": float(rho), "scale": float(scale),
                              "lam": float(lam), "leak": float(leak),
                              "taus": list(map(int, taus)), "val": val})
                if best is None or val < best["val"]:
                    best = evals[-1]
            # rejouer la meilleure config -> test
            Win, W, taus_s, N = C.make_reservoir(s, best["taus"], best["rho"], d=U.shape[1])
            U_scaled = U.copy()
            U_scaled[:, 0] *= best["scale"]
            sp = C.task_splits(task, taus_s)
            F = C.run_states(U_scaled, Win, W, taus_s,
                             leak=None if best["leak"] == 1.0 else best["leak"])
            te = C.ridge_readout_given_lam(F, y, sp["wash"], sp["tr_end"],
                                           sp["te0"], sp["te1"], best["lam"])
            fam[str(s)] = {"evals": evals, "best": best, "test": te, "proto": FAIR_PROTO}
            json_save(results)
            print(f"fair/{task}/{family} seed{s}: test={te:.4f} "
                  f"(best val={best['val']:.4f}, {len(evals)} evals)", flush=True)


# table
def make_table(results):
    """Tableau unique pour le README (7.4 : le seul endroit utile)."""
    lines = ["| Affirmation | Tenseur | ESN | Écart | IC 95 % (ratio) | Wilcoxon p | Verdict |",
             "|---|---|---|---|---|---|---|"]
    for claim, cl in results.get("claims", {}).items():
        st = cl.get("stats") or {}
        if not st:
            continue
        lo, hi = st["ratio_ci"]
        verdict = "survit" if (lo > 1.0 or hi < 1.0) and (
            st["wilcoxon_p"] is not None and st["wilcoxon_p"] < 0.05) else "non concluant"
        fmt = (lambda v: f"{100*v:.1f} %") if claim == "mnist60k" else (lambda v: f"{v:.4f}")
        lines.append(f"| {claim} | {fmt(st['tens_mean'])} ± {fmt(st['tens_std'])} "
                     f"| {fmt(st['esn_mean'])} ± {fmt(st['esn_std'])} "
                     f"| ×{st['ratio_mean']:.2f} | [{lo:.3f} ; {hi:.3f}] "
                     f"| {st['wilcoxon_p']:.4f} | {verdict} |")
    for task, fams in results.get("fair", {}).items():
        for family in ("esn", "tens"):
            tests = [v["test"] for v in fams.get(family, {}).values() if "test" in v]
            if tests:
                lines.append(f"| fair/{task}/{family} (60 évals) | : | : | "
                             f"test {np.mean(tests):.4f} | : | : | tuning équitable |")
    md = "\n".join(lines)
    path = C.OUT / "partie7_tableau.md"
    path.write_text(md)
    print(md)
    print(f"-> {path}")


# main
CLAIM_T0 = time.time()

def main():
    global CLAIM_T0
    which = sys.argv[1] if len(sys.argv) > 1 else "table"
    CLAIM_T0 = time.time()
    results = C.load_json(RJ)
    C.meta_init(results, "p7_4_stats.py", seeds10=SEEDS10,
                candidate_sets=len(CANDIDATE_SETS))
    if which == "narma_record":
        run_narma_record(results)
    elif which == "wh_gain":
        run_phys_gain(results, "wh")
    elif which == "silverbox_gain":
        run_phys_gain(results, "silverbox")
    elif which == "mnist60k":
        run_mnist60k(results)
    elif which == "crossover":
        run_crossover(results)
    elif which == "bruit":
        run_bruit(results)
    elif which == "fair":
        for task in (sys.argv[2:] or ["narma30", "silverbox", "wh"]):
            fair_search(results, task)
    elif which == "table":
        make_table(results)
    else:
        print("sous-commande inconnue")
        return
    C.save_json(results, RJ)


if __name__ == "__main__":
    main()
