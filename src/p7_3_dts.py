"""Partie 7.3 : La famille concurrente : constantes de temps hétérogènes
(DTS-ESN, Tanaka et al., Phys. Rev. Research 2022).

x(t) = (1 - a) . x(t-1) + a . tanh(W_in @ [u, 1] + W @ x(t-1)), N = 632 :
  - DTS  : a_i ~ log-uniforme sur [a_min ; 1], a_min dans {0.01, 0.03, 0.1} ;
  - fuite uniforme : a dans {0.1, 0.3, 0.5, 1.0} (a = 1 = ESN standard).
Ordre des tirages par seed : W_in, W (dense 632, rayon rho), puis a (DTS).
Memes taches que 7.1 (NARMA-30, Silverbox, WH, MNIST lignes, MC),
rho dans {0.5, 0.7, 0.9, 0.99}, 5 seeds.

Remarque arithmetique (7.3) : la fuite ajoute une tranche tau = 1 identite
(gain 1 - a) ; le DTS-ESN allonge la memoire en ralentissant la decroissance,
pas en raccourcissant les chemins. Prediction : il bat l'ESN sur Silverbox
(dynamique lente) mais pas le tenseur sur NARMA-30 ni MNIST.

Sortie : results_p7_dts.json (pas de figure demandee).
Usage : python p7_3_dts.py <narma30|silverbox|wh|mnist|mc>
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p7_common as C

RADII = [0.5, 0.7, 0.9, 0.99]
A_MINS = [0.01, 0.03, 0.1]
A_UNIFORM = [0.1, 0.3, 0.5, 1.0]
SEEDS = C.SEEDS5
N = 632 if not C.SMOKE else 200
RJ = C.OUT / "results_p7_dts.json"


def build_configs():
    out = []
    for amin in A_MINS:
        out.append((f"dts_{amin}", "dts", amin))
    for a in A_UNIFORM:
        out.append((f"leak_{a}", "uniform", a))
    return out


def make_dts(seed, rho, mode, aval):
    """W_in, W dense rayon rho, puis a. mode = 'dts' (vecteur log-uniforme
    [aval, 1]) ou 'uniform' (scalaire aval). Renvoie (Win, W, [1], leak)."""
    rng = np.random.default_rng(seed)
    Win = rng.uniform(-0.5, 0.5, size=(2, N)).astype(np.float32)
    M = rng.normal(0, 1, (N, N)).astype(np.float64)
    r = max(np.abs(np.linalg.eigvals(M)).max(), 1e-12)
    W = (M * (rho / r)).astype(np.float32)[None]
    if mode == "dts":
        a = np.exp(rng.uniform(np.log(aval), 0.0, size=N)).astype(np.float32)
    else:
        a = float(aval)
    return Win, W, [1], a


def eval_config(cfg, task, rho, seed, data_phys=None, mnist=None):
    key0, mode, aval = cfg
    if task == "mnist":
        Xtr, ytr, Xte, yte = mnist
        rng = np.random.default_rng(seed)
        Win = rng.uniform(-0.5, 0.5, size=(28, N)).astype(np.float32)
        M = rng.normal(0, 1, (N, N)).astype(np.float64)
        r = max(np.abs(np.linalg.eigvals(M)).max(), 1e-12)
        W = (M * (rho / r)).astype(np.float32)[None]
        if mode == "dts":
            a = np.exp(rng.uniform(np.log(aval), 0.0, size=N)).astype(np.float32)
        else:
            a = float(aval)
        Ftr = states_mnist_leak(Xtr, Win, W, a)
        Fte = states_mnist_leak(Xte, Win, W, a)
        acc, lam, vacc = C.mnist_readout(Ftr, ytr, Fte, yte)
        return {"test": acc, "lam": lam, "val": vacc}
    if task == "mc":
        U, u = C.mc_input(seed)
        Win, W, taus, a = make_dts(seed, rho, mode, aval)
        F = C.run_states(U, Win, W, taus, leak=a)
        mc, mc_val = C.mc_profile(F, u, with_val=True)
        return {"mc": mc, "mc_sum": float(sum(mc.values())), "test": float(sum(mc.values())),
                "val": float(sum(mc_val.values())), "lam": 0}
    if task.startswith("narma"):
        U, y, _ = C.narma30(seed)
    else:
        U, y = data_phys
    Win, W, taus, a = make_dts(seed, rho, mode, aval)
    F = C.run_states(U, Win, W, taus, leak=a)
    sp = C.task_splits(task, taus)
    nrmse, lam, val = C.ridge_readout(F, y, sp["wash"], sp["tr_end"], sp["te0"], sp["te1"])
    return {"test": nrmse, "lam": lam, "val": val}


def states_mnist_leak(X, Win, W, a, chunk=500):
    """MNIST lignes avec fuite : x = (1-a) x_prev + a tanh(Win px + W x_prev),
    etat remis a zero par image, historique inutile (tau = 1 seul)."""
    B = len(X)
    Nn = Win.shape[1]
    out = np.empty((B, Nn), np.float32)
    leak_vec = isinstance(a, np.ndarray)
    for lo in range(0, B, chunk):
        Bk = X[lo:lo + chunk].reshape(-1, 28, 28)
        nb = len(Bk)
        x = np.zeros((nb, Nn), np.float32)
        for t in range(28):
            z = np.tanh(Bk[:, t, :] @ Win + x @ W[0].T)
            if leak_vec:
                x = (1.0 - a[None, :]) * x + a[None, :] * z
            else:
                x = (1.0 - a) * x + a * z
        out[lo:lo + nb] = x
    return out


def main():
    task = sys.argv[1] if len(sys.argv) > 1 else "narma30"
    results = C.load_json(RJ)
    C.meta_init(results, "p7_3_dts.py", a_mins=A_MINS, a_uniform=A_UNIFORM,
                radii=RADII, seeds=SEEDS, N=N)
    cfgs = build_configs()
    data_phys = C.load_phys(task) if task in ("silverbox", "wh") else None
    mnist = C.get_mnist() if task == "mnist" else None

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
                val = eval_config(cfg, task, rho, s, data_phys, mnist)
                C.cell_add(results, key, s, val["test"], mode=cfg[1],
                            a=cfg[2], rho=rho, task=task, lam=val.get("lam"),
                            val=val.get("val"))
                if task == "mc" and "mc" in val:
                    results["cells"][key]["mc_last"] = val["mc"]
                print(f"{key} seed{s}: test={val['test']:.4f} ({time.time()-t0:.0f}s)",
                      flush=True)
                C.save_json(results, RJ)
    C.aggregate(results)
    C.save_json(results, RJ)
    done = all(not C.cell_missing(results, f"{task}|{c[0]}|rho{r}", SEEDS)
               for c in cfgs for r in RADII)
    print("COMPLETE" if done and not stop else ("TIME_STOP" if stop else "PARTIEL : relancer"))


if __name__ == "__main__":
    main()
