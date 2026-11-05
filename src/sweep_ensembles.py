"""Partie 2.1 : ensembles echantillonnes une couche sur deux.
Question : perd-on en performance avec {2,4,6,8,10} (sans tau=1) ou {1,3,5,7,9} vs {1..10} ?
Deux budgets : N=200 (= moitie des connexions, comme la campagne 1) et N=283 (= budget egal
400k). Protocole campagne 1/2 : narma30 wash 200 / test 2000, meilleur rho de {0.5,0.9,0.99} ;
silverbox/wh train 20k / wash 500 / test 10k, meilleur rho de {0.5,0.9} ; mnist ligne par ligne
5k/1k, meilleur rho de {0.9,0.99}. 3 seeds. Checkpoint : part2/results_ensembles.json.
Protocole corrige (14.2, 2026-10-29) : le meilleur rho par seed est choisi sur un
segment de VALIDATION (NVAL[task] pas avant le test ; mnist : 1000 dernieres
images du train), jamais sur le test ; readout final reajuste sur train+validation.
Les seeds stockees en dict ("test"/"per_rho") sont reprises telles quelles, les
anciennes (floats, choix sur le test) sont recalculees."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from chemins import BRUT  # noqa: E402
import numpy as np, json, time, sys, gzip, os
from pathlib import Path
from scipy.io import loadmat

OUT = (BRUT / "tensor-suite/part2")
RJ = OUT / "results_ensembles.json"
results = json.load(open(RJ)) if RJ.exists() else {"cells": {}}

SEEDS = [0, 1, 2]
TIME_BUDGET = float(__import__("os").environ.get("TEMPS_MAX", "215"))
CONFIGS = {
    "even_N200": dict(taus=[2, 4, 6, 8, 10], N=200),
    "odd_N200":  dict(taus=[1, 3, 5, 7, 9], N=200),
    "even_N283": dict(taus=[2, 4, 6, 8, 10], N=283),
    "odd_N283":  dict(taus=[1, 3, 5, 7, 9], N=283),
}

def narma30(seed, T=5200):
    rng = np.random.default_rng(seed)
    u = rng.uniform(0, 0.5, size=T)
    y = np.zeros(T, np.float32)
    a, b, g, d = 0.2, 0.004, 1.5, 0.001
    n = 30
    for t in range(n, T):
        y[t] = a * y[t-1] + b * y[t-1] * y[t-n:t].sum() + g * u[t-n] * u[t-1] + d
    return np.stack([u, np.ones_like(u)], 1).astype(np.float32), y

def load_phys(which):
    if which == "silverbox":
        d = loadmat("/tmp/datasets/silverbox/SilverboxFiles/SNLS80mV.mat")
        u = d["V1"].ravel().astype(np.float64); y = d["V2"].ravel().astype(np.float64)
    else:
        d = loadmat("/tmp/datasets/WH.mat")
        u = d["uBenchMark"].ravel().astype(np.float64); y = d["yBenchMark"].ravel().astype(np.float64)
    s = np.std(y[:100000])
    U = u[:30000]; Y = y[:30000] / s
    uu = np.stack([U / np.std(U), np.ones_like(U)], 1).astype(np.float32)
    return uu, Y

def get_mnist(n_train=5000, n_test=1000):
    import urllib.request
    BASE = "/tmp/mnist"
    os.makedirs(BASE, exist_ok=True)
    files = ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
             "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]
    for f in files:
        p = os.path.join(BASE, f)
        if not os.path.exists(p):
            urllib.request.urlretrieve(f"https://ossci-datasets.s3.amazonaws.com/mnist/{f}", p)
    def imgs(fn):
        with gzip.open(os.path.join(BASE, fn), "rb") as f:
            d = np.frombuffer(f.read(), np.uint8, offset=16)
        return d.reshape(-1, 28, 28).astype(np.float32) / 255.0
    def labs(fn):
        with gzip.open(os.path.join(BASE, fn), "rb") as f:
            return np.frombuffer(f.read(), np.uint8, offset=8).astype(np.int64)
    return imgs(files[0])[:n_train], labs(files[1])[:n_train], imgs(files[2])[:n_test], labs(files[3])[:n_test]

def make_W(N, taus, rho, seed):
    rng = np.random.default_rng(seed)
    W = []
    for _ in taus:
        M = rng.normal(0, 1, (N, N)).astype(np.float32)
        r = max(np.abs(np.linalg.eigvals(M.astype(np.float64))).max(), 1e-12)
        W.append((M * (rho / np.sqrt(len(taus)) / r)).astype(np.float32))
    return W

def run_seq(u, y, N, taus, rho, seed, wash, ntest, nval):
    rng = np.random.default_rng(seed + 7)
    Win = rng.uniform(-0.5, 0.5, size=(u.shape[1], N)).astype(np.float32)
    W = make_W(N, taus, rho, seed)
    T = len(u); tmax = max(taus)
    hist = np.zeros((tmax, N), np.float32)
    x_ = np.zeros(N, np.float32)
    F = np.empty((T, N), np.float32)
    Warr = np.stack(W)
    sel = np.array(taus) - 1
    for t in range(T):
        x_ = np.tanh(u[t] @ Win + np.einsum("kij,kj->i", Warr, hist[sel]))
        hist = np.concatenate([x_[None], hist[:-1]], axis=0)
        F[t] = x_
    Fb = np.hstack([F, np.ones((T, 1), np.float32)]).astype(np.float64)
    yv = np.asarray(y, np.float64)
    # 14.2 : renvoie (erreur de validation, erreur de test) ; le readout du test
    # est reajuste sur train+validation, apres le choix de rho sur la validation.
    Ftr, ytr = Fb[wash:-ntest-nval], yv[wash:-ntest-nval]
    Fva, yva = Fb[-ntest-nval:-ntest], yv[-ntest-nval:-ntest]
    Fte, yte = Fb[-ntest:], yv[-ntest:]
    A = Ftr.T @ Ftr + 1e-8 * np.eye(Ftr.shape[1])
    Wout = np.linalg.solve(A, Ftr.T @ ytr)
    val = float(np.sqrt(np.mean((Fva @ Wout - yva) ** 2)) / np.std(yva))
    Ftv, ytv = Fb[wash:-ntest], yv[wash:-ntest]
    A2 = Ftv.T @ Ftv + 1e-8 * np.eye(Ftv.shape[1])
    Wout2 = np.linalg.solve(A2, Ftv.T @ ytv)
    return val, float(np.sqrt(np.mean((Fte @ Wout2 - yte) ** 2)) / np.std(yte))

def states_mnist(Xb, N, taus, rho, seed, chunk=1000):
    rng = np.random.default_rng(seed)
    Win = rng.uniform(-0.5, 0.5, size=(28, N)).astype(np.float32)
    W = np.stack(make_W(N, taus, rho, seed + 1))
    tmax = max(taus)
    sel = np.array(taus) - 1
    out = np.empty((len(Xb), N), np.float32)
    for lo in range(0, len(Xb), chunk):
        B = Xb[lo:lo+chunk]
        x = np.zeros((len(B), N), np.float32)
        hist = np.zeros((tmax, len(B), N), np.float32)
        for t in range(28):
            drive = B[:, t, :] @ Win + np.tensordot(hist[sel], W, axes=([0, 2], [0, 2]))
            x = np.tanh(drive)
            hist = np.concatenate([x[None], hist[:-1]], axis=0)
        out[lo:lo+len(B)] = x
    return out

def classify_acc(Ftr, ytr, Fte, yte, ridge=1e-6):
    Ftr = np.hstack([Ftr, np.ones((len(Ftr), 1), np.float32)]).astype(np.float64)
    Fte = np.hstack([Fte, np.ones((len(Fte), 1), np.float32)]).astype(np.float64)
    Tt = np.zeros((len(ytr), 10)); Tt[np.arange(len(ytr)), ytr] = 1.0
    A = Ftr.T @ Ftr + ridge * np.eye(Ftr.shape[1])
    Wout = np.linalg.solve(A, Ftr.T @ Tt)
    return float((np.argmax(Fte @ Wout, axis=1) == yte).mean())

task = sys.argv[1]
RADII = {"narma30": [0.5, 0.9, 0.99], "silverbox": [0.5, 0.9], "wh": [0.5, 0.9], "mnist": [0.9, 0.99]}
PROTO = {"narma30": dict(wash=200, ntest=2000), "silverbox": dict(wash=500, ntest=10000),
         "wh": dict(wash=500, ntest=10000)}
NVAL = {"narma30": 1000, "silverbox": 2000, "wh": 2000}   # validation : choix de rho (14.2)
NVAL_MNIST = 1000

if task == "mnist":
    Xtr, ytr, Xte, yte = get_mnist()
else:
    u, y = narma30(0) if task == "narma30" else None, None
    if task != "narma30":
        u, y = load_phys(task)

t0 = time.time()
stop = False
for cname, cfg in CONFIGS.items():
    if stop: break
    key = f"{task}_{cname}"
    cell = results["cells"].setdefault(key, {"task": task, "config": cname, "taus": cfg["taus"], "N": cfg["N"], "seeds": {}})
    for s in SEEDS:
        if str(s) in cell["seeds"] and isinstance(cell["seeds"][str(s)], dict):
            continue  # deja calculee sous le protocole corrige (14.2)
        if time.time() - t0 > TIME_BUDGET: stop = True; break
        per_rho = {}
        if task == "mnist":
            best_val, best = -1.0, 0.0
            for rho in RADII[task]:
                Ftr = states_mnist(Xtr, cfg["N"], cfg["taus"], rho, s)
                Fte = states_mnist(Xte, cfg["N"], cfg["taus"], rho, s)
                va = classify_acc(Ftr[:-NVAL_MNIST], ytr[:-NVAL_MNIST],
                                  Ftr[-NVAL_MNIST:], ytr[-NVAL_MNIST:])
                per_rho[str(rho)] = {"val": va}
                if va > best_val:
                    best_val, best = va, classify_acc(Ftr, ytr, Fte, yte)
        else:
            if task == "narma30":
                u, y = narma30(s)
            best_val, best = np.inf, np.inf
            for rho in RADII[task]:
                va, te = run_seq(u, y, cfg["N"], cfg["taus"], rho, s, PROTO[task]["wash"], PROTO[task]["ntest"], NVAL[task])
                per_rho[str(rho)] = {"val": va, "test": te}
                if va < best_val:
                    best_val, best = va, te
        cell["seeds"][str(s)] = {"test": best, "per_rho": per_rho}
        print(f"{task} {cname} seed={s} score={best:.4f} ({time.time()-t0:.0f}s)", flush=True)
        results["cells"][key] = cell
        json.dump(results, open(RJ, "w"), indent=1)

remaining = sum(1 for t2 in RADII for c in CONFIGS
               if any(str(s) not in results["cells"].get(f"{t2}_{c}", {"seeds": {}})["seeds"]
                      or not isinstance(results["cells"].get(f"{t2}_{c}", {"seeds": {}})["seeds"][str(s)], dict)
                      for s in SEEDS))
print(f"restants : {remaining}")
if remaining == 0:
    print("COMPLETE")
    agg = {}
    for key, c in results["cells"].items():
        v = np.array([sv["test"] if isinstance(sv, dict) else sv for sv in c["seeds"].values()])
        agg[key] = dict(task=c["task"], config=c["config"], taus=c["taus"], N=c["N"],
                       mean=float(v.mean()), std=float(v.std()), n=len(v))
    results["agg"] = agg
    json.dump(results, open(RJ, "w"), indent=1)
    print("agregation ecrite")
