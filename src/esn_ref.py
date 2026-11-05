"""Reference ESN N=632 (budget 632^2 = 400k = parite) sous les DEUX protocoles
de la campagne 2, pour la heatmap de synthese (rapport NRMSE/NRMSE_ESN) :
  - protocole "std" : splits d'eval_pruned.py (narma wash 200 / train 3000 /
    test 2000 ; phys wash 500 / train 19,5k / test 10k ; mnist 5k/1k),
    rho choisi sur une VALIDATION prise a la fin du train (narma 1000, phys
    2000, mnist 1000), readout du test refit sur tout le train (14.2, corrige
    le 2026-11-02 ; avant : choix sur le test) ;
  - protocole "val" : splits 2.3-2.6/2.8 (narma 3000/1000/1000 ; phys
    train [500..15000) / val [15000..20000) / test [20000..30000) ; mnist
    4k/1k/1k), meilleur rho par seed A LA VALIDATION (disjointe).
ESN = W dense 632x632 de rayon rho, une seule tranche tau=1 : pas de tenseur.
Checkpoint : results_esn_ref.json."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from chemins import BRUT  # noqa: E402
import numpy as np, json, time, sys, gzip, os
from pathlib import Path
from scipy.io import loadmat

OUT = (BRUT / "tensor-suite/part2")
RJ = OUT / "results_esn_ref.json"
results = json.load(open(RJ, encoding="utf-8-sig")) if RJ.exists() else {}

N = 632
SEEDS = [0, 1, 2]

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
    return np.stack([U / np.std(U), np.ones_like(U)], 1).astype(np.float32), Y

def get_mnist():
    BASE = "/tmp/mnist"
    files = ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
             "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]
    def imgs(fn):
        with gzip.open(os.path.join(BASE, fn), "rb") as f:
            dd = np.frombuffer(f.read(), np.uint8, offset=16)
        return dd.reshape(-1, 28, 28).astype(np.float32) / 255.0
    def labs(fn):
        with gzip.open(os.path.join(BASE, fn), "rb") as f:
            return np.frombuffer(f.read(), np.uint8, offset=8).astype(np.int64)
    return imgs(files[0])[:5000], labs(files[1])[:5000], imgs(files[2])[:1000], labs(files[3])[:1000]

def make_esn(seed, rho):
    rng = np.random.default_rng(seed + 7)
    M = rng.normal(0, 1, (N, N)).astype(np.float32)
    r = max(np.abs(np.linalg.eigvals(M.astype(np.float64))).max(), 1e-12)
    return M * (rho / r)

def states_seq(u, W, seed):
    rng = np.random.default_rng(seed)
    T, d = u.shape
    Win = rng.uniform(-0.5, 0.5, size=(d, N)).astype(np.float32)
    x_ = np.zeros(N, np.float32)
    F = np.empty((T, N), np.float32)
    for t in range(T):
        x_ = np.tanh(u[t] @ Win + W @ x_)
        F[t] = x_
    return F

def states_mnist(Xb, W, seed, chunk=500):
    rng = np.random.default_rng(seed)
    Win = rng.uniform(-0.5, 0.5, size=(28, N)).astype(np.float32)
    out = np.empty((len(Xb), N), np.float32)
    for lo in range(0, len(Xb), chunk):
        B = Xb[lo:lo+chunk]
        x = np.zeros((len(B), N), np.float32)
        for t in range(28):
            x = np.tanh(B[:, t, :] @ Win + x @ W.T)
        out[lo:lo+len(B)] = x
    return out

task = sys.argv[1]
PROTO = sys.argv[2]  # "std" ou "val"
R = results.setdefault(task, {}).setdefault(PROTO, {"seeds": {}})

RADII = {"narma30": [0.5, 0.9, 0.99],
         "silverbox": [0.5, 0.9] if PROTO == "std" else [0.5, 0.9, 0.99],
         "wh": [0.5, 0.9] if PROTO == "std" else [0.5, 0.9, 0.99],
         "mnist": [0.9, 0.99]}
MAXIM = task == "mnist"

if task == "mnist":
    Xtr, ytr, Xte, yte = get_mnist()

t0 = time.time()
for s in SEEDS:
    if str(s) in R["seeds"] and R["seeds"][str(s)].get("proto") == "rho_sur_validation":
        continue
    if task == "narma30":
        u, y = narma30(s)
    elif task != "mnist":
        u, y = load_phys(task)
    best = None
    for rho in RADII[task]:
        W = make_esn(s, rho)
        if task != "mnist":
            F = states_seq(u, W, s)
            Fb = np.hstack([F, np.ones((len(F), 1), np.float32)]).astype(np.float64)
            yv = np.asarray(y, np.float64)
            if PROTO == "std":
                wash, ntest = 200, 2000
                Ftr_, ytr_ = Fb[wash:-ntest], yv[wash:-ntest]
                Fte_, yte_ = Fb[-ntest:], yv[-ntest:]
                nval = 1000 if task == "narma30" else 2000
                Fva_, yva_ = Ftr_[-nval:], ytr_[-nval:]      # 14.2 : validation = fin du train
                Fh_, yh_ = Ftr_[:-nval], ytr_[:-nval]
            else:
                Ftr_, ytr_ = Fb[200:3200], yv[200:3200]
                Fva_, yva_ = Fb[3200:4200], yv[3200:4200]
                Fte_, yte_ = Fb[4200:5200], yv[4200:5200]
        else:
            if PROTO == "std":
                Ftr = states_mnist(Xtr, W, s)
                Fte = states_mnist(Xte, W, s)
                Ftrb = np.hstack([Ftr, np.ones((len(Ftr), 1), np.float32)]).astype(np.float64)
                Fteb = np.hstack([Fte, np.ones((len(Fte), 1), np.float32)]).astype(np.float64)
                Tt = np.zeros((len(ytr), 10)); Tt[np.arange(len(ytr)), ytr] = 1.0
                nh = len(ytr) - 1000                          # 14.2 : validation = 1000 dernieres images
                Av = Ftrb[:nh].T @ Ftrb[:nh] + 1e-6 * np.eye(Ftrb.shape[1])
                Wv = np.linalg.solve(Av, Ftrb[:nh].T @ Tt[:nh])
                va = float((np.argmax(Ftrb[nh:] @ Wv, axis=1) == ytr[nh:]).mean())
                A = Ftrb.T @ Ftrb + 1e-6 * np.eye(Ftrb.shape[1])
                Wout = np.linalg.solve(A, Ftrb.T @ Tt)
                te = float((np.argmax(Fteb @ Wout, axis=1) == yte).mean())
            else:
                Ftr = states_mnist(Xtr[:4000], W, s)
                Fva = states_mnist(Xtr[4000:], W, s)
                Fte = states_mnist(Xte, W, s)
                def pre(F):
                    return np.hstack([F, np.ones((len(F), 1), np.float32)]).astype(np.float64)
                Tt = np.zeros((4000, 10)); Tt[np.arange(4000), ytr[:4000]] = 1.0
                A = pre(Ftr).T @ pre(Ftr) + 1e-6 * np.eye(pre(Ftr).shape[1])
                Wout = np.linalg.solve(A, pre(Ftr).T @ Tt)
                va = float((np.argmax(pre(Fva) @ Wout, axis=1) == ytr[4000:]).mean())
                te = float((np.argmax(pre(Fte) @ Wout, axis=1) == yte).mean())
            if best is None or va > best[0]:
                best = (va, te, rho)
            continue
        A = Ftr_.T @ Ftr_ + 1e-8 * np.eye(Ftr_.shape[1])
        Wout = np.linalg.solve(A, Ftr_.T @ ytr_)
        te = float(np.sqrt(np.mean((Fte_ @ Wout - yte_) ** 2)) / np.std(yte_))
        if PROTO == "std":
            Wv = np.linalg.solve(Fh_.T @ Fh_ + 1e-8 * np.eye(Fh_.shape[1]), Fh_.T @ yh_)
            va = float(np.sqrt(np.mean((Fva_ @ Wv - yva_) ** 2)) / np.std(yva_))
        else:
            va = float(np.sqrt(np.mean((Fva_ @ Wout - yva_) ** 2)) / np.std(yte_))
        sc = va
        if best is None or ((MAXIM and sc > best[0]) or (not MAXIM and sc < best[0])):
            best = (sc, te, rho)
    R["seeds"][str(s)] = {"test": best[1], "rho": best[2], "proto": "rho_sur_validation"}
    json.dump(results, open(RJ, "w", encoding="utf-8"), indent=1)
    print(f"{task}/{PROTO} seed{s} rho={best[2]} test={best[1]:.4f} ({time.time()-t0:.0f}s)", flush=True)

tests = [R["seeds"][str(s)]["test"] for s in SEEDS]
R["test_mean"] = float(np.mean(tests))
R["test_std"] = float(np.std(tests))
json.dump(results, open(RJ, "w", encoding="utf-8"), indent=1)
print(f"{task}/{PROTO} ESN 632: test={np.mean(tests):.4f} +/- {np.std(tests):.4f}")
