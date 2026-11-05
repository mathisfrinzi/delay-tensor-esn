"""Partie 2.5 : rayon spectral par tranche rho_tau individualise.
Les profils rho_tau = rho . w_k / sqrt(sum w_k^2) sur les 10 tranches baseline
(N=200) : budget spectral total sum rho_k^2 = rho^2 constant, seule la
repartition change. C'est la generalisation directe des bandes 2.4 (w=g).
Profils testes (les deux predictions opposees + intermediaires) :
  uniform    : w=1 (baseline = tenseur standard)
  croissant  : w=k   -> tranches longues PLUS fortes (prediction (a))
  decroissant: w=1/k -> tranches longues PLUS faibles (prediction (b))
  en-tente   : w=1-|k-5.5|/4.5 -> milieu fort (signature pruning 2.2)
  en-V       : w=|k-5.5|      -> extremes forts
Lecture attendue si l'un des profils bat uniform partout : prediction validee.
Verdict (cf. 2.3) : aucun profil impose ne doit battre le rho_tau APPRIS.
Protocole : splits identiques a 2.3/2.4 ; 3 seeds x rho balayage, meilleur rho
par seed a la validation, test reporte. Checkpoint : results_rho_tau.json."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from chemins import BRUT  # noqa: E402
import numpy as np, json, time, sys, gzip, os
from pathlib import Path
from scipy.io import loadmat

OUT = (BRUT / "tensor-suite/part2")
RJ = OUT / "results_rho_tau.json"
results = json.load(open(RJ)) if RJ.exists() else {}

K = 10
N = 200
PROFILES = {
    "uniform": np.ones(K),
    "croissant": np.arange(1, K + 1, dtype=float),
    "decroissant": 1.0 / np.arange(1, K + 1, dtype=float),
    "en-tente": 1 - np.abs(np.arange(1, K + 1) - 5.5) / 4.5,
    "en-V": np.abs(np.arange(1, K + 1) - 5.5),
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

def states_seq(u, W, Win):
    T = u.shape[0]
    hist = np.zeros((K, N), np.float32)
    x_ = np.zeros(N, np.float32)
    F = np.empty((T, N), np.float32)
    sel = np.arange(K)
    for t in range(T):
        x_ = np.tanh(u[t] @ Win + np.einsum("kij,kj->i", W, hist[sel]))
        hist = np.concatenate([x_[None], hist[:-1]], axis=0)
        F[t] = x_
    return F

def states_mnist(Xb, W, Win, chunk=1000):
    out = np.empty((len(Xb), N), np.float32)
    sel = np.arange(K)
    for lo in range(0, len(Xb), chunk):
        B = Xb[lo:lo+chunk]
        hist = np.zeros((K, len(B), N), np.float32)
        x = np.zeros((len(B), N), np.float32)
        for t in range(28):
            drive = B[:, t, :] @ Win + np.tensordot(hist[sel], W, axes=([0, 2], [0, 2]))
            x = np.tanh(drive)
            hist = np.concatenate([x[None], hist[:-1]], axis=0)
        out[lo:lo+len(B)] = x
    return out

task = sys.argv[1]
MAXIM = task == "mnist"
R = results.setdefault(task, {})

if task == "narma30":
    u, y = narma30(0)
    tr_sl, va_sl, te_sl = slice(200, 3200), slice(3200, 4200), slice(4200, 5200)
elif task in ("silverbox", "wh"):
    u, y = load_phys(task)
    tr_sl, va_sl, te_sl = slice(500, 15000), slice(15000, 20000), slice(20000, 30000)
else:
    Xtr, ytr, Xte, yte = get_mnist()

def evaluate(prof, seed=0, rho=0.9):
    w = PROFILES[prof]
    sn = np.sqrt((w ** 2).sum())
    rng = np.random.default_rng(seed + 7)
    W = np.empty((K, N, N), np.float32)
    for i in range(K):
        M = rng.normal(0, 1, (N, N)).astype(np.float32)
        r = max(np.abs(np.linalg.eigvals(M.astype(np.float64))).max(), 1e-12)
        W[i] = M * ((rho * w[i] / sn) / r)
    rng2 = np.random.default_rng(7 + 100 * seed)
    if task != "mnist":
        Win = rng2.uniform(-0.5, 0.5, size=(u.shape[1], N)).astype(np.float32)
        F = states_seq(u, W, Win)
        Fb = np.hstack([F, np.ones((len(F), 1), np.float32)]).astype(np.float64)
        yv = np.asarray(y, np.float64)
        A = Fb[tr_sl].T @ Fb[tr_sl] + 1e-8 * np.eye(Fb.shape[1])
        Wout = np.linalg.solve(A, Fb[tr_sl].T @ yv[tr_sl])
        va = float(np.sqrt(np.mean((Fb[va_sl] @ Wout - yv[va_sl]) ** 2)) / np.std(yv[te_sl]))
        te = float(np.sqrt(np.mean((Fb[te_sl] @ Wout - yv[te_sl]) ** 2)) / np.std(yv[te_sl]))
        return va, te
    Win = rng2.uniform(-0.5, 0.5, size=(28, N)).astype(np.float32)
    Ftr = states_mnist(Xtr[:4000], W, Win)
    Fva = states_mnist(Xtr[4000:], W, Win)
    Fte = states_mnist(Xte, W, Win)
    def pre(F):
        return np.hstack([F, np.ones((len(F), 1), np.float32)]).astype(np.float64)
    Tt = np.zeros((4000, 10)); Tt[np.arange(4000), ytr[:4000]] = 1.0
    A = pre(Ftr).T @ pre(Ftr) + 1e-6 * np.eye(pre(Ftr).shape[1])
    Wout = np.linalg.solve(A, pre(Ftr).T @ Tt)
    va = float((np.argmax(pre(Fva) @ Wout, axis=1) == ytr[4000:]).mean())
    te = float((np.argmax(pre(Fte) @ Wout, axis=1) == yte).mean())
    return va, te

t0 = time.time()
RADII = [0.9, 0.99] if task == "mnist" else [0.5, 0.9, 0.99]

for prof in PROFILES:
    cell = R.setdefault(prof, {"done": False})
    if cell["done"]:
        continue
    finals = []
    for seed in (0, 1, 2):
        best = None
        for rho in RADII:
            va, te = evaluate(prof, seed=seed, rho=rho)
            sc = va if task == "mnist" else -va
            if best is None or sc > best[0]:
                best = (sc, te, rho)
        finals.append({"seed": seed, "rho": best[2], "test": best[1]})
    tests = [f["test"] for f in finals]
    cell.update({"final": finals, "test_mean": float(np.mean(tests)),
                 "test_std": float(np.std(tests)), "done": True})
    json.dump(results, open(RJ, "w"), indent=1)
    print(f"{task} {prof}: test={np.mean(tests):.4f} +/- {np.std(tests):.4f} ({time.time()-t0:.0f}s)", flush=True)

print(f"{task} 2.5 rho_tau terminé")
