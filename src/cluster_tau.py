"""Partie 2.8 : tau par cluster de neurones : cas a 2 clusters.
Le N=200 est coupe en 2 sous-reservoirs independants (quasi-bloc-diagonal,
N/2=100 chacun) : le cluster A porte la bande [tau0,tau1] apprise depuis le
depart tenseur (1,10) (results_band_refine.json), le cluster B la bande
apprise depuis le depart ESN (1,1) (results_band_esn.json). Chaque cluster
a ses propres tranches, son propre Win, son historique : aucune interaction
inter-clusters. Rayon effectif par tranche : rho . g_k / sqrt(2 . sum g^2)
(le facteur 2 garde le budget spectral TOTAL constant : rho^2 pour l'ensemble,
comme la bande unique N=200). Question : le melange des deux bassins
bat-il chaque bande seule ? Protocole 2.4 : 3 seeds x rho, meilleur a la
validation. Checkpoint : results_cluster.json."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from chemins import BRUT  # noqa: E402
import numpy as np, json, time, sys, gzip, os
from pathlib import Path
from scipy.io import loadmat

OUT = (BRUT / "tensor-suite/part2")
RJ = OUT / "results_cluster.json"
results = json.load(open(RJ)) if RJ.exists() else {}

TMAX = 20
N = 200
NC = 100  # neurons par cluster

BANDS_A = json.load(open(OUT / "results_band_refine.json"))
BANDS_B = json.load(open(OUT / "results_band_esn.json"))

def gains(tau0, tau1):
    ks = np.arange(1, TMAX + 1)
    return np.clip(ks - tau0 + 1, 0, 1) * np.clip(tau1 - ks + 1, 0, 1)

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

task = sys.argv[1]
MAXIM = task == "mnist"
R = results.setdefault(task, {})

gA = gains(BANDS_A[task]["final_band"]["tau0"], BANDS_A[task]["final_band"]["tau1"])
gB = gains(BANDS_B[task]["learned_band"]["tau0"], BANDS_B[task]["learned_band"]["tau1"])
sA, sB = np.sqrt((gA**2).sum()), np.sqrt((gB**2).sum())
idxA, idxB = np.where(gA > 0)[0], np.where(gB > 0)[0]

def make_cluster_W(g, idx, sn, seed, rho):
    """Tranches du cluster : rayon effectif rho . g_k / sqrt(2 . sum g^2)."""
    rng = np.random.default_rng(seed + 7)
    Ws = []
    for i in idx:
        M = rng.normal(0, 1, (NC, NC)).astype(np.float32)
        r = max(np.abs(np.linalg.eigvals(M.astype(np.float64))).max(), 1e-12)
        Ws.append((M * (rho * g[i] / (np.sqrt(2) * sn) / r)).astype(np.float32))
    return np.stack(Ws)

def states_seq(u, WA, WB, WinA, WinB):
    T = u.shape[0]
    hA = np.zeros((TMAX, NC), np.float32); hB = np.zeros((TMAX, NC), np.float32)
    xA = np.zeros(NC, np.float32); xB = np.zeros(NC, np.float32)
    F = np.empty((T, 2 * NC), np.float32)
    for t in range(T):
        xA = np.tanh(u[t] @ WinA + np.einsum("kij,kj->i", WA, hA[idxA]))
        xB = np.tanh(u[t] @ WinB + np.einsum("kij,kj->i", WB, hB[idxB]))
        hA = np.concatenate([xA[None], hA[:-1]], axis=0)
        hB = np.concatenate([xB[None], hB[:-1]], axis=0)
        F[t] = np.concatenate([xA, xB])
    return F

def states_mnist(Xb, WA, WB, WinA, WinB, chunk=1000):
    out = np.empty((len(Xb), 2 * NC), np.float32)
    for lo in range(0, len(Xb), chunk):
        B = Xb[lo:lo+chunk]
        hA = np.zeros((TMAX, len(B), NC), np.float32); hB = np.zeros((TMAX, len(B), NC), np.float32)
        xA = np.zeros((len(B), NC), np.float32); xB = np.zeros((len(B), NC), np.float32)
        for t in range(28):
            xA = np.tanh(B[:, t, :] @ WinA + np.tensordot(hA[idxA], WA, axes=([0, 2], [0, 2])))
            xB = np.tanh(B[:, t, :] @ WinB + np.tensordot(hB[idxB], WB, axes=([0, 2], [0, 2])))
            hA = np.concatenate([xA[None], hA[:-1]], axis=0)
            hB = np.concatenate([xB[None], hB[:-1]], axis=0)
        out[lo:lo+len(B)] = np.concatenate([xA, xB], axis=1)
    return out

if task == "narma30":
    u, y = narma30(0)
    tr_sl, va_sl, te_sl = slice(200, 3200), slice(3200, 4200), slice(4200, 5200)
elif task in ("silverbox", "wh"):
    u, y = load_phys(task)
    tr_sl, va_sl, te_sl = slice(500, 15000), slice(15000, 20000), slice(20000, 30000)
else:
    Xtr, ytr, Xte, yte = get_mnist()

def evaluate(seed=0, rho=0.9):
    WA = make_cluster_W(gA, idxA, sA, seed, rho)
    WB = make_cluster_W(gB, idxB, sB, seed + 50, rho)
    rng = np.random.default_rng(7 + 100 * seed)
    if task != "mnist":
        WinA = rng.uniform(-0.5, 0.5, size=(u.shape[1], NC)).astype(np.float32)
        WinB = rng.uniform(-0.5, 0.5, size=(u.shape[1], NC)).astype(np.float32)
        F = states_seq(u, WA, WB, WinA, WinB)
        Fb = np.hstack([F, np.ones((len(F), 1), np.float32)]).astype(np.float64)
        yv = np.asarray(y, np.float64)
        A = Fb[tr_sl].T @ Fb[tr_sl] + 1e-8 * np.eye(Fb.shape[1])
        Wout = np.linalg.solve(A, Fb[tr_sl].T @ yv[tr_sl])
        va = float(np.sqrt(np.mean((Fb[va_sl] @ Wout - yv[va_sl]) ** 2)) / np.std(yv[te_sl]))
        te = float(np.sqrt(np.mean((Fb[te_sl] @ Wout - yv[te_sl]) ** 2)) / np.std(yv[te_sl]))
        return va, te
    WinA = rng.uniform(-0.5, 0.5, size=(28, NC)).astype(np.float32)
    WinB = rng.uniform(-0.5, 0.5, size=(28, NC)).astype(np.float32)
    Ftr = states_mnist(Xtr[:4000], WA, WB, WinA, WinB)
    Fva = states_mnist(Xtr[4000:], WA, WB, WinA, WinB)
    Fte = states_mnist(Xte, WA, WB, WinA, WinB)
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
finals = []
for seed in (0, 1, 2):
    best = None
    for rho in RADII:
        va, te = evaluate(seed=seed, rho=rho)
        sc = va if task == "mnist" else -va
        if best is None or sc > best[0]:
            best = (sc, te, rho)
    finals.append({"seed": seed, "rho": best[2], "test": best[1]})
    json.dump(results, open(RJ, "w"), indent=1)
    print(f"{task} seed{seed} rho={best[2]} test={best[1]:.4f} ({time.time()-t0:.0f}s)", flush=True)

tests = [f["test"] for f in finals]
R["bands"] = {"A": [float(BANDS_A[task]["final_band"]["tau0"]), float(BANDS_A[task]["final_band"]["tau1"])],
              "B": [float(BANDS_B[task]["learned_band"]["tau0"]), float(BANDS_B[task]["learned_band"]["tau1"])]}
R["final"] = finals
R["test_mean"] = float(np.mean(tests))
R["test_std"] = float(np.std(tests))
json.dump(results, open(RJ, "w"), indent=1)
print(f"{task} 2.8 melange 2 clusters: test={np.mean(tests):.4f} +/- {np.std(tests):.4f}")
