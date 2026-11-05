"""Partie 2.4 : bande continue apprise [tau0, tau1] (definition M. Frinzi).
Tmax = 20 couches. Gains par couche entiere k :
    g(k) = clip(k - tau0 + 1, 0, 1) * clip(tau1 - k + 1, 0, 1)
tau0 = 1   -> toutes les couches 1..tau1 pleines
tau0 = 1.5 -> 50 % de la couche 1
tau0 = 2.5 -> couche 1 nulle + 50 % de la couche 2   (idem cote tau1)
Normalisation spectrale : rayon effectif de la tranche k = rho * g_k / sqrt(sum g^2),
le budget spectral total (somme des carres) vaut rho^2 quel que soit le bandeau.
Recherche (tau0, tau1) par descente de coordonnees sur la validation (N=200, rho=0.9,
1 seed), depart (1, 10), 2 passes ; puis evaluation finale de la bande elue :
3 seeds x balayage rho, meilleur rho par seed (protocole maison).
Candidats : tau0 in {0.5, 1, 1.5, 2, 2.5, 3, 4, 5} ; tau1 in {4..10, 12, 14, 16, 20}.
Splits identiques a 2.3. Checkpoint : part2/results_band.json."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from chemins import BRUT  # noqa: E402
import numpy as np, json, time, sys, gzip, os
from pathlib import Path
from scipy.io import loadmat

OUT = (BRUT / "tensor-suite/part2")
RJ = OUT / "results_band.json"
results = json.load(open(RJ)) if RJ.exists() else {}

TMAX = 20
N = 200
TIME_BUDGET = 215
TAU0_CANDS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
TAU1_CANDS = [4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 14.0, 16.0, 20.0]

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

def make_W(seed, g, rho):
    """Tranches d'indice g>0, rayon effectif rho*g_k/sqrt(sum g^2)."""
    act = np.where(g > 0)[0]
    sn = np.sqrt((g ** 2).sum()) if (g ** 2).sum() > 0 else 1.0
    rng = np.random.default_rng(seed + 7)
    Ws, idx = [], []
    for i in act:
        M = rng.normal(0, 1, (N, N)).astype(np.float32)
        r = max(np.abs(np.linalg.eigvals(M.astype(np.float64))).max(), 1e-12)
        Ws.append((M * (rho * g[i] / sn / r)).astype(np.float32))
        idx.append(int(i))
    return np.stack(Ws), np.array(idx)

def states_seq(u, W, idx, Win):
    T = u.shape[0]
    hist = np.zeros((TMAX, N), np.float32)
    x_ = np.zeros(N, np.float32)
    F = np.empty((T, N), np.float32)
    for t in range(T):
        x_ = np.tanh(u[t] @ Win + np.einsum("kij,kj->i", W, hist[idx]))
        hist = np.concatenate([x_[None], hist[:-1]], axis=0)
        F[t] = x_
    return F

def states_mnist(Xb, W, idx, Win, chunk=1000):
    out = np.empty((len(Xb), N), np.float32)
    for lo in range(0, len(Xb), chunk):
        B = Xb[lo:lo+chunk]
        hist = np.zeros((TMAX, len(B), N), np.float32)
        x = np.zeros((len(B), N), np.float32)
        for t in range(28):
            drive = B[:, t, :] @ Win + np.tensordot(hist[idx], W, axes=([0, 2], [0, 2]))
            x = np.tanh(drive)
            hist = np.concatenate([x[None], hist[:-1]], axis=0)
        out[lo:lo+len(B)] = x
    return out

task = sys.argv[1]
R = results.setdefault(task, {"history": [], "done": False})

if task == "narma30":
    u, y = narma30(0)
    tr_sl, va_sl, te_sl = slice(200, 3200), slice(3200, 4200), slice(4200, 5200)
elif task in ("silverbox", "wh"):
    u, y = load_phys(task)
    tr_sl, va_sl, te_sl = slice(500, 15000), slice(15000, 20000), slice(20000, 30000)
else:
    Xtr, ytr, Xte, yte = get_mnist()

def evaluate(tau0, tau1, seed=0, rho=0.9):
    g = gains(tau0, tau1)
    W, idx = make_W(seed, g, rho)
    rng = np.random.default_rng(7 + 100 * seed)
    if task != "mnist":
        Win = rng.uniform(-0.5, 0.5, size=(u.shape[1], N)).astype(np.float32)
        F = states_seq(u, W, idx, Win)
        Fb = np.hstack([F, np.ones((len(F), 1), np.float32)]).astype(np.float64)
        yv = np.asarray(y, np.float64)
        A = Fb[tr_sl].T @ Fb[tr_sl] + 1e-8 * np.eye(Fb.shape[1])
        Wout = np.linalg.solve(A, Fb[tr_sl].T @ yv[tr_sl])
        va = float(np.sqrt(np.mean((Fb[va_sl] @ Wout - yv[va_sl]) ** 2)) / np.std(yv[te_sl]))
        te = float(np.sqrt(np.mean((Fb[te_sl] @ Wout - yv[te_sl]) ** 2)) / np.std(yv[te_sl]))
        return va, te
    Win = rng.uniform(-0.5, 0.5, size=(28, N)).astype(np.float32)
    Ftr = states_mnist(Xtr[:4000], W, idx, Win)
    Fva = states_mnist(Xtr[4000:], W, idx, Win)
    Fte = states_mnist(Xte, W, idx, Win)
    def pre(F):
        return np.hstack([F, np.ones((len(F), 1), np.float32)]).astype(np.float64)
    Tt = np.zeros((4000, 10)); Tt[np.arange(4000), ytr[:4000]] = 1.0
    A = pre(Ftr).T @ pre(Ftr) + 1e-6 * np.eye(pre(Ftr).shape[1])
    Wout = np.linalg.solve(A, pre(Ftr).T @ Tt)
    va = float((np.argmax(pre(Fva) @ Wout, axis=1) == ytr[4000:]).mean())
    te = float((np.argmax(pre(Fte) @ Wout, axis=1) == yte).mean())
    return va, te

t0 = time.time()
better = (lambda a, b: a > b) if task == "mnist" else (lambda a, b: a < b)

# etat de depart (1, 10) si pas encore calcule
if "start" not in R:
    va0, te0 = evaluate(1.0, 10.0)
    R["start"] = {"tau0": 1.0, "tau1": 10.0, "val": va0, "test": te0}
    json.dump(results, open(RJ, "w"), indent=1)
    print(f"{task} depart (1,10): val={va0:.4f} test={te0:.4f}", flush=True)

tau0, tau1 = R["start"]["tau0"], R["start"]["tau1"]
best_val = R["start"]["val"]

# descente de coordonnees, 2 passes
for p in range(2):
    if R.get("done") or time.time() - t0 > TIME_BUDGET:
        break
    for coord in ("tau0", "tau1"):
        cands = TAU0_CANDS if coord == "tau0" else TAU1_CANDS
        cur = tau0 if coord == "tau0" else tau1
        for c in cands:
            if coord == "tau0":
                if c >= tau1 - 0.5:
                    continue
                t0c, t1c = c, tau1
            else:
                if c <= tau0 + 0.5:
                    continue
                t0c, t1c = tau0, c
            va, _ = evaluate(t0c, t1c)
            if better(va, best_val):
                best_val = va
                cur = c
        if coord == "tau0":
            tau0 = cur
        else:
            tau1 = cur
        R["history"].append({"pass": p, "coord": coord, "tau0": tau0, "tau1": tau1,
                             "val": best_val})
        json.dump(results, open(RJ, "w"), indent=1)
        print(f"{task} pass{p} {coord} -> ({tau0}, {tau1}) val={best_val:.4f} ({time.time()-t0:.0f}s)", flush=True)

# evaluation finale : 3 seeds x balayage rho, meilleur rho par seed
RADII = [0.9, 0.99] if task == "mnist" else [0.5, 0.9, 0.99]
final = []
for seed in (0, 1, 2):
    best = None
    for rho in RADII:
        va, te = evaluate(tau0, tau1, seed=seed, rho=rho)
        sc = va if task == "mnist" else -va
        if best is None or sc > best[0]:
            best = (sc, te, rho)
    final.append({"seed": seed, "rho": best[2], "test": best[1]})
    print(f"{task} final seed{seed} rho={best[2]} test={best[1]:.4f}", flush=True)

tests = [f["test"] for f in final]
R["learned_band"] = {"tau0": tau0, "tau1": tau1, "val": best_val,
                     "final": final,
                     "test_mean": float(np.mean(tests)), "test_std": float(np.std(tests))}
R["done"] = True
json.dump(results, open(RJ, "w"), indent=1)
print(f"{task} BANDE ELUE: [{tau0}, {tau1}] test={np.mean(tests):.4f} +/- {np.std(tests):.4f}")
