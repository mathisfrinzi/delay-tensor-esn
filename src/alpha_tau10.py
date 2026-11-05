"""Partie 2.3 : alpha_tau appris sur les 10 tranches de la baseline (tau=1..10, N=200, rho=0.9).
x(t) = tanh(Win u(t) + somme_k alpha_k W_k x(t-k)) : descente de coordonnees sur la NRMSE de
validation, candidats multiplicatifs (0, 0.5, 2.0), 2 passes (1 pour mnist, plus lourd).
alpha uniforme (=1) vs appris, en test. Splits validation : narma train [200..3200) val [3200..4200)
test [4200..5200) ; phys train [500..15000) val [15000..20000) test [20000..30000) ; mnist 4000/1000
(val) / 1000 (test). Checkpoint : part2/results_alpha10.json."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from chemins import BRUT  # noqa: E402
import numpy as np, json, time, sys, gzip, os
from pathlib import Path
from scipy.io import loadmat

OUT = (BRUT / "tensor-suite/part2")
RJ = OUT / "results_alpha10.json"
results = json.load(open(RJ)) if RJ.exists() else {}

TAUS = list(range(1, 11))
N, RHO = 200, 0.9
TIME_BUDGET = 215

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
    return imgs(files[0])[:5000], labs(files[1])[:5000], imgs(files[2])[:1000], labs(files[3])[:1000]

def make_fixed_W(seed):
    """Tranches fixes (rho/sqrt(10) chacune), tirees une fois pour toute la recherche."""
    rng = np.random.default_rng(seed + 7)
    W = []
    for _ in TAUS:
        M = rng.normal(0, 1, (N, N)).astype(np.float32)
        r = max(np.abs(np.linalg.eigvals(M.astype(np.float64))).max(), 1e-12)
        W.append((M * (RHO / np.sqrt(len(TAUS)) / r)).astype(np.float32))
    return np.stack(W)

def states_seq_alpha(u, W, Win, alphas):
    T, d = u.shape
    tmax = max(TAUS)
    hist = np.zeros((tmax, N), np.float32)
    x_ = np.zeros(N, np.float32)
    F = np.empty((T, N), np.float32)
    Wa = W * (alphas[:, None, None]).astype(np.float32)
    sel = np.array(TAUS) - 1
    for t in range(T):
        x_ = np.tanh(u[t] @ Win + np.einsum("kij,kj->i", Wa, hist[sel]))
        hist = np.concatenate([x_[None], hist[:-1]], axis=0)
        F[t] = x_
    return F

def states_mnist_alpha(Xb, W, Win, alphas, chunk=1000):
    tmax = max(TAUS)
    sel = np.array(TAUS) - 1
    Wa = W * (alphas[:, None, None]).astype(np.float32)
    out = np.empty((len(Xb), N), np.float32)
    for lo in range(0, len(Xb), chunk):
        B = Xb[lo:lo+chunk]
        x = np.zeros((len(B), N), np.float32)
        hist = np.zeros((tmax, len(B), N), np.float32)
        for t in range(28):
            drive = B[:, t, :] @ Win + np.tensordot(hist[sel], Wa, axes=([0, 2], [0, 2]))
            x = np.tanh(drive)
            hist = np.concatenate([x[None], hist[:-1]], axis=0)
        out[lo:lo+len(B)] = x
    return out

def split_score_reg(F, y, tr_sl, va_sl, te_sl):
    Fb = np.hstack([F, np.ones((len(F), 1), np.float32)]).astype(np.float64)
    yv = np.asarray(y, np.float64)
    A = Fb[tr_sl].T @ Fb[tr_sl] + 1e-8 * np.eye(Fb.shape[1])
    Wout = np.linalg.solve(A, Fb[tr_sl].T @ yv[tr_sl])
    va = float(np.sqrt(np.mean((Fb[va_sl] @ Wout - yv[va_sl]) ** 2)) / np.std(yv[te_sl]))
    te = float(np.sqrt(np.mean((Fb[te_sl] @ Wout - yv[te_sl]) ** 2)) / np.std(yv[te_sl]))
    return va, te

def acc_cls(Ftr, ytr, Fva, yva, Fte, yte, ridge=1e-6):
    def pre(F):
        return np.hstack([F, np.ones((len(F), 1), np.float32)]).astype(np.float64)
    Tt = np.zeros((len(ytr), 10)); Tt[np.arange(len(ytr)), ytr] = 1.0
    A = pre(Ftr).T @ pre(Ftr) + ridge * np.eye(pre(Ftr).shape[1])
    Wout = np.linalg.solve(A, pre(Ftr).T @ Tt)
    return (float((np.argmax(pre(Fva) @ Wout, axis=1) == yva).mean()),
            float((np.argmax(pre(Fte) @ Wout, axis=1) == yte).mean()))

task = sys.argv[1]
R = results.setdefault(task, {"history": [], "done": False})
PASSES = 1 if task == "mnist" else 2
CANDS = (0.0, 0.5, 2.0)

if task == "narma30":
    u, y = narma30(0)
    tr_sl, va_sl, te_sl = slice(200, 3200), slice(3200, 4200), slice(4200, 5200)
elif task in ("silverbox", "wh"):
    u, y = load_phys(task)
    tr_sl, va_sl, te_sl = slice(500, 15000), slice(15000, 20000), slice(20000, 30000)
else:
    Xtr, ytr, Xte, yte = get_mnist()

W = make_fixed_W(0)
rng = np.random.default_rng(7)
if task != "mnist":
    Win = rng.uniform(-0.5, 0.5, size=(u.shape[1], N)).astype(np.float32)
else:
    Win = rng.uniform(-0.5, 0.5, size=(28, N)).astype(np.float32)

def evaluate(alphas):
    if task == "mnist":
        Ftr = states_mnist_alpha(Xtr[:4000], W, Win, alphas)
        Fva = states_mnist_alpha(Xtr[4000:], W, Win, alphas)
        Fte = states_mnist_alpha(Xte, W, Win, alphas)
        return acc_cls(Ftr, ytr[:4000], Fva, ytr[4000:], Fte, yte)
    F = states_seq_alpha(u, W, Win, alphas)
    return split_score_reg(F, y, tr_sl, va_sl, te_sl)

t0 = time.time()
alpha = np.ones(len(TAUS))
if "uniform" not in R:
    va0, te0 = evaluate(alpha)
    R["uniform"] = {"val": va0, "test": te0}
    json.dump(results, open(RJ, "w"), indent=1)
    print(f"{task} uniform: val={va0:.4f} test={te0:.4f}", flush=True)

best_val = R["uniform"]["val"]
for p in range(PASSES):
    if time.time() - t0 > TIME_BUDGET or R.get("done"):
        break
    for i in range(len(TAUS)):
        cur = alpha[i]
        for c in CANDS:
            a2 = cur * c
            if a2 < 0:
                continue
            alpha[i] = a2
            va, _ = evaluate(alpha)
            better = va > best_val if task == "mnist" else va < best_val
            if better:
                best_val = va; cur = a2
        alpha[i] = cur
        R["history"].append({"pass": p, "coord": TAUS[i], "alpha_after": [round(float(x), 4) for x in alpha]})
        json.dump(results, open(RJ, "w"), indent=1)
        print(f"{task} pass{p} tau={TAUS[i]} alpha={np.round(alpha, 2).tolist()} ({time.time()-t0:.0f}s)", flush=True)

va1, te1 = evaluate(alpha)
R["learned"] = {"val": va1, "test": te1, "alphas": [round(float(x), 4) for x in alpha]}
R["done"] = True
json.dump(results, open(RJ, "w"), indent=1)
print(f"{task} appris: test={te1:.4f} alphas={np.round(alpha, 2).tolist()}")
