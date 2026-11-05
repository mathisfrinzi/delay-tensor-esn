"""Partie 2.6 : sparsite differenciee par tranche : densite p(tau) APPRISE.
Chaque tranche W_tau est masquee a la densite p_tau (masques IMBRIQUES :
U_ij < p, donc p=1 inclut p=0.5 inclut p=0.2 : meme support, connexions
ajoutees progressivement), puis renormalisee au rayon rho/sqrt(10).
Descente de coordonnees sur p_tau (candidats {0.2, 0.5, 1.0}, depart p=1 =
baseline dense exacte), 2 passes, 1 seed rho=0.9, selection validation.
Puis evaluation standard 3 seeds x balayage rho du profil appris, avec
references uniformes p=0.2 et p=0.5 (campagne 1 : p uniforme ~neutre).
Splits identiques a 2.3-2.5. Checkpoint : results_p_tau.json."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from chemins import BRUT  # noqa: E402
import numpy as np, json, time, sys, gzip, os
from pathlib import Path
from scipy.io import loadmat

OUT = (BRUT / "tensor-suite/part2")
RJ = OUT / "results_p_tau.json"
results = json.load(open(RJ)) if RJ.exists() else {}

K = 10
N = 200
CANDS = [0.2, 0.5, 1.0]

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
R = results.setdefault(task, {"history": [], "cache": {}, "done": False})

if task == "narma30":
    u, y = narma30(0)
    tr_sl, va_sl, te_sl = slice(200, 3200), slice(3200, 4200), slice(4200, 5200)
elif task in ("silverbox", "wh"):
    u, y = load_phys(task)
    tr_sl, va_sl, te_sl = slice(500, 15000), slice(15000, 20000), slice(20000, 30000)
else:
    Xtr, ytr, Xte, yte = get_mnist()

def make_W(seed, ps, rho):
    """Tranches masquees a densite p_k (masques imbriques), rayon rho/sqrt(K)."""
    rng = np.random.default_rng(seed + 7)
    W = np.empty((K, N, N), np.float32)
    for i in range(K):
        M = rng.normal(0, 1, (N, N)).astype(np.float32)
        U = rng.random((N, N)).astype(np.float32)
        Mk = M * (U < ps[i])
        r = max(np.abs(np.linalg.eigvals(Mk.astype(np.float64))).max(), 1e-12)
        W[i] = Mk * ((rho / np.sqrt(K)) / r)
    return W

def evaluate(ps, seed=0, rho=0.9, want="va_te"):
    key = "|".join(str(p) for p in ps) + f"|{seed}|{rho}|{want}"
    if key in R["cache"]:
        return tuple(R["cache"][key])
    W = make_W(seed, ps, rho)
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
        out = (va, te)
    else:
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
        out = (va, te)
    R["cache"][key] = out
    return out

def save():
    if len(R["cache"]) > 400:
        R["cache"] = dict(list(R["cache"].items())[-200:])
    json.dump(results, open(RJ, "w"), indent=1)

t0 = time.time()
TIME_BUDGET = 240

# recherche : descente de coordonnees sur p (depart dense = baseline)
ps = R.get("cur", [1.0] * K)
best_va = evaluate(ps)[0] if "start_val" not in R else R["start_val"]
if "start_val" not in R:
    R["start_val"] = best_va
    save()
    print(f"{task} depart dense p=1: val={best_va:.4f}", flush=True)

for p_i in range(R.get("passes_done", 0), 2):
    if R.get("done") or time.time() - t0 > TIME_BUDGET:
        break
    for k in range(K):
        cur = ps[k]
        for c in CANDS:
            if c == cur:
                continue
            trial = list(ps); trial[k] = c
            va = evaluate(trial)[0]
            if (MAXIM and va > best_va) or (not MAXIM and va < best_va):
                best_va, cur = va, c
        ps[k] = cur
        R["cur"] = ps
        R["history"].append({"pass": p_i, "k": k, "p": ps})
        save()
        print(f"{task} pass{p_i} k={k+1} p={ps} ({time.time()-t0:.0f}s)", flush=True)
    R["passes_done"] = p_i + 1
    save()

# final : profil appris + references uniformes, 3 seeds x rho
if R.get("passes_done", 0) >= 2 or R.get("done"):
    RADII = [0.9, 0.99] if task == "mnist" else [0.5, 0.9, 0.99]
    def final_eval(psx, tag):
        finals = []
        for seed in (0, 1, 2):
            best = None
            for rho in RADII:
                va, te = evaluate(psx, seed=seed, rho=rho)
                sc = va if task == "mnist" else -va
                if best is None or sc > best[0]:
                    best = (sc, te, rho)
            finals.append({"seed": seed, "rho": best[2], "test": best[1]})
        tests = [f["test"] for f in finals]
        R[tag] = {"p": psx, "final": finals,
                  "test_mean": float(np.mean(tests)), "test_std": float(np.std(tests)),
                  "n_conn": int(sum(psx) * N * N)}
        save()
        print(f"{task} {tag}: p={psx} test={np.mean(tests):.4f} +/- {np.std(tests):.4f}", flush=True)
    final_eval(list(ps), "learned")
    if "uni02" not in R:
        final_eval([0.2] * K, "uni02")
    if "uni05" not in R:
        final_eval([0.5] * K, "uni05")
    R["done"] = True
    save()
    print(f"{task} 2.6 p(tau) terminé : appris={ps}")
else:
    print(f"{task} budget epuise, passes={R.get('passes_done', 0)}/2 : RELANCER")
