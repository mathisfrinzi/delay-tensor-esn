"""Partie 2.2 : pruning des tranches. Elimination gloutonne (backward) avec redistribution du
budget : on part de {1..10} N=200 ; a chaque pas on retire la tranche dont le retrait donne la
meilleille NRMSE de validation, puis N est recalcule pour garder 400k connexions (N=sqrt(400k/K)).
-> courbe perf(K) par tache + l'ordre d'elimination + la bande retenue. Selection : 1 seed,
rho=0.9. Eval finale de la config elue : protocole standard 3 seeds (meilleur rho).
Checkpoint : part2/results_prune.json - relancer jusqu'a FINAL_DONE."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from chemins import BRUT  # noqa: E402
import numpy as np, json, time, sys, gzip, os
from pathlib import Path
from scipy.io import loadmat

OUT = (BRUT / "tensor-suite/part2")
RJ = OUT / "results_prune.json"
results = json.load(open(RJ)) if RJ.exists() else {}
BUDGET = 400_000
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

def states_seq(u, N, taus, rho, seed):
    rng = np.random.default_rng(seed + 7)
    T, d = u.shape
    Win = rng.uniform(-0.5, 0.5, size=(d, N)).astype(np.float32)
    W = np.empty((len(taus), N, N), np.float32)
    for i in range(len(taus)):
        M = rng.normal(0, 1, (N, N)).astype(np.float32)
        r = max(np.abs(np.linalg.eigvals(M.astype(np.float64))).max(), 1e-12)
        W[i] = M * ((rho / np.sqrt(len(taus))) / r)
    tmax = max(taus)
    hist = np.zeros((tmax, N), np.float32)
    x_ = np.zeros(N, np.float32)
    F = np.empty((T, N), np.float32)
    sel = np.array(taus) - 1
    for t in range(T):
        x_ = np.tanh(u[t] @ Win + np.einsum("kij,kj->i", W, hist[sel]))
        hist = np.concatenate([x_[None], hist[:-1]], axis=0)
        F[t] = x_
    return F

def states_mnist(Xb, N, taus, rho, seed, chunk=1000):
    rng = np.random.default_rng(seed)
    Win = rng.uniform(-0.5, 0.5, size=(28, N)).astype(np.float32)
    W = np.empty((len(taus), N, N), np.float32)
    for i in range(len(taus)):
        M = rng.normal(0, 1, (N, N)).astype(np.float32)
        r = max(np.abs(np.linalg.eigvals(M.astype(np.float64))).max(), 1e-12)
        W[i] = M * ((rho / np.sqrt(len(taus))) / r)
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

def score_reg(F, y, tr_sl, va_sl, te_sl=None):
    Fb = np.hstack([F, np.ones((len(F), 1), np.float32)]).astype(np.float64)
    yv = np.asarray(y, np.float64)
    A = Fb[tr_sl].T @ Fb[tr_sl] + 1e-8 * np.eye(Fb.shape[1])
    Wout = np.linalg.solve(A, Fb[tr_sl].T @ yv[tr_sl])
    va = float(np.sqrt(np.mean((Fb[va_sl] @ Wout - yv[va_sl]) ** 2)) / np.std(yv[te_sl if te_sl else va_sl]))
    return va

def acc_cls(Ftr, ytr, Fva, yva, ridge=1e-6):
    Ftr = np.hstack([Ftr, np.ones((len(Ftr), 1), np.float32)]).astype(np.float64)
    Fva = np.hstack([Fva, np.ones((len(Fva), 1), np.float32)]).astype(np.float64)
    Tt = np.zeros((len(ytr), 10)); Tt[np.arange(len(ytr)), ytr] = 1.0
    A = Ftr.T @ Ftr + ridge * np.eye(Ftr.shape[1])
    Wout = np.linalg.solve(A, Ftr.T @ Tt)
    return float((np.argmax(Fva @ Wout, axis=1) == yva).mean())

task = sys.argv[1]
R = results.setdefault(task, {"path": [], "kept": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                               "step_scores": {}, "done": False, "final": {}})

if task == "narma30":
    u, y = narma30(0)
    tr_sl, va_sl, te_sl = slice(200, 3200), slice(3200, 4200), slice(4200, 5200)
elif task in ("silverbox", "wh"):
    u, y = load_phys(task)
    tr_sl, va_sl, te_sl = slice(500, 15000), slice(15000, 20000), slice(20000, 30000)
elif task == "mnist":
    Xtr, ytr, Xte, yte = get_mnist()
else:
    raise SystemExit("tache inconnue")

t0 = time.time()
while len(R["kept"]) > 1 and time.time() - t0 < TIME_BUDGET:
    kept = R["kept"]
    K = len(kept)
    for k in kept:
        ck = f"K{K}_rem{k}"
        if ck in R["step_scores"]:
            continue
        if time.time() - t0 > TIME_BUDGET:
            break
        cand = [x for x in kept if x != k]
        N = int(round((BUDGET / len(cand)) ** 0.5))
        if task == "mnist":
            Ftr = states_mnist(Xtr[:4000], N, cand, 0.9, 0)
            Fva = states_mnist(Xtr[4000:], N, cand, 0.9, 0)
            sc = acc_cls(Ftr, ytr[:4000], Fva, ytr[4000:])
        else:
            F = states_seq(u, N, cand, 0.9, 0)
            sc = score_reg(F, y, tr_sl, va_sl, te_sl)
        R["step_scores"][ck] = sc
        print(f"{task} K={K} retire?{k} val={sc:.4f} N={N} ({time.time()-t0:.0f}s)", flush=True)
        json.dump(results, open(RJ, "w"), indent=1)
    if f"K{K}_rem{kept[0]}" not in R["step_scores"]:
        break  # budget epuise avant de finir l'etape
    # choisir la meilleure suppression (min val pour reg, max acc pour mnist)
    scores = {k: R["step_scores"][f"K{K}_rem{k}"] for k in kept}
    best_k = min(scores, key=scores.get) if task != "mnist" else max(scores, key=scores.get)
    R["path"].append({"K": K, "removed": best_k, "val": scores[best_k], "kept_after": [x for x in kept if x != best_k]})
    R["kept"] = [x for x in kept if x != best_k]
    print(f"{task} ==> retire {best_k} (val={scores[best_k]:.4f}), reste {R['kept']}", flush=True)
    json.dump(results, open(RJ, "w"), indent=1)

if len(R["kept"]) == 1:
    R["done"] = True
    # meilleur K sur le chemin (val de la config K tranches)
    if task == "mnist":
        best_step = max(R["path"], key=lambda s: s["val"])
    else:
        best_step = min(R["path"], key=lambda s: s["val"])
    R["best"] = {"K": best_step["K"] - 0, "kept": best_step["kept_after"], "val": best_step["val"]}
    print("GREEDY_DONE", R["best"])
    json.dump(results, open(RJ, "w"), indent=1)

all_done = all(results.get(t, {}).get("done") for t in ("narma30", "silverbox", "wh", "mnist"))
print("ALL_DONE" if all_done else "pas encore")
