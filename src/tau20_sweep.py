"""1.3.5 : etendre la bande - memes ensembles mais tau = {k0..20}, k0=1..20, budget constant
N^2*K = 400k (K=20 -> N=141 ; K=1 -> N=632). But : ou est la falaise quand la bande monte
jusqu'a 20 ? Taches : narma30 (rho=0.9, T=5200, fam. uniforme), silverbox & wh (train 20k /
wash 500 / test 10k, meilleur rho de {0.5,0.9}), mnist ligne par ligne (5k/1k, meilleur rho de
{0.9,0.99}). 3 seeds partout. Checkpoint : part2b/results_tau20.json - relancer jusqu'a COMPLETE.
Protocole corrige (14.2, 2026-10-29) : quand plusieurs rho sont proposes (silverbox, wh,
mnist), le meilleur est choisi sur un segment de VALIDATION (NVAL[task] pas avant le test ;
mnist : 1000 dernieres images du train), jamais sur le test ; readout final reajuste sur
train+validation. Les seeds stockees en dict ("test"/"per_rho") sont reprises, les anciennes
recalculees."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from chemins import BRUT  # noqa: E402
import numpy as np, json, time, sys, gzip, os
from pathlib import Path
from scipy.io import loadmat

OUT = (BRUT / "tensor-suite/part2b")
RJ = OUT / "results_tau20.json"
results = json.load(open(RJ)) if RJ.exists() else {"cells": {}}

BUDGET = 400_000
SEEDS = [0, 1, 2]
TMAX = 20
CONFIGS = {f"k{k0}": list(range(k0, TMAX + 1)) for k0 in range(1, TMAX + 1)}  # {1..20}, ..., {20}
WASH, NTEST, NTRAIN = 200, 1000, 4000  # narma
TIME_BUDGET = 215

def narma_fam(seed, T=5200):
    rng = np.random.default_rng(seed + 100)
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
    U = u[:NTRAIN + NTEST]; Y = y[:NTRAIN + NTEST] / s
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

def make_W(N, K, rho, seed):
    rng = np.random.default_rng(seed)
    W = np.empty((K, N, N), np.float32)
    for i in range(K):
        M = rng.normal(0, 1, (N, N)).astype(np.float32)
        r = max(np.abs(np.linalg.eigvals(M.astype(np.float64))).max(), 1e-12)
        W[i] = M * ((rho / np.sqrt(K)) / r)
    return W

def run_seq(u, y, taus, rho, seed, wash, ntest, nval):
    """Serie temporelle : u (T,d), y (T,). -> (NRMSE validation, NRMSE test, N) (14.2)."""
    K = len(taus)
    N = int(round((BUDGET / K) ** 0.5))
    rng = np.random.default_rng(seed)
    Win = rng.uniform(-0.5, 0.5, size=(u.shape[1], N)).astype(np.float32)
    W = make_W(N, K, rho, seed + 1)
    T = len(u); tmax = max(taus)
    hist = np.zeros((tmax, N), np.float32)
    x_ = np.zeros(N, np.float32)
    F = np.empty((T, N), np.float32)
    sel = np.array(taus) - 1
    for t in range(T):
        x_ = np.tanh(u[t] @ Win + np.einsum("kij,kj->i", W, hist[sel]))
        hist = np.concatenate([x_[None], hist[:-1]], axis=0)
        F[t] = x_
    # 14.2 : renvoie (validation, test) ; le readout du test est reajuste sur
    # train+validation, apres le choix de rho sur la validation.
    Fb = np.hstack([F, np.ones((T, 1), np.float32)]).astype(np.float64)
    yv = np.asarray(y, np.float64)
    Ftr, ytr = Fb[wash:-ntest-nval], yv[wash:-ntest-nval]
    Fva, yva = Fb[-ntest-nval:-ntest], yv[-ntest-nval:-ntest]
    Fte, yte = Fb[-ntest:], yv[-ntest:]
    A = Ftr.T @ Ftr + 1e-8 * np.eye(Ftr.shape[1])
    Wout = np.linalg.solve(A, Ftr.T @ ytr)
    val = float(np.sqrt(np.mean((Fva @ Wout - yva) ** 2)) / np.std(yva))
    Ftv, ytv = Fb[wash:-ntest], yv[wash:-ntest]
    A2 = Ftv.T @ Ftv + 1e-8 * np.eye(Ftv.shape[1])
    Wout2 = np.linalg.solve(A2, Ftv.T @ ytv)
    return val, float(np.sqrt(np.mean((Fte @ Wout2 - yte) ** 2)) / np.std(yte)), N

def states_mnist(Xb, N, taus, rho, seed, chunk=1000):
    K = len(taus)
    rng = np.random.default_rng(seed)
    Win = rng.uniform(-0.5, 0.5, size=(28, N)).astype(np.float32)
    W = make_W(N, K, rho, seed + 1)
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
RADII = {"narma30": [0.9], "silverbox": [0.5, 0.9], "wh": [0.5, 0.9], "mnist": [0.9, 0.99]}
NVAL = {"narma30": 1000, "silverbox": 2000, "wh": 2000}  # validation avant le test (14.2)
NVAL_MNIST = 1000  # dernieres images du train : le rho y est choisi (14.2)

if task == "narma30":
    data = None
elif task in ("silverbox", "wh"):
    WASH, NTEST = 500, 10000
    NTRAIN = 20000
    u, y = load_phys(task)
    data = (u, y)
elif task == "mnist":
    Xtr, ytr, Xte, yte = get_mnist()
    data = (Xtr, ytr, Xte, yte)
else:
    raise SystemExit("tache inconnue " + task)

t0 = time.time()
stop = False
for name, taus in CONFIGS.items():
    if stop: break
    key = f"{task}_{name}"
    cell = results["cells"].setdefault(key, {"task": task, "config": name, "taus": taus, "seeds": {}})
    for s in SEEDS:
        if str(s) in cell["seeds"] and isinstance(cell["seeds"][str(s)], dict):
            continue  # deja calculee sous le protocole corrige (14.2)
        if time.time() - t0 > TIME_BUDGET:
            stop = True; break
        per_rho = {}
        if task in ("narma30", "silverbox", "wh"):
            if task == "narma30":
                u, y = narma_fam(s)
            best_val, best, N = np.inf, np.inf, None
            for rho in RADII[task]:
                va, te, N = run_seq(u, y, taus, rho, s, WASH, NTEST, NVAL[task])
                per_rho[str(rho)] = {"val": va, "test": te}
                if va < best_val:
                    best_val, best = va, te
        else:  # mnist
            K = len(taus)
            N = int(round((BUDGET / K) ** 0.5))
            best_val, best = -1.0, 0.0
            for rho in RADII[task]:
                Ftr = states_mnist(Xtr, N, taus, rho, s)
                Fte = states_mnist(Xte, N, taus, rho, s)
                # 14.2 : rho choisi sur les NVAL_MNIST dernieres images du train
                va = classify_acc(Ftr[:-NVAL_MNIST], ytr[:-NVAL_MNIST],
                                 Ftr[-NVAL_MNIST:], ytr[-NVAL_MNIST:])
                per_rho[str(rho)] = {"val": va}
                if va > best_val:
                    best_val = va
                    best = classify_acc(Ftr, ytr, Fte, yte)
        cell["seeds"][str(s)] = {"test": best, "per_rho": per_rho}
        cell["N"] = N
        print(f"{task} {name}(K={len(taus)}) seed={s} score={best:.4f} ({time.time()-t0:.0f}s)", flush=True)
        results["cells"][key] = cell
        json.dump(results, open(RJ, "w"), indent=1)

remaining = sum(1 for t2 in ("narma30", "silverbox", "wh", "mnist") for name in CONFIGS
               if any(str(s) not in results["cells"].get(f"{t2}_{name}", {"seeds": {}})["seeds"]
                      or not isinstance(results["cells"].get(f"{t2}_{name}", {"seeds": {}})["seeds"][str(s)], dict)
                      for s in SEEDS))
print(f"restants : {remaining}")
if remaining == 0:
    print("COMPLETE")
    agg = {}
    for key, c in results["cells"].items():
        v = np.array([sv["test"] if isinstance(sv, dict) else sv for sv in c["seeds"].values()])
        agg[key] = dict(task=c["task"], config=c["config"], taus=c["taus"], K=len(c["taus"]),
                       N=c["N"], mean=float(v.mean()), std=float(v.std()), n=len(v))
    results["agg"] = agg
    json.dump(results, open(RJ, "w"), indent=1)
    print("agregation ecrite")
