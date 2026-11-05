"""Partie 1, nouvelles taches (ex-1.4, devenue 1.3) : la batterie des taches predites par la loi.
Taches : parite retardee (D=10,20,50, L=6) ; Santa Fe laser (set A reel, 1000 pts) ; chaine de
Markov a etats longs (switching, filtrage) ; robustesse au bruit d'entree (NARMA-30 + bruit) ;
geometrie du parcours MNIST (lignes / colonnes / spirale).
Configs Partie 1 a budget egal (5) via suite_part1. Checkpoint : part1/results_newtasks.json.
Usage : python suite_newtasks.py <parite10|parite20|parite50|santafe|markov|bruit|mnist>
Protocole corrige (14.2, 2026-10-29) : le meilleur rho par seed est choisi sur un segment de
VALIDATION (nval pas avant le test ; MNIST : NVAL_MNIST dernieres images du train), jamais
sur le test ; readout final reajuste sur train+validation. Les entrees marquees
proto="rho_sur_validation" sont reprises, les anciennes recalculees."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from chemins import BRUT  # noqa: E402
import numpy as np, json, sys, gzip, os, time
from pathlib import Path

from suite_part1 import states_all, CONFIGS, nrmse_score

OUT = (BRUT / "tensor-suite/part1")
RJ = OUT / "results_newtasks.json"
results = json.load(open(RJ, encoding="utf-8-sig")) if RJ.exists() else {}
RADII = [0.5, 0.9, 0.99]
SEEDS = [0, 1, 2]
WASH, NTEST = 200, 2000
NVAL = 1000  # segment de validation avant le test : le rho y est choisi (14.2)
NVAL_MNIST = 1000  # dernieres images du train : le rho y est choisi (14.2)
T = 5200
TIME_BUDGET = float(__import__("os").environ.get("TEMPS_MAX", "215"))

def nrmse_valtest(F, y, wash, ntest, nval):
    """14.2 : (NRMSE validation, NRMSE test). Readout du test reajuste sur train+validation."""
    Fb = np.hstack([F, np.ones((len(F), 1), np.float32)]).astype(np.float64)
    y = np.asarray(y, np.float64)
    Ftr, ytr = Fb[wash:-ntest-nval], y[wash:-ntest-nval]
    Fva, yva = Fb[-ntest-nval:-ntest], y[-ntest-nval:-ntest]
    Fte, yte = Fb[-ntest:], y[-ntest:]
    A = Ftr.T @ Ftr + 1e-8 * np.eye(Ftr.shape[1])
    Wout = np.linalg.solve(A, Ftr.T @ ytr)
    val = float(np.sqrt(np.mean((Fva @ Wout - yva) ** 2)) / np.std(yva))
    Ftv, ytv = Fb[wash:-ntest], y[wash:-ntest]
    A2 = Ftv.T @ Ftv + 1e-8 * np.eye(Ftv.shape[1])
    Wout2 = np.linalg.solve(A2, Ftv.T @ ytv)
    return val, float(np.sqrt(np.mean((Fte @ Wout2 - yte) ** 2)) / np.std(yte))

def classify_valtest(Ftr, ytr, Fte, yte, nval, ridge=1e-6):
    """14.2 : (acc validation, acc test). Validation = nval dernieres images du train ;
    le readout du test est reajuste sur tout le train."""
    Ftrb = np.hstack([Ftr, np.ones((len(Ftr), 1), np.float32)]).astype(np.float64)
    nfit = len(ytr) - nval
    Tt = np.zeros((nfit, 10)); Tt[np.arange(nfit), ytr[:nfit]] = 1.0
    A = Ftrb[:nfit].T @ Ftrb[:nfit] + ridge * np.eye(Ftrb.shape[1])
    Wout = np.linalg.solve(A, Ftrb[:nfit].T @ Tt)
    va = float((np.argmax(Ftrb[nfit:] @ Wout, axis=1) == ytr[nfit:]).mean())
    return va, classify_acc(Ftr, ytr, Fte, yte, ridge=ridge)

def run_task(key, gen, ntest=NTEST, wash=WASH, radii=RADII, seeds=SEEDS, metric="nrmse", nval=NVAL):
    """gen(seed) -> (u, y). Metriques : nrmse (regression) ou acc (classification)."""
    res = results.get(key, {})
    t0 = time.time()
    stop = False
    for cid, name, N, taus in CONFIGS:
        if (cid in res and res[cid].get("proto") == "rho_sur_validation") or stop:
            continue
        bests = []
        for seed in seeds:
            if stop:
                break
            u, y = gen(seed)
            best_val, best = np.inf, np.inf
            for rho in radii:
                F = states_all(u, N, rho, taus, seed)
                va, te = nrmse_valtest(F, y, wash, ntest, nval)
                if va < best_val:
                    best_val, best = va, te
            bests.append(best)
            if time.time() - t0 > TIME_BUDGET:
                stop = True
        if bests and len(bests) == len(seeds):
            res[cid] = {"name": name, "proto": "rho_sur_validation",
                        "mean": float(np.mean(bests)), "std": float(np.std(bests)),
                        "per_seed": [float(b) for b in bests], "metric": metric}
            print(f"  {key}/{name:35s} {res[cid]['mean']:.4f} +/- {res[cid]['std']:.4f} ({time.time()-t0:.0f}s)", flush=True)
            results[key] = res
            json.dump(results, open(RJ, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    if not stop and len(res) == len(CONFIGS):
        print(f"[{key}] COMPLETE")
    elif stop:
        print(f"[{key}] TIME_STOP")

# generateurs
def parity_gen(D, L=6, T=T):
    """y(t) = parite (XOR) de la fenetre u(t-D-L+1 .. t-D). u binaire i.i.d."""
    def gen(seed):
        rng = np.random.default_rng(seed)
        u = rng.integers(0, 2, size=T).astype(np.float32)
        y = np.zeros(T, np.float32)
        cs = np.concatenate([[0.0], np.cumsum(u)])  # parite = somme mod 2 de la fenetre finissant a t-D
        for t in range(D + L - 1, T):
            y[t] = (cs[t - D + 1] - cs[t - D + 1 - L]) % 2
        uu = np.stack([u, np.ones_like(u)], 1).astype(np.float32)
        return uu, y
    return gen

def santafe_gen():
    x = np.loadtxt("/tmp/datasets/santafe/SF_A.dat", delimiter=",")
    assert len(x) == 1000
    x = x / 255.0
    def gen(seed):
        uu = np.stack([x[:-1], np.ones(len(x) - 1)], 1).astype(np.float32)
        return uu, x[1:].astype(np.float32)
    return gen

def markov_gen(dwell=20, noise=0.5, T=T):
    """Etat cache s(t) = +-1, dwell moyen 20 pas ; u = s + bruit ; cible y = s (filtrage)."""
    def gen(seed):
        rng = np.random.default_rng(seed)
        p = 1.0 / dwell
        s = np.empty(T, np.float32)
        s[0] = rng.integers(0, 2) * 2 - 1
        for t in range(1, T):
            if rng.random() < p:
                s[t] = -s[t - 1]
            else:
                s[t] = s[t - 1]
        u = s + noise * rng.standard_normal(T).astype(np.float32)
        uu = np.stack([u, np.ones_like(u)], 1).astype(np.float32)
        return uu, s
    return gen

def bruit_gen(sigma, T=T):
    """NARMA-30 (famille 0.2/0.004/1.5/0.001), entree bruitee u' = u + sigma*eta."""
    def gen(seed):
        rng = np.random.default_rng(seed)
        u = rng.uniform(0, 0.5, size=T)
        y = np.zeros(T, np.float32)
        a, b, g, d = 0.2, 0.004, 1.5, 0.001
        n = 30
        for t in range(n, T):
            y[t] = a * y[t-1] + b * y[t-1] * y[t-n:t].sum() + g * u[t-n] * u[t-1] + d
        u2 = u + sigma * rng.standard_normal(T)
        uu = np.stack([u2, np.ones_like(u2)], 1).astype(np.float32)
        return uu, y
    return gen

# MNIST geometrie
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
        return d.reshape(-1, 784).astype(np.float32) / 255.0
    def labs(fn):
        with gzip.open(os.path.join(BASE, fn), "rb") as f:
            return np.frombuffer(f.read(), np.uint8, offset=8).astype(np.int64)
    return imgs(files[0])[:n_train], labs(files[1])[:n_train], imgs(files[2])[:n_test], labs(files[3])[:n_test]

SPIRAL = None
def spiral_index():
    """Ordre de lecture en spirale entree par le coin haut-gauche (784 -> permutation)."""
    global SPIRAL
    if SPIRAL is None:
        order, seen = [], np.zeros((28, 28), bool)
        top, bottom, left, right = 0, 27, 0, 27
        while top <= bottom and left <= right:
            for j in range(left, right + 1): order.append((top, j))
            for i in range(top + 1, bottom + 1): order.append((i, right))
            if top < bottom and left < right:
                for j in range(right - 1, left - 1, -1): order.append((bottom, j))
                for i in range(bottom - 1, top, -1): order.append((i, left))
            top += 1; bottom -= 1; left += 1; right -= 1
        SPIRAL = np.array([i * 28 + j for i, j in order])
    return SPIRAL

def states_px(Xb, N, rho, taus, seed, chunk=1000):
    """Xb: (B, 784) a plat -> (B, N). 28 pas de 28 pixels consecutifs (geometrie = ordre des pixels)."""
    rng = np.random.default_rng(seed)
    Win = rng.uniform(-0.5, 0.5, size=(28, N)).astype(np.float32)
    K = 1 if taus is None else len(taus)
    W = np.empty((K, N, N), np.float32)
    for i in range(K):
        M = rng.normal(0, 1, (N, N)).astype(np.float32)
        r = max(np.abs(np.linalg.eigvals(M.astype(np.float64))).max(), 1e-12)
        W[i] = M * ((rho / (np.sqrt(K) if K > 1 else 1)) / r)
    tmax = 1 if taus is None else max(taus)
    sel = np.arange(K) if taus is None else np.array(taus) - 1
    out = np.empty((len(Xb), N), np.float32)
    for lo in range(0, len(Xb), chunk):
        B = Xb[lo:lo+chunk].reshape(len(Xb[lo:lo+chunk]), 28, 28)
        x = np.zeros((len(B), N), np.float32)
        hist = np.zeros((tmax, len(B), N), np.float32)
        for t in range(28):
            drive = B[:, t, :] @ Win
            if taus is None:
                drive = drive + hist[0] @ W[0]
            else:
                drive = drive + np.tensordot(hist[sel], W, axes=([0, 2], [0, 2]))
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

def run_mnist_geom():
    Xtr, ytr, Xte, yte = get_mnist()
    geoms = {
        "lignes":   Xtr.copy(),
        "colonnes": Xtr.reshape(-1, 28, 28).transpose(0, 2, 1).reshape(-1, 784),
        "spirale":  Xtr[:, spiral_index()],
    }
    for gname, Xg in geoms.items():
        key = f"mnist_{gname}"
        if key in results and len(results[key]) == len(CONFIGS) and all(
                isinstance(results[key].get(cid), dict)
                and results[key][cid].get("proto") == "rho_sur_validation"
                for cid, *_ in CONFIGS):
            print(f"[{key}] skip"); continue
        res = results.get(key, {})
        t0 = time.time(); stop = False
        for cid, name, N, taus in CONFIGS:
            if (cid in res and res[cid].get("proto") == "rho_sur_validation") or stop: continue
            accs = []
            for seed in [0, 1, 2]:
                if time.time() - t0 > TIME_BUDGET: stop = True; break
                best_val, best = -1.0, 0.0
                for rho in (0.9, 0.99):
                    Ftr = states_px(Xg, N, rho, taus, seed)
                    Fte = states_px(Xte if gname == "lignes" else
                                    (Xte.reshape(-1, 28, 28).transpose(0, 2, 1).reshape(-1, 784) if gname == "colonnes"
                                     else Xte[:, spiral_index()]),
                                    N, rho, taus, seed)
                    va, te = classify_valtest(Ftr, ytr, Fte, yte, NVAL_MNIST)
                    if va > best_val:
                        best_val, best = va, te
                accs.append(best)
            if accs:
                res[cid] = {"name": name, "proto": "rho_sur_validation",
                            "mean": float(np.mean(accs)), "std": float(np.std(accs)),
                            "per_seed": [float(a) for a in accs], "metric": "acc"}
                print(f"  {key}/{name:35s} {res[cid]['mean']*100:.2f}% +/- {res[cid]['std']*100:.2f} ({time.time()-t0:.0f}s)", flush=True)
                results[key] = res
                json.dump(results, open(RJ, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
        if not stop and len(res) == len(CONFIGS):
            print(f"[{key}] COMPLETE")
        elif stop:
            print(f"[{key}] TIME_STOP")

if __name__ == "__main__":
    which = sys.argv[1]
    if which == "parite10":
        run_task("parite_D10", parity_gen(10))
    elif which == "parite20":
        run_task("parite_D20", parity_gen(20))
    elif which == "parite50":
        run_task("parite_D50", parity_gen(50))
    elif which == "parite2_10":
        run_task("parite2_D10", parity_gen(10, L=2))
    elif which == "parite2_20":
        run_task("parite2_D20", parity_gen(20, L=2))
    elif which == "parite2_50":
        run_task("parite2_D50", parity_gen(50, L=2))
    elif which == "santafe":
        run_task("santafe", santafe_gen(), ntest=200, wash=200, nval=200)
    elif which == "markov":
        run_task("markov_dw20", markov_gen())
    elif which == "bruit":
        for s in (0.01, 0.05, 0.1, 0.2):
            run_task(f"bruit_s{s}", bruit_gen(s), radii=[0.9], seeds=[0, 1, 2])
    elif which == "mnist":
        run_mnist_geom()
    else:
        print("tache inconnue", which)
