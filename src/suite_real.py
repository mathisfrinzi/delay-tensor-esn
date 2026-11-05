"""Suite Partie 1bis : donnees reelles + MNIST pixel par pixel.
Toutes les taches utilisent les memes 5 configs a budget egal (tenseur 10x200^2 vs 632^2).
Protocole corrige (14.2, 2026-10-29) : le meilleur rho par seed est choisi sur un segment
de VALIDATION (taches temporelles : nval pas avant le test ; MNIST : NVAL_MNIST dernieres
images du train), jamais sur le test ; readout final reajuste sur train+validation. Les
entrees marquees proto="rho_sur_validation" sont reprises, les anciennes recalculees."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from chemins import BRUT  # noqa: E402
import numpy as np, json, os, sys, gzip, csv
from pathlib import Path

from suite_part1 import states_all, CONFIGS  # noyau valident

OUT = (BRUT / "tensor-suite/part1")
RJ = OUT / "results_real.json"
results = json.load(open(RJ, encoding="utf-8-sig")) if RJ.exists() else {}
NVAL_MNIST = 400  # dernieres images du train (2k) : le rho y est choisi (14.2)

def nrmse_reg(F, y, wash, ntest, ridge=1e-8):
    """y: (T,) ou (T, k). Readout ridge float64, NRMSE moyen par composante."""
    Fb = np.hstack([F, np.ones((len(F), 1), np.float32)]).astype(np.float64)
    y = np.asarray(y, np.float64)
    if y.ndim == 1: y = y[:, None]
    Ftr, ytr = Fb[wash:-ntest], y[wash:-ntest]
    Fte, yte = Fb[-ntest:], y[-ntest:]
    A = Ftr.T @ Ftr + ridge * np.eye(Ftr.shape[1])
    Wout = np.linalg.solve(A, Ftr.T @ ytr)
    pred = Fte @ Wout
    return float(np.mean(np.sqrt(np.mean((pred - yte) ** 2, 0)) / np.std(yte, 0)))

def classify(Ftr, ytr, Fte, yte, ridge=1e-6):
    Ftr = np.hstack([Ftr, np.ones((len(Ftr), 1), np.float32)]).astype(np.float64)
    Fte = np.hstack([Fte, np.ones((len(Fte), 1), np.float32)]).astype(np.float64)
    T = np.zeros((len(ytr), int(ytr.max()) + 1)); T[np.arange(len(ytr)), ytr] = 1.0
    A = Ftr.T @ Ftr + ridge * np.eye(Ftr.shape[1])
    Wout = np.linalg.solve(A, Ftr.T @ T)
    return float((np.argmax(Fte @ Wout, axis=1) == yte).mean())

def nrmse_reg_valtest(F, y, wash, ntest, nval, ridge=1e-8):
    """14.2 : (NRMSE validation, NRMSE test). Readout du test reajuste sur train+validation."""
    Fb = np.hstack([F, np.ones((len(F), 1), np.float32)]).astype(np.float64)
    y = np.asarray(y, np.float64)
    if y.ndim == 1: y = y[:, None]
    Ftr, ytr = Fb[wash:-ntest-nval], y[wash:-ntest-nval]
    Fva, yva = Fb[-ntest-nval:-ntest], y[-ntest-nval:-ntest]
    Fte, yte = Fb[-ntest:], y[-ntest:]
    A = Ftr.T @ Ftr + ridge * np.eye(Ftr.shape[1])
    Wout = np.linalg.solve(A, Ftr.T @ ytr)
    val = float(np.mean(np.sqrt(np.mean((Fva @ Wout - yva) ** 2, 0)) / np.std(yva, 0)))
    Ftv, ytv = Fb[wash:-ntest], y[wash:-ntest]
    A2 = Ftv.T @ Ftv + ridge * np.eye(Ftv.shape[1])
    Wout2 = np.linalg.solve(A2, Ftv.T @ ytv)
    return val, float(np.mean(np.sqrt(np.mean((Fte @ Wout2 - yte) ** 2, 0)) / np.std(yte, 0)))

def classify_valtest(Ftr, ytr, Fte, yte, nval, ridge=1e-6):
    """14.2 : (acc validation, acc test). Validation = nval dernieres images du train ;
    le readout du test est reajuste sur tout le train."""
    Ftrb = np.hstack([Ftr, np.ones((len(Ftr), 1), np.float32)]).astype(np.float64)
    nfit = len(ytr) - nval
    T = np.zeros((nfit, int(ytr.max()) + 1)); T[np.arange(nfit), ytr[:nfit]] = 1.0
    A = Ftrb[:nfit].T @ Ftrb[:nfit] + ridge * np.eye(Ftrb.shape[1])
    Wout = np.linalg.solve(A, Ftrb[:nfit].T @ T)
    va = float((np.argmax(Ftrb[nfit:] @ Wout, axis=1) == ytr[nfit:]).mean())
    return va, classify(Ftr, ytr, Fte, yte, ridge=ridge)

def ridge_states_px(Xb, N, rho, taus, seed, chunk=500):
    """Xb: (B, 784) pixels a plat. Entree scalaire = valeur du pixel. -> (B, N)."""
    rng = np.random.default_rng(seed)
    Win = (rng.uniform(-0.5, 0.5, size=(1, N))).astype(np.float32)
    if taus is None:
        W = [ (rng.normal(0, 1, (N, N)) * (rho / max(np.abs(np.linalg.eigvals(rng.normal(0, 1, (N, N)))).max(), 1e-12))).astype(np.float32) ]
        # NB: correction faite en place ci-dessous (erreur de frappe evitee)
        W = [ (rng.normal(0, 1, (N, N))).astype(np.float32) ]
        r = max(np.abs(np.linalg.eigvals(W[0])).max(), 1e-12)
        W[0] *= rho / r
    else:
        W = [(rng.normal(0, 1, (N, N))).astype(np.float32) for _ in taus]
        for i in range(len(W)):
            r = max(np.abs(np.linalg.eigvals(W[i])).max(), 1e-12)
            W[i] *= (rho / np.sqrt(len(taus))) / r
    tmax = 1 if taus is None else max(taus)
    out = np.empty((len(Xb), N), np.float32)
    for lo in range(0, len(Xb), chunk):
        B = Xb[lo:lo+chunk]
        x = np.zeros((len(B), N), np.float32)
        hist = np.zeros((tmax, len(B), N), np.float32)
        for t in range(B.shape[1]):
            drive = B[:, t:t+1] @ Win
            if taus is None:
                drive = drive + x @ W[0]
                x = np.tanh(drive)
            else:
                for i, tau in enumerate(taus):
                    drive = drive + hist[tau-1] @ W[i]
                x = np.tanh(drive)
                hist = np.concatenate([x[None], hist[:-1]], axis=0)
        out[lo:lo+len(B)] = x
    return out

# chargement donnees
def load_csv_col(path, col):
    import pandas as pd
    df = pd.read_csv(path)
    return df[col].values.astype(np.float64)

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
        return d.reshape(-1, 784).astype(np.float32) / 255.0
    def labs(fn):
        with gzip.open(os.path.join(BASE, fn), "rb") as f:
            return np.frombuffer(f.read(), np.uint8, offset=8).astype(np.int64)
    return imgs(files[0]), labs(files[1]), imgs(files[2]), labs(files[3])

# taches
def task_regression(key, gen, wash, ntest, radii=(0.5, 0.9, 0.99), seeds=3, note="", nval=500):
    if key in results and results[key].get("_proto") == "rho_sur_validation": print(f"[skip] {key}"); return
    res = {}
    for cid, name, N, taus in CONFIGS:
        bests = []
        for seed in range(seeds):
            u, y = gen()
            best_val, best = np.inf, np.inf
            for rho in radii:
                F = states_all(u, N, rho, taus, seed)[:len(y)]
                va, te = nrmse_reg_valtest(F, y, wash, ntest, nval)
                if va < best_val:
                    best_val, best = va, te
            bests.append(best)
        res[cid] = {"name": name, "mean": float(np.mean(bests)), "std": float(np.std(bests)),
                    "per_seed": [float(b) for b in bests]}
        print(f"  {name:35s} {res[cid]['mean']:.4f} +/- {res[cid]['std']:.4f}", flush=True)
    results[key] = {"_proto": "rho_sur_validation", **({"_note": note} if note else {}), **res}
    json.dump(results, open(RJ, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

def run_mnist_pixel():
    key = "mnist_pixel"
    if key in results and all(isinstance(results[key].get(cid), dict)
                               and results[key][cid].get("proto") == "rho_sur_validation"
                               for cid, *_ in CONFIGS):
        print("[skip]", key); return
    Xtr, ytr, Xte, yte = get_mnist()
    Xtr, ytr = Xtr[:2000], ytr[:2000]
    Xte, yte = Xte[:1000], yte[:1000]
    print(f"MNIST pixel-par-pixel: train {len(Xtr)}, test {len(Xte)}, 784 pas")
    res = {}
    for cid, name, N, taus in CONFIGS:
        accs = []
        for seed in range(2):
            best_val, best = -1.0, 0.0
            for rho in (0.9, 0.99):
                Ftr = ridge_states_px(Xtr, N, rho, taus, seed)
                Fte = ridge_states_px(Xte, N, rho, taus, seed)
                va, te = classify_valtest(Ftr, ytr, Fte, yte, NVAL_MNIST)
                if va > best_val:
                    best_val, best = va, te
            accs.append(best)
        res[cid] = {"name": name, "proto": "rho_sur_validation",
                    "mean": float(np.mean(accs)), "std": float(np.std(accs)),
                    "per_seed": [float(a) for a in accs]}
        print(f"  {name:35s} acc = {np.mean(accs)*100:.2f}% +/- {np.std(accs)*100:.2f}", flush=True)
    results[key] = res
    json.dump(results, open(RJ, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

def melbourne_data():
    import pandas as pd
    t = pd.read_csv("/tmp/datasets/melbourne.csv")["Temp"].values.astype(np.float64)
    # entree : temperature courante + biais ; cible : temperature suivante
    u = np.stack([t[:-1], np.ones(len(t)-1)], 1).astype(np.float32)
    return u, t[1:]

def sunspots_data():
    import pandas as pd
    s = pd.read_csv("/tmp/datasets/sunspots.csv")["Sunspots"].values.astype(np.float64)
    u = np.stack([s[:-1] / s.max(), np.ones(len(s)-1)], 1).astype(np.float32)
    return u, s[1:] / s.max()

def etth1_data(n=None):
    import pandas as pd
    df = pd.read_csv("/tmp/datasets/etth1.csv")
    cols = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
    V = df[cols].values.astype(np.float64)
    if n: V = V[-n:]
    V = (V - V.mean(0)) / V.std(0)
    u = np.hstack([V[:-1], np.ones((len(V)-1, 1), np.float32)]).astype(np.float32)
    return u, V[1:, -1]  # cible : OT normalise

def silverbox_data():
    from scipy.io import loadmat
    d = loadmat("/tmp/datasets/silverbox/SilverboxFiles/SNLS80mV.mat")
    u_all = d["V1"].ravel().astype(np.float64)
    y_all = d["V2"].ravel().astype(np.float64)
    # normalisation standard : amplitude de sortie
    scale = np.std(y_all[:100000])
    n_train, n_test = 40000, 131072 - 100000 + 10000  # train 40k, test = fin du record (31k + marge)
    n_test = 131072 - 100000
    u = u_all[:100000 + n_test]
    y = y_all[:100000 + n_test] / scale
    uu = np.stack([u / np.std(u), np.ones_like(u)], 1).astype(np.float32)
    return uu, y / scale

def wh_data():
    from scipy.io import loadmat
    d = loadmat("/tmp/datasets/WH.mat")
    u_all = d["uBenchMark"].ravel().astype(np.float64)
    y_all = d["yBenchMark"].ravel().astype(np.float64)
    scale = np.std(y_all[:100000])
    n_test = 88000
    u = u_all[:100000 + n_test]
    y = y_all[:100000 + n_test] / scale
    uu = np.stack([u / np.std(u), np.ones_like(u)], 1).astype(np.float32)
    return uu, y

def run_series(key, gen, wash, ntest, radii, seeds, note, ntrunc, nval=2000):
    """Pour Silverbox/WH : tronque l'entrainement pour rester rapide, test complet."""
    if key in results and results[key].get("_proto") == "rho_sur_validation": print(f"[skip] {key}"); return
    res = {}
    for cid, name, N, taus in CONFIGS:
        bests = []
        for seed in range(seeds):
            u, y = gen()
            # entrainement sur ntrunc premiers (apres washout), test sur les ntest derniers
            u_run = u[:ntrunc + ntest]
            y_run = y[:ntrunc + ntest]
            best_val, best = np.inf, np.inf
            for rho in radii:
                F = states_all(u_run, N, rho, taus, seed)[:len(y_run)]
                va, te = nrmse_reg_valtest(F, y_run, wash, ntest, nval)
                if va < best_val:
                    best_val, best = va, te
            bests.append(best)
        res[cid] = {"name": name, "mean": float(np.mean(bests)), "std": float(np.std(bests)),
                    "per_seed": [float(b) for b in bests]}
        print(f"  {name:35s} {res[cid]['mean']:.4f} +/- {res[cid]['std']:.4f}", flush=True)
    results[key] = {"_proto": "rho_sur_validation", "_note": note, **res}
    json.dump(results, open(RJ, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

if __name__ == "__main__":
    which = sys.argv[1]
    if which == "mnist_pixel":
        run_mnist_pixel()
    elif which == "melbourne":
        print("== Melbourne (temp mini journalieres, 1 pas) ==")
        task_regression("melbourne", melbourne_data, 300, 800)
    elif which == "sunspots":
        print("== Sunspots (mensuels, 1 pas) ==")
        task_regression("sunspots", sunspots_data, 300, 800)
    elif which == "etth1":
        print("== ETTh1 (temperature huile transfo, 1 pas) ==")
        task_regression("etth1", lambda: etth1_data(n=8000), 300, 2000)
    elif which == "silverbox":
        print("== Silverbox (systeme electronique reel) ==")
        run_series("silverbox", silverbox_data, 500, 20000, (0.5, 0.9), 2,
                   "Train 40k echantillons (multisine), test 20k fin de record, entree u dispo.", 40000)
    elif which == "wh":
        print("== Wiener-Hammerstein (circuit electronique reel) ==")
        run_series("wh", wh_data, 500, 20000, (0.5, 0.9), 2,
                   "Train 40k, test 20k sur les 88k de test officiels (tronques pour le temps de calcul).", 40000)

def run_mnist_pixel_one(cid_want):
    key = "mnist_pixel"
    res = results.get(key, {})
    if cid_want in res and res[cid_want].get("proto") == "rho_sur_validation":
        print("[skip]", cid_want); return
    Xtr, ytr, Xte, yte = get_mnist()
    Xtr, ytr = Xtr[:2000], ytr[:2000]
    Xte, yte = Xte[:1000], yte[:1000]
    for cid, name, N, taus in CONFIGS:
        if cid != cid_want: continue
        accs = []
        for seed in range(2):
            best_val, best = -1.0, 0.0
            for rho in (0.9, 0.99):
                Ftr = ridge_states_px(Xtr, N, rho, taus, seed)
                Fte = ridge_states_px(Xte, N, rho, taus, seed)
                va, te = classify_valtest(Ftr, ytr, Fte, yte, NVAL_MNIST)
                if va > best_val:
                    best_val, best = va, te
            accs.append(best)
        res[cid] = {"name": name, "proto": "rho_sur_validation",
                    "mean": float(np.mean(accs)), "std": float(np.std(accs)),
                    "per_seed": [float(a) for a in accs]}
        print(f"  {name:35s} acc = {np.mean(accs)*100:.2f}% +/- {np.std(accs)*100:.2f}", flush=True)
    results[key] = res
    json.dump(results, open(RJ, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

if len(sys.argv) > 1 and sys.argv[1].startswith("mnist_px:"):
    run_mnist_pixel_one(sys.argv[1].split(":")[1])
