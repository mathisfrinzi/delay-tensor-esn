"""MNIST via reservoir: image lue ligne par ligne (28 pas, entree 28D), classification
de l'etat final (ridge one-hot). ESN standard vs tenseur W(i,j,tau), tau=1..10.
Tout est vectorise par batch: etats (B, N).
Protocole corrige (14.2, 2026-10-29) : le meilleur rho par seed est choisi sur un jeu de
VALIDATION (NVAL_MNIST dernieres images du train ; 60k : NVAL_60K dernieres images), jamais
sur le test ; readout final reajuste sur tout le train. Les entrees marquees
proto="rho_sur_validation" sont reprises, les anciennes recalculees."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from chemins import BRUT  # noqa: E402
import numpy as np, gzip, json, os
from pathlib import Path

BASE = "/tmp/mnist"

def load():
    def imgs(fn):
        with gzip.open(fn, "rb") as f:
            d = np.frombuffer(f.read(), np.uint8, offset=16)
        return d.reshape(-1, 28, 28).astype(np.float32) / 255.0
    def labs(fn):
        with gzip.open(fn, "rb") as f:
            return np.frombuffer(f.read(), np.uint8, offset=8).astype(np.int64)
    return imgs(f"{BASE}/train-images-idx3-ubyte.gz"), labs(f"{BASE}/train-labels-idx1-ubyte.gz"), \
           imgs(f"{BASE}/t10k-images-idx3-ubyte.gz"), labs(f"{BASE}/t10k-labels-idx1-ubyte.gz")

def spectral(W, rho):
    return W * (rho / max(np.abs(np.linalg.eigvals(W)).max(), 1e-12))

def run_std(Xb, N, rho, seed, chunk=8000):
    """Xb: (B, 28, 28). rows -> sequence. retourne etats finaux (B, N). Traite par chunks memoire."""
    outs = [run_std_chunk(Xb[i:i+chunk], N, rho, seed) for i in range(0, len(Xb), chunk)]
    return np.vstack(outs)

def run_std_chunk(Xb, N, rho, seed):
    rng = np.random.default_rng(seed)
    Win = rng.uniform(-1, 1, size=(28, N)) * 0.5
    W = spectral(rng.normal(0, 1, size=(N, N)), rho)
    B = Xb.shape[0]
    x = np.zeros((B, N), dtype=np.float32)
    for t in range(28):
        x = np.tanh(Xb[:, t, :] @ Win + x @ W)
    return x

def run_tensor(Xb, N, taus, rho, seed, chunk=8000):
    outs = [run_tensor_chunk(Xb[i:i+chunk], N, taus, rho, seed) for i in range(0, len(Xb), chunk)]
    return np.vstack(outs)

def run_tensor_chunk(Xb, N, taus, rho, seed):
    rng = np.random.default_rng(seed)
    Win = rng.uniform(-1, 1, size=(28, N)) * 0.5
    W = {tau: spectral(rng.normal(0, 1, size=(N, N)), rho / np.sqrt(len(taus))) for tau in taus}
    tmax = max(taus)
    B = Xb.shape[0]
    hist = np.zeros((tmax, B, N), dtype=np.float32)  # hist[k] = x(t-k-1)
    for t in range(28):
        drive = Xb[:, t, :] @ Win
        for tau in taus:
            drive = drive + hist[tau - 1] @ W[tau]
        x = np.tanh(drive)
        hist = np.concatenate([x[None, :, :], hist[:-1]], axis=0)
    return x

def ridge_acc(Ftr, ytr, Fte, yte, ridge=1e-4):
    T = np.zeros((len(ytr), 10), dtype=np.float32)
    T[np.arange(len(ytr)), ytr] = 1.0
    Fb_tr = np.hstack([Ftr, np.ones((len(Ftr), 1), np.float32)])
    Fb_te = np.hstack([Fte, np.ones((len(Fte), 1), np.float32)])
    A = Fb_tr.T @ Fb_tr + ridge * np.eye(Fb_tr.shape[1], dtype=np.float32)
    Wout = np.linalg.solve(A, Fb_tr.T @ T)
    acc = (np.argmax(Fb_te @ Wout, axis=1) == yte).mean()
    return float(acc)

def ridge_acc_valtest(Ftr, ytr, Fte, yte, nval, ridge=1e-4):
    """14.2 : (acc validation, acc test). Validation = nval dernieres images du train ;
    le readout du test est reajuste sur tout le train."""
    nfit = len(ytr) - nval
    T = np.zeros((nfit, 10), dtype=np.float32)
    T[np.arange(nfit), ytr[:nfit]] = 1.0
    Fb_tr = np.hstack([Ftr[:nfit], np.ones((nfit, 1), np.float32)])
    A = Fb_tr.T @ Fb_tr + ridge * np.eye(Fb_tr.shape[1], dtype=np.float32)
    Wout = np.linalg.solve(A, Fb_tr.T @ T)
    Fb_va = np.hstack([Ftr[nfit:], np.ones((nval, 1), np.float32)])
    va = float((np.argmax(Fb_va @ Wout, axis=1) == ytr[nfit:]).mean())
    te = ridge_acc(Ftr, ytr, Fte, yte, ridge=ridge)
    return va, te

if not os.path.exists(BASE + "/train-images-idx3-ubyte.gz"):
    import subprocess
    os.makedirs(BASE, exist_ok=True)
    for f in ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
              "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]:
        subprocess.run(["curl", "-sL", "-o", f"{BASE}/{f}",
                       f"https://ossci-datasets.s3.amazonaws.com/mnist/{f}"], check=True)

Xtr, ytr, Xte, yte = load()
N_TRAIN = 5000
NVAL_MNIST = 1000  # dernieres images du train : le rho y est choisi (14.2)
Xtr, ytr, Xte, yte = Xtr[:N_TRAIN], ytr[:N_TRAIN], Xte[:1000], yte[:1000]
print("data:", Xtr.shape, Xte.shape)

TAUS = list(range(1, 11))
CONFIGS = [
    ("ESN standard N=632 (=FLOPs)", lambda r, s: run_std(Xtr, 632, r, s), lambda r, s: run_std(Xte, 632, r, s)),
    ("ESN standard N=200 (=unités)", lambda r, s: run_std(Xtr, 200, r, s), lambda r, s: run_std(Xte, 200, r, s)),
    ("Tenseur tau=1..10 N=200", lambda r, s: run_tensor(Xtr, 200, TAUS, r, s), lambda r, s: run_tensor(Xte, 200, TAUS, r, s)),
    ("Tenseur log {1,2,4,8} N=200", lambda r, s: run_tensor(Xtr, 200, [1,2,4,8], r, s), lambda r, s: run_tensor(Xte, 200, [1,2,4,8], r, s)),
    ("Tenseur tau=1 seul (contrôle)", lambda r, s: run_tensor(Xtr, 200, [1], r, s), lambda r, s: run_tensor(Xte, 200, [1], r, s)),
]
radii = [0.5, 0.9, 0.99]
out = (BRUT / "tensor-esn-mnist")
out.mkdir(parents=True, exist_ok=True)
RJ = out / "results.json"
results = json.load(open(RJ, encoding="utf-8-sig")) if RJ.exists() else {}
for name, ftr, fte in CONFIGS:
    if name in results and results[name].get("proto") == "rho_sur_validation":
        print(f"[skip] {name}"); continue
    accs = []
    for seed in range(3):
        best_val, best = -1.0, 0.0
        for rho in radii:
            Ftr = ftr(rho, seed); Fte = fte(rho, seed)
            # IMPORTANT: même seed -> mêmes poids pour train/test (les lambdas re-tirent avec le même rng seed)
            va, te = ridge_acc_valtest(Ftr, ytr, Fte, yte, NVAL_MNIST)
            if va > best_val:
                best_val, best = va, te
        accs.append(best)
    accs = np.array(accs)
    results[name] = {"proto": "rho_sur_validation", "mean": float(accs.mean()), "std": float(accs.std()),
                     "per_seed": accs.tolist()}
    print(f"{name:35s} acc = {accs.mean()*100:.2f}% +/- {accs.std()*100:.2f}")

out = (BRUT / "tensor-esn-mnist")
out.mkdir(parents=True, exist_ok=True)
with open(out / "results.json", "w") as f:
    json.dump(results, f, indent=2)
print("saved")

# validation: jeu d'entrainement complet (60k), top-2 configs, sweep rho plus large
Xtr_f, ytr_f, Xte_f, yte_f = load()
NVAL_60K = 10000  # dernieres images du train 60k : le rho y est choisi (14.2)
radii2 = [0.3, 0.5, 0.7, 0.9, 0.99]
final = {}
for name, mk in [
    ("ESN standard N=632 (60k)", lambda r, s: (run_std(Xtr_f, 632, r, s), run_std(Xte_f, 632, r, s))),
    ("Tenseur tau=1..10 N=200 (60k)", lambda r, s: (run_tensor(Xtr_f, 200, TAUS, r, s), run_tensor(Xte_f, 200, TAUS, r, s))),
]:
    if name in results and results[name].get("proto") == "rho_sur_validation":
        print(f"[skip] {name}"); continue
    accs = []
    for seed in range(3):
        best_val, best = -1.0, 0.0
        for rho in radii2:
            Ftr, Fte = mk(rho, seed)
            va, te = ridge_acc_valtest(Ftr, ytr_f, Fte, yte_f, NVAL_60K, ridge=1e-6)
            if va > best_val:
                best_val, best = va, te
        accs.append(best)
    accs = np.array(accs)
    final[name] = {"proto": "rho_sur_validation", "mean": float(accs.mean()), "std": float(accs.std()),
                   "per_seed": accs.tolist()}
    print(f"{name:35s} acc = {accs.mean()*100:.2f}% +/- {accs.std()*100:.2f}")
results.update(final)
with open(str(BRUT / "tensor-esn-mnist/results.json"), "w") as f:
    json.dump(results, f, indent=2)
print("saved")
