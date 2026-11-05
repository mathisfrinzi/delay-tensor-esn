"""Validation 60k: ESN standard N=632 vs tenseur tau=1..10 N=200, hygiene memoire stricte.
Protocole corrige (14.2, 2026-10-29) : le meilleur rho par seed est choisi sur les NVAL_60K
dernieres images du train (validation), jamais sur le test ; readout final reajuste sur tout
le train. Les entrees marquees proto="rho_sur_validation" sont reprises, les anciennes
recalculees."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from chemins import BRUT  # noqa: E402
import numpy as np, gzip, json, gc, os
from pathlib import Path

BASE = "/tmp/mnist"
MN = "train-images-idx3-ubyte.gz"; ML = "train-labels-idx1-ubyte.gz"
TN = "t10k-images-idx3-ubyte.gz"; TL = "t10k-labels-idx1-ubyte.gz"

if not os.path.exists(f"{BASE}/{MN}"):
    os.makedirs(BASE, exist_ok=True)
    import subprocess
    for f in [MN, ML, TN, TL]:
        subprocess.run(["curl", "-sL", "-o", f"{BASE}/{f}",
                        f"https://ossci-datasets.s3.amazonaws.com/mnist/{f}"], check=True)

def imgs(fn):
    with gzip.open(fn, "rb") as f:
        d = np.frombuffer(f.read(), np.uint8, offset=16)
    return d.reshape(-1, 28, 28).astype(np.float32) / 255.0

def labs(fn):
    with gzip.open(fn, "rb") as f:
        return np.frombuffer(f.read(), np.uint8, offset=8).astype(np.int64)

Xtr, ytr, Xte, yte = imgs(f"{BASE}/{MN}"), labs(f"{BASE}/{ML}"), imgs(f"{BASE}/{TN}"), labs(f"{BASE}/{TL}")
print("loaded", Xtr.shape, Xte.shape, flush=True)

def spectral(W, rho):
    return W * (rho / max(np.abs(np.linalg.eigvals(W)).max(), 1e-12))

def states(Xb, N, rho, taus, seed, chunk=4000):
    rng = np.random.default_rng(seed)
    Win = (rng.uniform(-1, 1, size=(28, N)) * 0.5).astype(np.float32)
    if taus is None:
        W = spectral(rng.normal(0, 1, size=(N, N)), rho).astype(np.float32)
    else:
        W = {t: spectral(rng.normal(0, 1, size=(N, N)), rho / np.sqrt(len(taus))).astype(np.float32) for t in taus}
    tmax = 1 if taus is None else max(taus)
    out = np.empty((len(Xb), N), np.float32)
    for lo in range(0, len(Xb), chunk):
        B = Xb[lo:lo+chunk]
        x = np.zeros((len(B), N), np.float32)
        hist = np.zeros((tmax, len(B), N), np.float32)
        for t in range(28):
            drive = B[:, t, :] @ Win
            if taus is None:
                drive = drive + x @ W
                x = np.tanh(drive)
            else:
                for tau in taus:
                    drive = drive + hist[tau-1] @ W[tau]
                x = np.tanh(drive)
                hist = np.concatenate([x[None], hist[:-1]], axis=0)
        out[lo:lo+len(B)] = x
        del x, hist
    return out

def ridge_acc(Ftr, ytr_, Fte, yte_, ridge=1e-6):
    T = np.zeros((len(ytr_), 10), np.float32); T[np.arange(len(ytr_)), ytr_] = 1.0
    Ftr = np.hstack([Ftr, np.ones((len(Ftr),1), np.float32)])
    Fte = np.hstack([Fte, np.ones((len(Fte),1), np.float32)])
    A = Ftr.T @ Ftr + ridge*np.eye(Ftr.shape[1], dtype=np.float32)
    Wout = np.linalg.solve(A, Ftr.T @ T)
    del A, T, Ftr
    return float((np.argmax(Fte @ Wout, axis=1) == yte_).mean())

def ridge_acc_valtest(Ftr, ytr_, Fte, yte_, nval, ridge=1e-6):
    """14.2 : (acc validation, acc test). Validation = nval dernieres images du train ;
    le readout du test est reajuste sur tout le train."""
    nfit = len(ytr_) - nval
    T = np.zeros((nfit, 10), np.float32); T[np.arange(nfit), ytr_[:nfit]] = 1.0
    Fb = np.hstack([Ftr[:nfit], np.ones((nfit, 1), np.float32)])
    A = Fb.T @ Fb + ridge * np.eye(Fb.shape[1], dtype=np.float32)
    Wout = np.linalg.solve(A, Fb.T @ T)
    Fva = np.hstack([Ftr[nfit:], np.ones((nval, 1), np.float32)])
    va = float((np.argmax(Fva @ Wout, axis=1) == ytr_[nfit:]).mean())
    te = ridge_acc(Ftr, ytr_, Fte, yte_, ridge=ridge)
    return va, te

NVAL_60K = 10000  # dernieres images du train 60k : le rho y est choisi (14.2)
results = json.load(open(str(BRUT / "tensor-esn-mnist/results.json")))
for name, N, taus in [("ESN standard N=632 (60k)", 632, None),
                      ("Tenseur tau=1..10 N=200 (60k)", 200, list(range(1,11)))]:
    if name in results and results[name].get("proto") == "rho_sur_validation":
        print(f"[skip] {name}"); continue
    accs = []
    for seed in range(2):
        best_val, best = -1.0, 0.0
        for rho in [0.5, 0.7, 0.9, 0.99]:
            Ftr = states(Xtr, N, rho, taus, seed); Fte = states(Xte, N, rho, taus, seed)
            va, te = ridge_acc_valtest(Ftr, ytr, Fte, yte, NVAL_60K)
            if va > best_val:
                best_val, best = va, te
            del Ftr, Fte; gc.collect()
        accs.append(best)
    results[name] = {"proto": "rho_sur_validation", "mean": float(np.mean(accs)),
                     "std": float(np.std(accs)), "per_seed": accs}
    print(f"{name:35s} acc = {np.mean(accs)*100:.2f}% +/- {np.std(accs)*100:.2f}", flush=True)

with open(str(BRUT / "tensor-esn-mnist/results.json"), "w") as f:
    json.dump(results, f, indent=2)
print("saved")
