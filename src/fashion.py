import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from chemins import BRUT  # noqa: E402
import numpy as np, gzip, json, sys
from suite_part1 import ridge_states, CONFIGS, RADII, results, RJ

def imgs(fn):
    with gzip.open(f"/tmp/fashion/{fn}", "rb") as f:
        d = np.frombuffer(f.read(), np.uint8, offset=16)
    return d.reshape(-1, 28, 28).astype(np.float32) / 255.0

def labs(fn):
    with gzip.open(f"/tmp/fashion/{fn}", "rb") as f:
        return np.frombuffer(f.read(), np.uint8, offset=8).astype(np.int64)

Xtr, ytr, Xte, yte = imgs("train-images-idx3-ubyte.gz"), labs("train-labels-idx1-ubyte.gz"), imgs("t10k-images-idx3-ubyte.gz"), labs("t10k-labels-idx1-ubyte.gz")
Xtr, ytr = Xtr[:5000], ytr[:5000]
print("Fashion-MNIST:", Xtr.shape, Xte.shape)

def acc(Ftr, ytr, Fte, yte, ridge=1e-6):
    Ftr = np.hstack([Ftr, np.ones((len(Ftr), 1), np.float32)]).astype(np.float64)
    Fte = np.hstack([Fte, np.ones((len(Fte), 1), np.float32)]).astype(np.float64)
    T = np.zeros((len(ytr), 10)); T[np.arange(len(ytr)), ytr] = 1.0
    A = Ftr.T @ Ftr + ridge * np.eye(Ftr.shape[1])
    Wout = np.linalg.solve(A, Ftr.T @ T)
    return float((np.argmax(Fte @ Wout, axis=1) == yte).mean())

# 14.2 : le meilleur rho par seed est choisi sur les 1000 dernieres images du
# train (validation), jamais sur le test ; readout final reajuste sur tout le train.
NVAL_MNIST = 1000
res = {}
for cid, name, N, taus in CONFIGS:
    if results.get("fashion", {}).get(cid, {}).get("proto") == "rho_sur_validation":
        res[cid] = results["fashion"][cid]
        continue
    accs = []
    for seed in range(3):
        best_val, best_test = -1.0, 0.0
        for rho in RADII:
            Ftr = ridge_states(Xtr, N, rho, taus, seed)
            Fte = ridge_states(Xte, N, rho, taus, seed)
            va = acc(Ftr[:-NVAL_MNIST], ytr[:-NVAL_MNIST], Ftr[-NVAL_MNIST:], ytr[-NVAL_MNIST:])
            if va > best_val:
                best_val = va
                best_test = acc(Ftr, ytr, Fte, yte)
        accs.append(best_test)
    res[cid] = {"name": name, "mean": float(np.mean(accs)), "std": float(np.std(accs)),
                "per_seed": accs, "proto": "rho_sur_validation"}
    print(f"  {name:35s} acc = {np.mean(accs)*100:.2f}% +/- {np.std(accs)*100:.2f}", flush=True)
results["fashion"] = res
json.dump(results, open(RJ, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
