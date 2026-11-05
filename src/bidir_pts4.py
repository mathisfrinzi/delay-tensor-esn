"""2.10ter : Controle de parite de readout MNIST : le miroir independant elu
(2.10, 87,5 % avec 2x223=446 features : 1 point par direction) relu a
4 points (892 features, parite exacte avec le flat couple xpm de cpl2_bidir)
et a 8 points (1784 features, avantage x2 au miroir). Meme budget recurrent
(400k), memes elus {2,5,7,10}, memes tirages que bi_dir.py (make_W/make_Win).
Points lus : fwd(0),fwd(27),bwd(27),bwd(0) pour pts4 ; les 4 pas
{0,9,18,27} de chaque direction pour pts8. Si le miroir depasse le flat
couple (89,3 %) meme a 892 features, le record du couple n'est pas un
record de methode mais de readout. Checkpoint : part2/results_bidir.json
(cles tens_elected_pts4 / tens_elected_pts8, mnist)."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from chemins import BRUT  # noqa: E402
import numpy as np, json, gzip, os
from pathlib import Path

OUT = (BRUT / "tensor-suite/part2")
RJ = OUT / "results_bidir.json"
results = json.load(open(RJ, encoding="utf-8-sig")) if RJ.exists() else {}
SEEDS = [0, 1, 2]
ELECTED = [2, 5, 7, 10]
RADII = [0.9, 0.99]
N_F = N_B = 223  # floor(sqrt(632^2 / 4)) : budget par direction = elu 2.2
KEYS = {"pts4": [0, 27], "pts8": [0, 9, 18, 27]}


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
            dd = np.frombuffer(f.read(), np.uint8, offset=16)
        return dd.reshape(-1, 28, 28).astype(np.float32) / 255.0
    def labs(fn):
        with gzip.open(os.path.join(BASE, fn), "rb") as f:
            return np.frombuffer(f.read(), np.uint8, offset=8).astype(np.int64)
    return imgs(files[0])[:5000], labs(files[1])[:5000], imgs(files[2])[:1000], labs(files[3])[:1000]


def make_W(N, taus, rho, seed):
    rng = np.random.default_rng(seed + 7)
    K = len(taus)
    W = np.empty((K, N, N), np.float32)
    for i in range(K):
        M = rng.normal(0, 1, (N, N)).astype(np.float32)
        r = max(np.abs(np.linalg.eigvals(M.astype(np.float64))).max(), 1e-12)
        W[i] = M * ((rho / np.sqrt(K)) / r)
    return W


def make_Win(d, N, seed):
    rng = np.random.default_rng(seed)
    return rng.uniform(-0.5, 0.5, size=(d, N)).astype(np.float32)


def states_mnist_all(Xb, taus, W, Win, chunk=1000):
    """Comme bi_dir.states_mnist mais retourne TOUS les pas : F (28, nb, N)."""
    K, N, _ = W.shape
    tmax = max(taus)
    sel = np.array(taus) - 1
    F = np.empty((28, len(Xb), N), np.float32)
    for lo in range(0, len(Xb), chunk):
        B = Xb[lo:lo+chunk]
        x = np.zeros((len(B), N), np.float32)
        hist = np.zeros((tmax, len(B), N), np.float32)
        for t in range(28):
            drive = B[:, t, :] @ Win + np.tensordot(hist[sel], W, axes=([0, 2], [0, 2]))
            x = np.tanh(drive)
            hist = np.concatenate([x[None], hist[:-1]], axis=0)
            F[t, lo:lo+len(B)] = x
    return F


NVAL_MNIST = 1000   # 14.2 : validation = 1000 dernieres images du train (choix de rho)


def readout(Ftr, ytr, Fte, yte):
    """Retourne (acc validation, acc test). Test : readout refit sur tout le train."""
    Ftrb = np.hstack([Ftr, np.ones((len(Ftr), 1), np.float32)]).astype(np.float64)
    Fteb = np.hstack([Fte, np.ones((len(Fte), 1), np.float32)]).astype(np.float64)
    Tt = np.zeros((len(ytr), 10)); Tt[np.arange(len(ytr)), ytr] = 1.0
    nh = len(ytr) - NVAL_MNIST
    Av = Ftrb[:nh].T @ Ftrb[:nh] + 1e-6 * np.eye(Ftrb.shape[1])
    Wv = np.linalg.solve(Av, Ftrb[:nh].T @ Tt[:nh])
    va = float((np.argmax(Ftrb[nh:] @ Wv, axis=1) == ytr[nh:]).mean())
    A = Ftrb.T @ Ftrb + 1e-6 * np.eye(Ftrb.shape[1])
    Wout = np.linalg.solve(A, Ftrb.T @ Tt)
    return va, float((np.argmax(Fteb @ Wout, axis=1) == yte).mean())


Xtr, ytr, Xte, yte = get_mnist()
for key in KEYS:
    cfg = f"tens_elected_{key}"
    cell = results.setdefault(cfg, {}).setdefault("mnist", {
        "taus": ELECTED, "N_f": N_F, "N_b": N_B, "budget": 632 ** 2, "seeds": {}})
    for s in SEEDS:
        if str(s) in cell["seeds"] and cell["seeds"][str(s)].get("proto") == "rho_sur_validation":
            print(f"{cfg}/mnist seed{s} deja fait", flush=True)
            continue
        best = (-1.0, None, None)
        for rho in RADII:
            Ftr_f = states_mnist_all(Xtr, ELECTED, make_W(N_F, ELECTED, rho, s), make_Win(28, N_F, s))
            Ftr_b = states_mnist_all(Xtr[:, ::-1, :], ELECTED, make_W(N_B, ELECTED, rho, s + 100), make_Win(28, N_B, s + 1000))
            Fte_f = states_mnist_all(Xte, ELECTED, make_W(N_F, ELECTED, rho, s), make_Win(28, N_F, s))
            Fte_b = states_mnist_all(Xte[:, ::-1, :], ELECTED, make_W(N_B, ELECTED, rho, s + 100), make_Win(28, N_B, s + 1000))
            pts = KEYS[key]
            Ftr = np.hstack([Ftr_f[p, :, :].copy() for p in pts] + [Ftr_b[p, :, :].copy() for p in pts])
            Fte = np.hstack([Fte_f[p, :, :].copy() for p in pts] + [Fte_b[p, :, :].copy() for p in pts])
            va, te = readout(Ftr, ytr, Fte, yte)
            print(f"  rho={rho} {Ftr.shape[1]-1} features val={va:.4f} test={te:.4f}", flush=True)
            if va > best[0]:          # 14.2 : rho choisi sur la validation
                best = (va, rho, te)
        cell["seeds"][str(s)] = {"test": best[2], "rho": best[1], "val": best[0], "proto": "rho_sur_validation"}
        json.dump(results, open(RJ, "w", encoding="utf-8"), indent=1)
    tests = [cell["seeds"][str(s)]["test"] for s in SEEDS if str(s) in cell["seeds"] and "test" in cell["seeds"][str(s)]]
    if len(tests) == len(SEEDS):
        cell["test_mean"] = float(np.mean(tests)); cell["test_std"] = float(np.std(tests))
        json.dump(results, open(RJ, "w", encoding="utf-8"), indent=1)
        print(f"{cfg}/mnist COMPLETE: {np.mean(tests):.4f} +/- {np.std(tests):.4f}")
    else:
        print(f"{cfg}/mnist pas encore complet")
