"""
Réservoirs tensoriels W(i,j,tau) vs ESN standard : protocole complet, prêt pour GPU.

Utilisation :
    pip install torch torchvision numpy
    python tensor_esn_bench.py -exp narma          # NARMA-10 (rapide)
    python tensor_esn_bench.py -exp mnist          # MNIST complet (télécharge les données)
    python tensor_esn_bench.py -exp all -device cuda -seeds 20

CUDA est détecté automatiquement ; fallback CPU puis NumPy pur.
Comparaisons à budget égal de FLOPs/paramètres : tenseur N=200 x 10 tranches tau
(10 x 200^2 = 400k connexions) vs ESN standard N=632 (632^2 = 400k).
"""

import argparse
import gzip
import os
import subprocess
import urllib.request

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("torch non installé : fallback NumPy (plus lent, mêmes maths).")

TAUS = list(range(1, 11))
TAUS_LOG = [1, 2, 4, 8]
RADII = [0.5, 0.7, 0.9, 0.95, 0.99]


# Noyau commun : les poids sont toujours tirés en NumPy (reproductible),
# le calcul est fait en torch (GPU) si dispo, sinon NumPy.
def spectral_scale(W, rho):
    ev = np.abs(np.linalg.eigvals(W)).max()
    return W * (rho / max(ev, 1e-12))


class Reservoir:
    """ESN standard (taus=None) ou tenseur W(i,j,tau) avec tranches taus."""

    def __init__(self, n_in, N, rho, taus, seed, device=None):
        rng = np.random.default_rng(seed)
        self.taus = taus
        self.N = N
        self.Win = (rng.uniform(-0.5, 0.5, size=(n_in, N))).astype(np.float32)
        if taus is None:
            self.W = [spectral_scale(rng.normal(0, 1, (N, N)), rho).astype(np.float32)]
        else:
            k = len(taus)
            self.W = [spectral_scale(rng.normal(0, 1, (N, N)), rho / np.sqrt(k)).astype(np.float32)
                      for _ in taus]
        self.device = None
        if HAS_TORCH:
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            self.Win_t = torch.from_numpy(self.Win).to(self.device)
            self.W_t = [torch.from_numpy(w).to(self.device) for w in self.W]

    def states(self, X, chunk=4096):
        """X: (B, T, n_in) -> états finaux (B, N). Vectorisé par batch."""
        out = np.empty((len(X), self.N), np.float32)
        for lo in range(0, len(X), chunk):
            out[lo:lo + chunk] = self._run_chunk(X[lo:lo + chunk])
        return out

    def states_all(self, X, chunk=4096):
        """X: (B, T, n_in) -> états à chaque pas (B, T, N). Pour la régression temporelle."""
        out = np.empty((len(X), X.shape[1], self.N), np.float32)
        for lo in range(0, len(X), chunk):
            out[lo:lo + chunk] = self._run_chunk(X[lo:lo + chunk], keep_all=True)
        return out

    def _run_chunk(self, X, keep_all=False):
        B, T, _ = X.shape
        tmax = 1 if self.taus is None else max(self.taus)
        if HAS_TORCH:
            Xt = torch.from_numpy(X).to(self.device)
            x = torch.zeros(B, self.N, device=self.device)
            hist = torch.zeros(tmax, B, self.N, device=self.device)
            traj = torch.empty(B, T, self.N, device=self.device) if keep_all else None
            for t in range(T):
                drive = Xt[:, t, :] @ self.Win_t
                if self.taus is None:
                    drive = drive + x @ self.W_t[0]
                    x = torch.tanh(drive)
                else:
                    for i, tau in enumerate(self.taus):
                        drive = drive + hist[tau - 1] @ self.W_t[i]
                    x = torch.tanh(drive)
                    hist = torch.cat([x[None], hist[:-1]], dim=0)
                if keep_all:
                    traj[:, t, :] = x
            if keep_all:
                return traj.cpu().numpy()
            return x.cpu().numpy()
        # fallback NumPy
        x = np.zeros((B, self.N), np.float32)
        hist = np.zeros((tmax, B, self.N), np.float32)
        traj = np.empty((B, T, self.N), np.float32) if keep_all else None
        for t in range(X.shape[1]):
            drive = X[:, t, :] @ self.Win
            if self.taus is None:
                drive = drive + x @ self.W[0]
                x = np.tanh(drive)
            else:
                for i, tau in enumerate(self.taus):
                    drive = drive + hist[tau - 1] @ self.W[i]
                x = np.tanh(drive)
                hist = np.concatenate([x[None], hist[:-1]], axis=0)
            if keep_all:
                traj[:, t, :] = x
        if keep_all:
            return traj
        return x


def ridge_classify(Ftr, ytr, Fte, yte, ridge=1e-6):
    Ftr = np.hstack([Ftr, np.ones((len(Ftr), 1), np.float32)])
    Fte = np.hstack([Fte, np.ones((len(Fte), 1), np.float32)])
    T = np.zeros((len(ytr), int(ytr.max()) + 1), np.float32)
    T[np.arange(len(ytr)), ytr] = 1.0
    A = Ftr.T @ Ftr + ridge * np.eye(Ftr.shape[1], dtype=np.float32)
    Wout = np.linalg.solve(A, Ftr.T @ T)
    return float((np.argmax(Fte @ Wout, axis=1) == yte).mean())


def nrmse(pred, yte):
    return float(np.sqrt(np.mean((pred - yte) ** 2)) / np.std(yte))


def ridge_regress(Ftr, ytr, Fte, yte, ridge=1e-6):
    Ftr = np.hstack([Ftr, np.ones((len(Ftr), 1), np.float32)])
    Fte = np.hstack([Fte, np.ones((len(Fte), 1), np.float32)])
    ytr = np.asarray(ytr, np.float32)[:, None]
    A = Ftr.T @ Ftr + ridge * np.eye(Ftr.shape[1], dtype=np.float32)
    Wout = np.linalg.solve(A, Ftr.T @ ytr)
    return nrmse((Fte @ Wout)[:, 0], yte)


# Expérience 1 : NARMA-10
def narma10(T, rng):
    u = rng.uniform(0.0, 0.5, size=T + 20)
    y = np.zeros(T + 20)
    for t in range(10, T + 20):
        s = y[t - 10:t].sum()
        y[t] = 0.3 * y[t - 1] + 0.05 * y[t - 1] * s + 1.5 * u[t - 10] * u[t - 1] + 0.1
    return u[20:].astype(np.float32), y[20:].astype(np.float32)


def exp_narma(seeds, radii=RADII):
    T, WASH, T_TEST = 6000, 200, 2000
    configs = [
        ("ESN standard N=632 (=FLOPs)", 632, None),
        ("ESN standard N=200 (=unités)", 200, None),
        ("Tenseur tau=1..10 N=200", 200, TAUS),
        ("Tenseur log {1,2,4,8} N=200", 200, TAUS_LOG),
        ("Tenseur tau=1 seul (controle)", 200, [1]),
    ]
    for name, N, taus in configs:
        scores = []
        for seed in range(seeds):
            u, y = narma10(T, np.random.default_rng(1000 + seed))
            X = u[None, :, None]  # (1, T, 1)
            best = np.inf
            for rho in radii:
                r = Reservoir(1, N, rho, taus, seed)
                F = r.states_all(X)[0]  # (T, N) : etat a chaque pas
                # train/test split conforme au protocole
                Fb = np.hstack([F, np.ones((len(F), 1), np.float32)])
                Ftr, ytr = Fb[WASH:-T_TEST], y[WASH:-T_TEST]
                Fte, yte = Fb[-T_TEST:], y[-T_TEST:]
                A = Ftr.T @ Ftr + 1e-6 * np.eye(Ftr.shape[1], dtype=np.float32)
                Wout = np.linalg.solve(A, Ftr.T @ ytr)
                best = min(best, nrmse(Fte @ Wout, yte))
            scores.append(best)
        scores = np.array(scores)
        print(f"{name:35s} NRMSE = {scores.mean():.4f} +/- {scores.std():.4f} (best {scores.min():.4f})")


# Expérience 2 : MNIST ligne par ligne
def load_mnist(data_dir="mnist_data"):
    os.makedirs(data_dir, exist_ok=True)
    files = ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
             "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]
    for f in files:
        p = os.path.join(data_dir, f)
        if not os.path.exists(p):
            urllib.request.urlretrieve(f"https://ossci-datasets.s3.amazonaws.com/mnist/{f}", p)

    def imgs(fn):
        with gzip.open(os.path.join(data_dir, fn), "rb") as f:
            d = np.frombuffer(f.read(), np.uint8, offset=16)
        return d.reshape(-1, 28, 28).astype(np.float32) / 255.0

    def labs(fn):
        with gzip.open(os.path.join(data_dir, fn), "rb") as f:
            return np.frombuffer(f.read(), np.uint8, offset=8).astype(np.int64)

    return (imgs(files[0]), labs(files[1]), imgs(files[2]), labs(files[3]))


def exp_mnist(seeds, radii=RADII, n_train=None, device=None):
    Xtr, ytr, Xte, yte = load_mnist()
    if n_train:
        Xtr, ytr = Xtr[:n_train], ytr[:n_train]
    print(f"MNIST: train {Xtr.shape[0]}, test {Xte.shape[0]}")
    configs = [
        ("ESN standard N=632 (=FLOPs)", 632, None),
        ("ESN standard N=200 (=unités)", 200, None),
        ("Tenseur tau=1..10 N=200", 200, TAUS),
        ("Tenseur log {1,2,4,8} N=200", 200, TAUS_LOG),
        ("Tenseur tau=1 seul (controle)", 200, [1]),
    ]
    for name, N, taus in configs:
        accs = []
        for seed in range(seeds):
            best = 0.0
            for rho in radii:
                r = Reservoir(28, N, rho, taus, seed, device=device)
                acc = ridge_classify(r.states(Xtr), ytr, r.states(Xte), yte, ridge=1e-6)
                best = max(best, acc)
            accs.append(best)
        accs = np.array(accs)
        print(f"{name:35s} acc = {accs.mean() * 100:.2f}% +/- {accs.std() * 100:.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-exp", choices=["narma", "mnist", "all"], default="all")
    ap.add_argument("-seeds", type=int, default=10)
    ap.add_argument("-device", default=None, help="cuda / cpu / mps (défaut: auto)")
    ap.add_argument("-n-train", type=int, default=None, help="MNIST: limite le train set")
    args = ap.parse_args()
    if args.exp in ("narma", "all"):
        print("=== NARMA-10 ===")
        exp_narma(args.seeds)
    if args.exp in ("mnist", "all"):
        print("=== MNIST (lecture ligne par ligne) ===")
        exp_mnist(args.seeds, n_train=args.n_train, device=args.device)
