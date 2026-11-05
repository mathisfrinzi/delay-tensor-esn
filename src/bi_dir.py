"""2.10 : Bidirectionnalite : bi-ESN directionnel et tenseur D a retards
negatifs (miroir temporel). Deux directions : avant (causale, retards +tau) et
arriere (acausale, "retards negatifs" implementes en passant la sequence
INVERSEE dans un reservoir miroir de memes tau). Le budget recurrent total
(632^2 = 400k) est coupe entre les deux directions selon la part beta allouee
a l'arriere : esn_b25 (25 % arriere), esn_b50 (symetrique), esn_b100 (100 %
arriere, anti-causal pur). L'ESN = tenseur a tranche unique tau=1 (rayon rho
plein), donc bi-ESN = bi-tenseur {+1} U {-1}.
Versions tenseur D : tens_uni (uniforme {1..10} en chaque direction, N=141)
et tens_elected (elus 2.2 en chaque direction, budget egal au causal elu).
Washout de la direction arriere : le futur de la sequence AU-DELA de la
fenetre d'evaluation (200 pts NARMA / 500 pts phys) : l'extension d'entree ne
sert qu'a ca, cibles et splits inchanges (protocole std d'eval_pruned, meilleur
rho par seed sur la VALIDATION depuis 14.2). MNIST : l'image lue ligne par ligne, la direction
arriere = lignes en ordre inverse (pas de washout, symetrique du causal).
Checkpoint : part2/results_bidir.json.
Protocole corrige (14.2, 2026-11-02) : le meilleur rho par seed est choisi sur un
segment de VALIDATION (NVAL[task] pas avant le test ; mnist : 1000 dernieres images
du train), jamais sur le test ; le readout du test est reajuste sur train+validation.
Les seeds sans "proto" (ancien choix sur le test) sont recalculees."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from chemins import BRUT  # noqa: E402
import numpy as np, json, time, sys, gzip, os
from pathlib import Path
from scipy.io import loadmat

OUT = (BRUT / "tensor-suite/part2")
RJ = OUT / "results_bidir.json"
results = json.load(open(RJ, encoding="utf-8-sig")) if RJ.exists() else {}
SEEDS = [0, 1, 2]
BUDGET = 632 ** 2
TIME_BUDGET = float(os.environ.get("TEMPS_MAX", "240"))
NVAL = {"narma30": 1000, "silverbox": 2000, "wh": 2000}   # validation : choix de rho (14.2)
NVAL_MNIST = 1000

ELECTED = {"narma30": [1, 10], "silverbox": [2, 3, 4, 5, 7, 8, 10],
           "wh": [4, 7], "mnist": [2, 5, 7, 10]}

def narma30(seed, T=5200, ext=200):
    rng = np.random.default_rng(seed)
    u = rng.uniform(0, 0.5, size=T + ext)
    y = np.zeros(T, np.float32)
    a, b, g, d = 0.2, 0.004, 1.5, 0.001
    for t in range(30, T):
        y[t] = a * y[t-1] + b * y[t-1] * y[t-30:t].sum() + g * u[t-30] * u[t-1] + d
    return np.stack([u, np.ones_like(u)], 1).astype(np.float32), y

def load_phys(which, ext=500):
    if which == "silverbox":
        d = loadmat("/tmp/datasets/silverbox/SilverboxFiles/SNLS80mV.mat")
        u = d["V1"].ravel().astype(np.float64); y = d["V2"].ravel().astype(np.float64)
    else:
        d = loadmat("/tmp/datasets/WH.mat")
        u = d["uBenchMark"].ravel().astype(np.float64); y = d["yBenchMark"].ravel().astype(np.float64)
    s = np.std(y[:100000])
    U = u[:30000 + ext]; Y = y[:30000] / s
    U = U / np.std(u[:30000])
    return np.stack([U, np.ones_like(U)], 1).astype(np.float32), Y.astype(np.float32)

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

def states_seq(u, taus, W, Win):
    T = len(u)
    K, N, _ = W.shape
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

def states_mnist(Xb, taus, W, Win, chunk=1000):
    K, N, _ = W.shape
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


def bi_states_seq(u_ext, T, N_f, N_b, taus, rho, seed):
    """Etats concatanes [x_fwd(t) ; x_bwd(t)]. u_ext : fenetre + extension
    futur (washout arriere). x_bwd(t) = etat arriere ayant consomme
    u_ext[T+ext-1] ... u_ext[t]."""
    u_fwd = u_ext[:T]
    u_bwd = u_ext[::-1]
    ext = len(u_ext) - T
    parts = []
    if N_f > 0:  # direction avant : meme tirage que les refs (seed+7 / seed)
        parts.append(states_seq(u_fwd, taus, make_W(N_f, taus, rho, seed), make_Win(2, N_f, seed)))
    if N_b > 0:
        Z = states_seq(u_bwd, taus, make_W(N_b, taus, rho, seed + 100), make_Win(2, N_b, seed + 1000))
        parts.append(Z[ext:][::-1])
    return np.hstack(parts) if len(parts) > 1 else parts[0]

def bi_states_mnist(X, N_f, N_b, taus, rho, seed):
    parts = []
    if N_f > 0:
        W_f = make_W(N_f, taus, rho, seed)
        parts.append(states_mnist(X, taus, W_f, make_Win(28, N_f, seed)))
    if N_b > 0:
        W_b = make_W(N_b, taus, rho, seed + 100)
        parts.append(states_mnist(X[:, ::-1, :], taus, W_b, make_Win(28, N_b, seed + 1000)))
    return np.hstack(parts) if len(parts) > 1 else parts[0]

RADII = {"narma30": [0.5, 0.9, 0.99], "silverbox": [0.5, 0.9], "wh": [0.5, 0.9], "mnist": [0.9, 0.99]}
SPLITS = {"narma30": (200, 2000), "silverbox": (500, 10000), "wh": (500, 10000)}

def resolve(task, config):
    scale = 1.0
    if config == "esn_b25":
        taus, beta = [1], 0.25
    elif config == "esn_b50":
        taus, beta = [1], 0.5
    elif config == "esn_b100":
        taus, beta = [1], 1.0
    elif config == "esn_half":
        taus, beta, scale = [1], 0.0, 0.5
    elif config == "tens_half":
        taus, beta, scale = list(range(1, 11)), 0.0, 0.5
    elif config == "tens_uni":
        taus, beta = list(range(1, 11)), 0.5
    elif config == "tens_elected":
        taus, beta = ELECTED[task], 0.5
    else:
        raise ValueError(config)
    K = len(taus)
    N_b = int((BUDGET * scale * beta / K) ** 0.5)
    N_f = int((BUDGET * scale * (1 - beta) / K) ** 0.5)
    return taus, N_f, N_b

task, config = sys.argv[1], sys.argv[2]
taus, N_f, N_b = resolve(task, config)
budget_used = (N_f ** 2 + N_b ** 2) * len(taus)
cell = results.setdefault(config, {}).setdefault(task, {"taus": taus, "N_f": N_f, "N_b": N_b, "budget": budget_used, "seeds": {}})
MAXIM = task == "mnist"

if task == "mnist":
    Xtr, ytr, Xte, yte = get_mnist()
t0 = time.time()
for s in SEEDS:
    if str(s) in cell["seeds"] and cell["seeds"][str(s)].get("proto") == "rho_sur_validation":
        continue
    if time.time() - t0 > TIME_BUDGET:
        print("time budget : reprendre", flush=True)
        break
    if task == "mnist":
        best = (0.0, None); best_val = -1.0
        for rho in RADII[task]:
            Ftr = bi_states_mnist(Xtr, N_f, N_b, taus, rho, s)
            Fte = bi_states_mnist(Xte, N_f, N_b, taus, rho, s)
            Ftrb = np.hstack([Ftr, np.ones((len(Ftr), 1), np.float32)]).astype(np.float64)
            Fteb = np.hstack([Fte, np.ones((len(Fte), 1), np.float32)]).astype(np.float64)
            nfit = len(ytr) - NVAL_MNIST
            Tf = np.zeros((nfit, 10)); Tf[np.arange(nfit), ytr[:nfit]] = 1.0
            Af = Ftrb[:nfit].T @ Ftrb[:nfit] + 1e-6 * np.eye(Ftrb.shape[1])
            Wv = np.linalg.solve(Af, Ftrb[:nfit].T @ Tf)
            va = float((np.argmax(Ftrb[nfit:] @ Wv, axis=1) == ytr[nfit:]).mean())
            Tt = np.zeros((len(ytr), 10)); Tt[np.arange(len(ytr)), ytr] = 1.0
            A = Ftrb.T @ Ftrb + 1e-6 * np.eye(Ftrb.shape[1])
            Wout = np.linalg.solve(A, Ftrb.T @ Tt)
            te = float((np.argmax(Fteb @ Wout, axis=1) == yte).mean())
            if va > best_val:
                best_val = va; best = (te, rho)
    else:
        u_ext, y = narma30(s) if task == "narma30" else load_phys(task)
        T = len(y)
        wash, ntest = SPLITS[task]
        best = (np.inf, None); best_val = np.inf
        nval = NVAL[task]
        for rho in RADII[task]:
            F = bi_states_seq(u_ext, T, N_f, N_b, taus, rho, s)
            Fb = np.hstack([F, np.ones((len(F), 1), np.float32)]).astype(np.float64)
            yv = np.asarray(y, np.float64)
            Ftr, ytr2 = Fb[wash:-ntest], yv[wash:-ntest]
            Fte, yte2 = Fb[-ntest:], yv[-ntest:]
            # validation : readout ajuste avant le segment de validation
            Fa, ya = Ftr[:-nval], ytr2[:-nval]
            Wv = np.linalg.solve(Fa.T @ Fa + 1e-8 * np.eye(Fa.shape[1]), Fa.T @ ya)
            va = float(np.sqrt(np.mean((Ftr[-nval:] @ Wv - ytr2[-nval:]) ** 2)) / np.std(ytr2[-nval:]))
            A = Ftr.T @ Ftr + 1e-8 * np.eye(Ftr.shape[1])
            Wout = np.linalg.solve(A, Ftr.T @ ytr2)
            te = float(np.sqrt(np.mean((Fte @ Wout - yte2) ** 2)) / np.std(yte2))
            if va < best_val:
                best_val = va; best = (te, rho)
    cell["seeds"][str(s)] = {"test": best[0], "rho": best[1], "val": best_val, "proto": "rho_sur_validation"}
    json.dump(results, open(RJ, "w", encoding="utf-8"), indent=1)
    print(f"{config}/{task} N=({N_f},{N_b}) x{len(taus)}tau seed{s} rho={best[1]} test={best[0]:.4f} ({time.time()-t0:.0f}s)", flush=True)

tests = [cell["seeds"][str(s)]["test"] for s in SEEDS if str(s) in cell["seeds"] and cell["seeds"][str(s)].get("proto") == "rho_sur_validation"]
if len(tests) == len(SEEDS):
    cell["test_mean"] = float(np.mean(tests)); cell["test_std"] = float(np.std(tests))
    json.dump(results, open(RJ, "w", encoding="utf-8"), indent=1)
    print(f"{config}/{task} COMPLETE: {np.mean(tests):.4f} +/- {np.std(tests):.4f}")
else:
    print("pas encore complet")
