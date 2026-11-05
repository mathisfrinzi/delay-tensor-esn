"""Partie 2.4, RAFFINEMENT CONTINU LOCAL : la section doree globale depuis (1,10)
(band_tau_cont.py) a montre un paysage multimodal : elle tombe dans un autre bassin
que la grille grossiere (band_tau.py) selon la tache. Protocole final :
1. depart = meilleur des deux bandes connues (grille vs section doree globale),
   a la validation (seed 0) ;
2. raffinement continu par section doree dans une fenetre locale
   (tau0 +/- 0.75, tau1 +/- 1.5, fenetre etendue une fois si l'optimum touche un bord),
   2 passes, precision ~0.02-0.05 ;
3. bande FINALE = argmin(val) parmi {grille, globale, raffinee} ;
   evaluation standard 3 seeds x balayage rho de la bande finale.
Checkpoints lus : results_band.json (grille), results_band_cont.json (globale).
Checkpoint ecrit : part2/results_band_refine.json."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from chemins import BRUT  # noqa: E402
import numpy as np, json, time, sys, gzip, os
from pathlib import Path
from scipy.io import loadmat

OUT = (BRUT / "tensor-suite/part2")
RJ = OUT / "results_band_refine.json"
results = json.load(open(RJ)) if RJ.exists() else {}

TMAX = 20
N = 200
TIME_BUDGET = 170
INV_PHI = (np.sqrt(5) - 1) / 2  # 0.618

def gains(tau0, tau1):
    ks = np.arange(1, TMAX + 1)
    return np.clip(ks - tau0 + 1, 0, 1) * np.clip(tau1 - ks + 1, 0, 1)

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
    BASE = "/tmp/mnist"
    files = ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
             "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]
    def imgs(fn):
        with gzip.open(os.path.join(BASE, fn), "rb") as f:
            dd = np.frombuffer(f.read(), np.uint8, offset=16)
        return dd.reshape(-1, 28, 28).astype(np.float32) / 255.0
    def labs(fn):
        with gzip.open(os.path.join(BASE, fn), "rb") as f:
            return np.frombuffer(f.read(), np.uint8, offset=8).astype(np.int64)
    return imgs(files[0])[:5000], labs(files[1])[:5000], imgs(files[2])[:1000], labs(files[3])[:1000]

def make_W(seed, g, rho):
    act = np.where(g > 0)[0]
    sn = np.sqrt((g ** 2).sum()) if (g ** 2).sum() > 0 else 1.0
    rng = np.random.default_rng(seed + 7)
    Ws, idx = [], []
    for i in act:
        M = rng.normal(0, 1, (N, N)).astype(np.float32)
        r = max(np.abs(np.linalg.eigvals(M.astype(np.float64))).max(), 1e-12)
        Ws.append((M * (rho * g[i] / sn / r)).astype(np.float32))
        idx.append(int(i))
    return np.stack(Ws), np.array(idx)

def states_seq(u, W, idx, Win):
    T = u.shape[0]
    hist = np.zeros((TMAX, N), np.float32)
    x_ = np.zeros(N, np.float32)
    F = np.empty((T, N), np.float32)
    for t in range(T):
        x_ = np.tanh(u[t] @ Win + np.einsum("kij,kj->i", W, hist[idx]))
        hist = np.concatenate([x_[None], hist[:-1]], axis=0)
        F[t] = x_
    return F

def states_mnist(Xb, W, idx, Win, chunk=1000):
    out = np.empty((len(Xb), N), np.float32)
    for lo in range(0, len(Xb), chunk):
        B = Xb[lo:lo+chunk]
        hist = np.zeros((TMAX, len(B), N), np.float32)
        x = np.zeros((len(B), N), np.float32)
        for t in range(28):
            drive = B[:, t, :] @ Win + np.tensordot(hist[idx], W, axes=([0, 2], [0, 2]))
            x = np.tanh(drive)
            hist = np.concatenate([x[None], hist[:-1]], axis=0)
        out[lo:lo+len(B)] = x
    return out

task = sys.argv[1]
R = results.setdefault(task, {"history": [], "cache": {}, "done": False})

if task == "narma30":
    u, y = narma30(0)
    tr_sl, va_sl, te_sl = slice(200, 3200), slice(3200, 4200), slice(4200, 5200)
elif task in ("silverbox", "wh"):
    u, y = load_phys(task)
    tr_sl, va_sl, te_sl = slice(500, 15000), slice(15000, 20000), slice(20000, 30000)
else:
    Xtr, ytr, Xte, yte = get_mnist()

def evaluate(tau0, tau1, seed=0, rho=0.9, want="val"):
    key = "|".join(map(str, (round(float(tau0), 4), round(float(tau1), 4), seed, rho, want)))
    if key in R["cache"]:
        return tuple(R["cache"][key])
    g = gains(tau0, tau1)
    W, idx = make_W(seed, g, rho)
    rng = np.random.default_rng(7 + 100 * seed)
    if task != "mnist":
        Win = rng.uniform(-0.5, 0.5, size=(u.shape[1], N)).astype(np.float32)
        F = states_seq(u, W, idx, Win)
        Fb = np.hstack([F, np.ones((len(F), 1), np.float32)]).astype(np.float64)
        yv = np.asarray(y, np.float64)
        A = Fb[tr_sl].T @ Fb[tr_sl] + 1e-8 * np.eye(Fb.shape[1])
        Wout = np.linalg.solve(A, Fb[tr_sl].T @ yv[tr_sl])
        va = float(np.sqrt(np.mean((Fb[va_sl] @ Wout - yv[va_sl]) ** 2)) / np.std(yv[te_sl]))
        te = float(np.sqrt(np.mean((Fb[te_sl] @ Wout - yv[te_sl]) ** 2)) / np.std(yv[te_sl]))
        out = (va, te)
    else:
        Win = rng.uniform(-0.5, 0.5, size=(28, N)).astype(np.float32)
        Ftr = states_mnist(Xtr[:4000], W, idx, Win)
        Fva = states_mnist(Xtr[4000:], W, idx, Win)
        Fte = states_mnist(Xte, W, idx, Win)
        def pre(F):
            return np.hstack([F, np.ones((len(F), 1), np.float32)]).astype(np.float64)
        Tt = np.zeros((4000, 10)); Tt[np.arange(4000), ytr[:4000]] = 1.0
        A = pre(Ftr).T @ pre(Ftr) + 1e-6 * np.eye(pre(Ftr).shape[1])
        Wout = np.linalg.solve(A, pre(Ftr).T @ Tt)
        va = float((np.argmax(pre(Fva) @ Wout, axis=1) == ytr[4000:]).mean())
        te = float((np.argmax(pre(Fte) @ Wout, axis=1) == yte).mean())
        out = (va, te)
    R["cache"][key] = out
    return out

t0 = time.time()
MAXIM = task == "mnist"

def golden(f, a, b, iters):
    """Section doree : minimise f sur [a,b] (ou maximise si MAXIM)."""
    c, d = b - INV_PHI * (b - a), a + INV_PHI * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(iters):
        if (fc < fd) != MAXIM:  # min : garder [a,d] ; max : garder [c,b]
            b, d, fd = d, c, fc
            c = b - INV_PHI * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + INV_PHI * (b - a)
            fd = f(d)
    return (a + b) / 2

def save():
    # cache limite pour ne pas exploser le JSON
    if len(R["cache"]) > 400:
        R["cache"] = dict(list(R["cache"].items())[-200:])
    json.dump(results, open(RJ, "w"), indent=1)

# bandes candidates connues (grille + section doree globale)
GRID = json.load(open(OUT / "results_band.json"))
CONT = json.load(open(OUT / "results_band_cont.json"))
MAXIM = task == "mnist"

gb = GRID[task]["learned_band"]; cb = CONT[task]["learned_band"]
cands = {"grille": (float(gb["tau0"]), float(gb["tau1"])),
         "globale": (float(cb["tau0"]), float(cb["tau1"]))}
vals = {k: evaluate(*v)[0] for k, v in cands.items()}

# depart du raffinement : meilleur bassin connu a la validation
if "start" not in R:
    bk = (max if MAXIM else min)(vals, key=vals.get)
    R["start"] = {"which": bk, "tau0": cands[bk][0], "tau1": cands[bk][1], "val": vals[bk]}
    save()
    print(f"{task} depart raffinement = bande {bk} {cands[bk]} val={vals[bk]:.4f}", flush=True)

tau0, tau1 = R.get("cur", [R["start"]["tau0"], R["start"]["tau1"]])
best_val = evaluate(tau0, tau1)[0]

# 2 passes : section doree locale, fenetre etendue si un bord est touche
for p in range(R.get("passes_done", 0), 2):
    if R.get("done") or time.time() - t0 > TIME_BUDGET:
        break
    # tau0 : fenetre locale +/-0.75 (1.5 apres extension)
    ext = R.get("ext_tau0", False)
    half = 1.5 if ext else 0.75
    lo, hi = max(1.0, tau0 - half), min(tau1 - 1.0, tau0 + half)
    t0n = golden(lambda x: evaluate(x, tau1)[0], lo, hi, 12)
    v0n = evaluate(t0n, tau1)[0]
    if (MAXIM and v0n > best_val) or (not MAXIM and v0n < best_val):
        tau0, best_val = t0n, v0n
    if not ext and (abs(t0n - lo) < 0.02 or abs(t0n - hi) < 0.02):
        R["ext_tau0"] = True  # optimum au bord -> elargir a la passe suivante
    R["cur"] = [tau0, tau1]
    R["history"].append({"pass": p, "coord": "tau0", "tau0": tau0, "tau1": tau1, "val": best_val})
    save()
    print(f"{task} pass{p} tau0 -> ({tau0:.3f}, {tau1:.3f}) val={best_val:.4f} ({time.time()-t0:.0f}s)", flush=True)
    if time.time() - t0 > TIME_BUDGET:
        break
    # tau1 : fenetre locale +/-1.5 (3.0 apres extension)
    ext = R.get("ext_tau1", False)
    half = 3.0 if ext else 1.5
    lo, hi = max(tau0 + 1.0, tau1 - half), min(20.5, tau1 + half)
    t1n = golden(lambda x: evaluate(tau0, x)[0], lo, hi, 12)
    v1n = evaluate(tau0, t1n)[0]
    if (MAXIM and v1n > best_val) or (not MAXIM and v1n < best_val):
        tau1, best_val = t1n, v1n
    if not ext and (abs(t1n - lo) < 0.02 or abs(t1n - hi) < 0.02):
        R["ext_tau1"] = True
    R["cur"] = [tau0, tau1]
    R["history"].append({"pass": p, "coord": "tau1", "tau0": tau0, "tau1": tau1, "val": best_val})
    R["passes_done"] = p + 1
    save()
    print(f"{task} pass{p} tau1 -> ({tau0:.3f}, {tau1:.3f}) val={best_val:.4f} ({time.time()-t0:.0f}s)", flush=True)

# bande FINALE : argmin(val) parmi {grille, globale, raffinee} + eval standard
if R.get("passes_done", 0) >= 2 or R.get("done"):
    cands["raffinee"] = (round(float(tau0), 3), round(float(tau1), 3))
    vals["raffinee"] = best_val
    fk = (max if MAXIM else min)(vals, key=vals.get)
    f0, f1 = cands[fk]
    R["final_band"] = {"which": fk, "tau0": f0, "tau1": f1, "val": vals[fk],
                       "all_vals": {k: round(v, 5) for k, v in vals.items()}}
    RADII = [0.9, 0.99] if task == "mnist" else [0.5, 0.9, 0.99]
    final = []
    for seed in (0, 1, 2):
        best = None
        for rho in RADII:
            va, te = evaluate(f0, f1, seed=seed, rho=rho)
            sc = va if task == "mnist" else -va
            if best is None or sc > best[0]:
                best = (sc, te, rho)
        final.append({"seed": seed, "rho": best[2], "test": best[1]})
        print(f"{task} final seed{seed} rho={best[2]} test={best[1]:.4f}", flush=True)

    tests = [f["test"] for f in final]
    R["final_band"]["final"] = final
    R["final_band"]["test_mean"] = float(np.mean(tests))
    R["final_band"]["test_std"] = float(np.std(tests))
    R["done"] = True
    save()
    print(f"{task} BANDE FINALE ({fk}): [{f0}, {f1}] test={np.mean(tests):.4f} +/- {np.std(tests):.4f}")
else:
    print(f"{task} budget epuise, passes={R.get('passes_done', 0)}/2 : RELANCER pour continuer")
