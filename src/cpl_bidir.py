"""2.10bis : Tenseur D VRAIMENT acausal et couple (proposition M. Frinzi) :
  x+(t) = tanh(Win u(t) + W  x(t-1))          (pas unique, W partage)
  x-(t) = tanh(Win u(t) + W  x(t+1))          (miroir temporel, meme W)
  x(t)  = tanh(Win u(t) + somme_{tau in D} W_-tau x+(t-tau) + W_+tau x-(t+tau))
Les deux directions se parlent via l'etat fusionne x : le systeme couple tous
les instants de la sequence -> point fixe global, resolu par balayages Jacobi
relaxes (warm start = deux passes ESN une-etape independantes, puis
max_sweeps balayages vectorises, arret si max|dx| < tol).
Rayons : W au rayon spectal rho (comme un ESN) ; les 2K tranches signees a
rho/sqrt(2K) chacune (comme un tenseur 2K-tranches a budget spectral rho^2).
Budget recurrent = N^2 (1 + 2K) = 400k (632^2) : N = floor(sqrt(400k/(1+2K))).
Readout ridge sur DEUX variantes : "x" seul (l'etat de l'architecture) et
"xpm" = [x+ ; x- ; x] concatene. Meilleur rho par seed pour chaque variante.
Configs : cpl_uni (D = {1..10}) et cpl_elected (elus 2.2). Protocole std,
3 seeds. Checkpoint : part2/results_bidir_cpl.json.
Protocole corrige (14.2, 2026-11-02) : rho choisi sur la VALIDATION (NVAL[task] pas avant le
test ; mnist : 1000 dernieres images du train), readout du test reajuste sur train+validation ;
chaque rec porte val_x/val_xpm ; les recs sans val (ancien choix sur le test) sont recalculees."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from chemins import BRUT  # noqa: E402
import numpy as np, json, time, sys, gzip, os
from pathlib import Path
from scipy.io import loadmat

OUT = (BRUT / "tensor-suite/part2")
RJ = OUT / "results_bidir_cpl.json"
results = json.load(open(RJ, encoding="utf-8-sig")) if RJ.exists() else {}
SEEDS = [0, 1, 2]
TIME_LIM = float(os.environ.get("TEMPS_MAX", "240"))
NVAL = {"narma30": 1000, "silverbox": 2000, "wh": 2000}   # validation : choix de rho (14.2)
NVAL_MNIST = 1000
BUDGET = 632 ** 2

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
    U = u[:30000 + ext] / np.std(u[:30000])
    Y = (y[:30000] / s).astype(np.float32)
    return np.stack([U, np.ones_like(U)], 1).astype(np.float32), Y

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

def rand_scaled(rng, N, radius):
    M = rng.normal(0, 1, (N, N)).astype(np.float32)
    r = max(np.abs(np.linalg.eigvals(M.astype(np.float64))).max(), 1e-12)
    return M * (radius / r)

def coupled_seq(u_ext, N, taus, rho, seed, max_sweeps, damp=0.7, tol=1e-4):
    """Retourne (x, xp, xm, sweeps, delta) sur la fenetre etendue."""
    K = len(taus)
    rng = np.random.default_rng(seed + 7)
    Win = rng.uniform(-0.5, 0.5, (u_ext.shape[1], N)).astype(np.float32)
    W1 = rand_scaled(rng, N, rho)                       # pas unique, partage
    Wp = np.stack([rand_scaled(rng, N, rho / np.sqrt(2 * K)) for _ in range(K)])
    Wm = np.stack([rand_scaled(rng, N, rho / np.sqrt(2 * K)) for _ in range(K)])
    W1T, WpT, WmT = W1.T, np.transpose(Wp, (0, 2, 1)), np.transpose(Wm, (0, 2, 1))
    Text = len(u_ext)
    uW = (u_ext @ Win).astype(np.float32)
    # warm start : deux passes ESN une-etape independantes
    xp = np.empty((Text, N), np.float32); xp[0] = np.tanh(uW[0])
    for t in range(1, Text):
        xp[t] = np.tanh(uW[t] + xp[t-1] @ W1T)
    xm = np.empty((Text, N), np.float32); xm[-1] = np.tanh(uW[-1])
    for t in range(Text - 2, -1, -1):
        xm[t] = np.tanh(uW[t] + xm[t+1] @ W1T)
    def eq3(xp_, xm_):
        acc = uW.copy()
        for i, tau in enumerate(taus):
            acc[tau:] += xp_[:Text-tau] @ WpT[i]
            acc[:Text-tau] += xm_[tau:] @ WmT[i]
        return np.tanh(acc)
    x = eq3(xp, xm)
    sweeps, delta = 0, 1.0
    for s in range(max_sweeps):
        xp_new = np.empty_like(xp); xp_new[0] = np.tanh(uW[0])
        xp_new[1:] = np.tanh(uW[1:] + x[:-1] @ W1T)
        xm_new = np.empty_like(xm); xm_new[-1] = np.tanh(uW[-1])
        xm_new[:-1] = np.tanh(uW[:-1] + x[1:] @ W1T)
        x_new = eq3(xp_new, xm_new)
        delta = float(np.max(np.abs(x_new - x)))
        x = x + damp * (x_new - x)
        sweeps = s + 1
        if delta < tol:
            xp, xm = xp_new, xm_new
            break
    else:
        pass
    # directions finales recalculees depuis le x final (cas converge ou non)
    xp = np.empty_like(xp); xp[0] = np.tanh(uW[0])
    xp[1:] = np.tanh(uW[1:] + x[:-1] @ W1T)
    xm = np.empty_like(xm); xm[-1] = np.tanh(uW[-1])
    xm[:-1] = np.tanh(uW[:-1] + x[1:] @ W1T)
    return x, xp, xm, sweeps, delta

def coupled_mnist(Xb, N, taus, rho, seed, max_sweeps, damp=0.7, tol=1e-4, chunk=1000):
    """Xb : (B, 28, 28). Retourne features par image : [xp(27), xm(0), x(27)] et x(27)."""
    K = len(taus)
    out = np.empty((len(Xb), 3 * N), np.float32)
    outx = np.empty((len(Xb), N), np.float32)
    all_sweeps, all_delta = [], []
    for lo in range(0, len(Xb), chunk):
        B = Xb[lo:lo+chunk]
        rng = np.random.default_rng(seed + 7)
        Win = rng.uniform(-0.5, 0.5, (28, N)).astype(np.float32)
        W1 = rand_scaled(rng, N, rho)
        Wp = np.stack([rand_scaled(rng, N, rho / np.sqrt(2 * K)) for _ in range(K)])
        Wm = np.stack([rand_scaled(rng, N, rho / np.sqrt(2 * K)) for _ in range(K)])
        W1T, WpT, WmT = W1.T, np.transpose(Wp, (0, 2, 1)), np.transpose(Wm, (0, 2, 1))
        nb = len(B)
        uW = np.transpose((B @ Win).astype(np.float32), (1, 0, 2))   # (28, nb, N)
        # warm start
        xp = np.empty((28, nb, N), np.float32); xp[0] = np.tanh(uW[0])
        for t in range(1, 28):
            xp[t] = np.tanh(uW[t] + xp[t-1] @ W1T)
        xm = np.empty((28, nb, N), np.float32); xm[27] = np.tanh(uW[27])
        for t in range(26, -1, -1):
            xm[t] = np.tanh(uW[t] + xm[t+1] @ W1T)
        def eq3(xp_, xm_):
            acc = uW.copy()
            for i, tau in enumerate(taus):
                if tau < 28:
                    acc[tau:] += xp_[:28-tau] @ WpT[i]
                    acc[:28-tau] += xm_[tau:] @ WmT[i]
            return np.tanh(acc)
        x = eq3(xp, xm)
        sweeps, delta = 0, 1.0
        for s in range(max_sweeps):
            xp_new = np.empty_like(xp); xp_new[0] = np.tanh(uW[0])
            xp_new[1:] = np.tanh(uW[1:] + x[:-1] @ W1T)
            xm_new = np.empty_like(xm); xm_new[27] = np.tanh(uW[27])
            xm_new[:27] = np.tanh(uW[:27] + x[1:] @ W1T)
            x_new = eq3(xp_new, xm_new)
            delta = float(np.max(np.abs(x_new - x)))
            x = x + damp * (x_new - x)
            sweeps = s + 1
            if delta < tol:
                xp, xm = xp_new, xm_new
                break
        else:
            xp = np.empty_like(xp); xp[0] = np.tanh(uW[0])
            xp[1:] = np.tanh(uW[1:] + x[:-1] @ W1T)
            xm = np.empty_like(xm); xm[27] = np.tanh(uW[27])
            xm[:27] = np.tanh(uW[:27] + x[1:] @ W1T)
        out[lo:lo+nb] = np.hstack([xp[27], xm[0], x[27]])
        outx[lo:lo+nb] = x[27]
        all_sweeps.append(sweeps); all_delta.append(delta)
    return out, outx, max(all_sweeps), max(all_delta)

RADII = {"narma30": [0.5, 0.9, 0.99], "silverbox": [0.5, 0.9], "wh": [0.5, 0.9], "mnist": [0.9, 0.99]}
SPLITS = {"narma30": (200, 2000), "silverbox": (500, 10000), "wh": (500, 10000)}
# balayages apres warm start (le warm start integre deja toute la sequence ;
# les balayages reconcilient seulement le couplage, delta enregistre)
MAXSWEEPS = {"narma30": 20, "silverbox": 10, "wh": 10, "mnist": 5}

task, config = sys.argv[1], sys.argv[2]
taus = list(range(1, 11)) if config == "cpl_uni" else ELECTED[task]
K = len(taus)
N = int((BUDGET / (1 + 2 * K)) ** 0.5)
cell = results.setdefault(config, {}).setdefault(
    task, {"taus": taus, "N": N, "budget": (1 + 2*K) * N * N,
           "max_sweeps": MAXSWEEPS[task], "seeds": {}})

if task == "mnist":
    Xtr, ytr, Xte, yte = get_mnist()
    Tt = np.zeros((len(ytr), 10)); Tt[np.arange(len(ytr)), ytr] = 1.0
t0 = time.time()
NR = len(RADII[task])
for s in SEEDS:
    recs = [r for r in cell["seeds"].setdefault(str(s), []) if "val_x" in r]   # 14.2 : anciennes recs (rho choisi sur le test) recalculees
    cell["seeds"][str(s)] = recs
    if len(recs) == NR:
        continue
    if time.time() - t0 > TIME_LIM:
        print("time budget : reprendre", flush=True)
        break
    if task == "mnist":
        for rho in RADII[task][len(recs):]:
            Ftr, Ftrx, sw, de = coupled_mnist(Xtr, N, taus, rho, s, MAXSWEEPS[task])
            Fte, Ftex, _, _ = coupled_mnist(Xte, N, taus, rho, s, MAXSWEEPS[task])
            def acc(Ftr_, Fte_):
                """14.2 : (acc validation, acc test). Validation = NVAL_MNIST dernieres images du train."""
                Ftrb = np.hstack([Ftr_, np.ones((len(Ftr_), 1), np.float32)]).astype(np.float64)
                Fteb = np.hstack([Fte_, np.ones((len(Fte_), 1), np.float32)]).astype(np.float64)
                nfit = len(ytr) - NVAL_MNIST
                Af = Ftrb[:nfit].T @ Ftrb[:nfit] + 1e-6 * np.eye(Ftrb.shape[1])
                Wv = np.linalg.solve(Af, Ftrb[:nfit].T @ Tt[:nfit])
                va = float((np.argmax(Ftrb[nfit:] @ Wv, axis=1) == ytr[nfit:]).mean())
                A = Ftrb.T @ Ftrb + 1e-6 * np.eye(Ftrb.shape[1])
                Wo = np.linalg.solve(A, Ftrb.T @ Tt)
                return va, float((np.argmax(Fteb @ Wo, axis=1) == yte).mean())
            vx, tx = acc(Ftrx, Ftex); vp, tp = acc(Ftr, Fte)
            recs.append({"rho": rho, "val_x": vx, "test_x": tx, "val_xpm": vp, "test_xpm": tp, "sweeps": sw, "delta": de})
            cell["seeds"][str(s)] = recs
            json.dump(results, open(RJ, "w", encoding="utf-8"), indent=1)
            print(f"  rho={rho} x={recs[-1]['test_x']:.4f} xpm={recs[-1]['test_xpm']:.4f} sweeps={sw} delta={de:.1e} ({time.time()-t0:.0f}s)", flush=True)
            if time.time() - t0 > TIME_LIM:
                print("time budget : reprendre", flush=True)
                break
    else:
        u_ext, y = narma30(s) if task == "narma30" else load_phys(task)
        T = len(y)
        wash, ntest = SPLITS[task]
        yv = np.asarray(y, np.float64)
        for rho in RADII[task][len(recs):]:
            x, xp, xm, sw, de = coupled_seq(u_ext, N, taus, rho, s, MAXSWEEPS[task])
            def nrmse(F):
                """14.2 : (nrmse validation, nrmse test). Validation = NVAL[task] pas avant le test."""
                Fb = np.hstack([F, np.ones((len(F), 1), np.float32)]).astype(np.float64)
                Ftr_, ytr_ = Fb[wash:-ntest], yv[wash:-ntest]
                nval = NVAL[task]
                Fa, ya = Ftr_[:-nval], ytr_[:-nval]
                Wv = np.linalg.solve(Fa.T @ Fa + 1e-8 * np.eye(Fa.shape[1]), Fa.T @ ya)
                va = float(np.sqrt(np.mean((Ftr_[-nval:] @ Wv - ytr_[-nval:]) ** 2)) / np.std(ytr_[-nval:]))
                A = Ftr_.T @ Ftr_ + 1e-8 * np.eye(Ftr_.shape[1])
                Wo = np.linalg.solve(A, Ftr_.T @ ytr_)
                Fte_ = Fb[-ntest:]
                return va, float(np.sqrt(np.mean((Fte_ @ Wo - yv[-ntest:]) ** 2)) / np.std(yv[-ntest:]))
            Fx = x[:T]
            Fpm = np.hstack([xp[:T], xm[:T], x[:T]])
            vx, tx = nrmse(Fx); vp, tp = nrmse(Fpm)
            recs.append({"rho": rho, "val_x": vx, "test_x": tx, "val_xpm": vp, "test_xpm": tp, "sweeps": sw, "delta": de})
            cell["seeds"][str(s)] = recs
            json.dump(results, open(RJ, "w", encoding="utf-8"), indent=1)
            print(f"  rho={rho} x={recs[-1]['test_x']:.4f} xpm={recs[-1]['test_xpm']:.4f} sweeps={sw} delta={de:.1e} ({time.time()-t0:.0f}s)", flush=True)
            if time.time() - t0 > TIME_LIM:
                print("time budget : reprendre", flush=True)
                break

full = [s for s in SEEDS if len(cell["seeds"].get(str(s), [])) == NR]
if len(full) == len(SEEDS):
    for var in ("test_x", "test_xpm"):
        vk = "val_" + var[5:]
        pick = (max if task == "mnist" else min)
        # 14.2 : rho choisi sur la validation, on rapporte le test a ce rho
        bests = [pick(cell["seeds"][str(s)], key=lambda r: r[vk])[var] for s in SEEDS]
        cell[f"{var}_mean"] = float(np.mean(bests)); cell[f"{var}_std"] = float(np.std(bests))
    json.dump(results, open(RJ, "w", encoding="utf-8"), indent=1)
    print(f"{config}/{task} COMPLETE x={cell['test_x_mean']:.4f} xpm={cell['test_xpm_mean']:.4f}")
else:
    print(f"pas encore complet ({len(full)}/3 seeds)")
