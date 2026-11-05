"""2.10ter : Couple v2 : les retards multiples DANS la recurrence (cf. 2.10bis
ou la recurrence monopas perdait la memoire multi-retards du tenseur elu).
Deux variantes, budget recurrent total 400k (632^2) :
  mode "dir" (4K matrices, N = sqrt(400k/(4K))) :
    x+(t) = tanh(Win u(t) + somme_tau W+_tau x(t-tau))   (tenseur K-tranches, rayons rho/sqrt(K))
    x-(t) = tanh(Win u(t) + somme_tau W-_tau x(t+tau))   (miroir)
    x(t)  = tanh(Win u(t) + somme_tau Wa_tau x+(t-tau) + Wb_tau x-(t+tau))  (2K tranches, rho/sqrt(2K))
  mode "flat" (2K matrices, N = sqrt(400k/(2K))) :
    x(t) = tanh(Win u(t) + somme_tau Wa_tau x(t-tau) + Wb_tau x(t+tau))
    : le tenseur D acausal canonique, un seul etat, recurrence signee.
Point fixe global par balayages Jacobi relaxes (warm start = passes tenseur
independantes, puis max_sweeps balayages ; delta = max|dx| enregistre).
Readout : "x" seul ; "xpm" = [x+;x-;x] (dir) ou points de lecture repartis
[x(0);x(9);x(18);x(27)] (flat MNIST, un seul point ne couvre pas l'image).
Configs : cpl2_dir_uni / cpl2_dir_elected / cpl2_flat_uni / cpl2_flat_elected.
Protocole std, 3 seeds, meilleur rho par seed et par variante.
Checkpoint par rho : part2/results_bidir_cpl2.json.
Protocole corrige (14.2, 2026-11-02) : rho choisi sur la VALIDATION (NVAL[task] pas avant le
test ; mnist : 1000 dernieres images du train), readout du test reajuste sur train+validation ;
chaque rec porte val_x/val_xpm ; les recs sans val (ancien choix sur le test) sont recalculees."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from chemins import BRUT  # noqa: E402
import numpy as np, json, time, sys, gzip, os
from pathlib import Path
from scipy.io import loadmat

OUT = (BRUT / "tensor-suite/part2")
RJ = OUT / "results_bidir_cpl2.json"
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

def coupled2_seq(u_ext, N, taus, rho, seed, mode, max_sweeps, damp=0.7, tol=1e-4):
    """Point fixe du couple v2. Retourne (x, xp, xm, sweeps, delta)."""
    K = len(taus)
    rng = np.random.default_rng(seed + 7)
    Win = rng.uniform(-0.5, 0.5, (u_ext.shape[1], N)).astype(np.float32)
    if mode == "dir":
        Wp = np.stack([rand_scaled(rng, N, rho / np.sqrt(K)) for _ in range(K)])
        Wm = np.stack([rand_scaled(rng, N, rho / np.sqrt(K)) for _ in range(K)])
        Wa = np.stack([rand_scaled(rng, N, rho / np.sqrt(2 * K)) for _ in range(K)])
        Wb = np.stack([rand_scaled(rng, N, rho / np.sqrt(2 * K)) for _ in range(K)])
        WpT, WmT = np.transpose(Wp, (0, 2, 1)), np.transpose(Wm, (0, 2, 1))
        WaT, WbT = np.transpose(Wa, (0, 2, 1)), np.transpose(Wb, (0, 2, 1))
    else:  # flat
        Wa = np.stack([rand_scaled(rng, N, rho / np.sqrt(2 * K)) for _ in range(K)])
        Wb = np.stack([rand_scaled(rng, N, rho / np.sqrt(2 * K)) for _ in range(K)])
        WaT, WbT = np.transpose(Wa, (0, 2, 1)), np.transpose(Wb, (0, 2, 1))
        WpT = WmT = None
    Text = len(u_ext)
    uW = (u_ext @ Win).astype(np.float32)
    shifts = np.array(taus)

    def shiftsum(states, WT_, direction):
        """Somme des tranches sur states[t - tau] (direction=-1) ou [t + tau] (+1)."""
        acc = np.zeros_like(uW)
        for i, tau in enumerate(shifts):
            if direction < 0:
                acc[tau:] += states[:Text-tau] @ WT_[i]
            else:
                acc[:Text-tau] += states[tau:] @ WT_[i]
        return acc

    def tensor_pass_own(WT_, backward=False):
        """Passe causale (ou miroir) sur propre histoire (warm start)."""
        F = np.empty((Text, N), np.float32)
        x_ = np.zeros(N, np.float32)
        hist = np.zeros((int(shifts.max()), N), np.float32)
        sel = shifts - 1
        for t in range(Text):
            x_ = np.tanh(uW[t] + np.einsum("kij,kj->i", WT_, hist[sel]))
            F[t] = x_
            hist = np.concatenate([x_[None], hist[:-1]], axis=0)
        return F

    if mode == "dir":
        xp = tensor_pass_own(WpT)
        xm = tensor_pass_own(WmT)[::-1]
        x = np.tanh(uW + shiftsum(xp, WaT, -1) + shiftsum(xm, WbT, +1))
    else:
        x = tensor_pass_own(WaT)
        xp = xm = None
    sweeps, delta = 0, 1.0
    for s in range(max_sweeps):
        if mode == "dir":
            xp_new = np.tanh(uW + shiftsum(x, WpT, -1))
            xm_new = np.tanh(uW + shiftsum(x, WmT, +1))
            x_new = np.tanh(uW + shiftsum(xp_new, WaT, -1) + shiftsum(xm_new, WbT, +1))
        else:
            xp_new = xm_new = None
            x_new = np.tanh(uW + shiftsum(x, WaT, -1) + shiftsum(x, WbT, +1))
        delta = float(np.max(np.abs(x_new - x)))
        x = x + damp * (x_new - x)
        sweeps = s + 1
        if delta < tol:
            xp, xm = xp_new, xm_new
            break
    # etats finaux recalculues depuis le x final
    if mode == "dir":
        xp = np.tanh(uW + shiftsum(x, WpT, -1))
        xm = np.tanh(uW + shiftsum(x, WmT, +1))
    return x, xp, xm, sweeps, delta

def coupled2_mnist(Xb, N, taus, rho, seed, mode, max_sweeps, damp=0.7, tol=1e-4, chunk=1000):
    """Retourne par image : features (dir : [xp(27),xm(0),x(27)] ; flat : [x0,x9,x18,x27]) et x(27)."""
    K = len(taus)
    out = np.empty((len(Xb), 3 * N if mode == "dir" else 4 * N), np.float32)
    outx = np.empty((len(Xb), N), np.float32)
    all_sw, all_de = [], []
    for lo in range(0, len(Xb), chunk):
        B = Xb[lo:lo+chunk]
        rng = np.random.default_rng(seed + 7)
        Win = rng.uniform(-0.5, 0.5, (28, N)).astype(np.float32)
        if mode == "dir":
            Wp = np.stack([rand_scaled(rng, N, rho / np.sqrt(K)) for _ in range(K)])
            Wm = np.stack([rand_scaled(rng, N, rho / np.sqrt(K)) for _ in range(K)])
            Wa = np.stack([rand_scaled(rng, N, rho / np.sqrt(2 * K)) for _ in range(K)])
            Wb = np.stack([rand_scaled(rng, N, rho / np.sqrt(2 * K)) for _ in range(K)])
            WpT, WmT = np.transpose(Wp, (0, 2, 1)), np.transpose(Wm, (0, 2, 1))
            WaT, WbT = np.transpose(Wa, (0, 2, 1)), np.transpose(Wb, (0, 2, 1))
        else:
            Wa = np.stack([rand_scaled(rng, N, rho / np.sqrt(2 * K)) for _ in range(K)])
            Wb = np.stack([rand_scaled(rng, N, rho / np.sqrt(2 * K)) for _ in range(K)])
            WaT, WbT = np.transpose(Wa, (0, 2, 1)), np.transpose(Wb, (0, 2, 1))
        nb = len(B)
        uW = np.transpose((B @ Win).astype(np.float32), (1, 0, 2))   # (28, nb, N)
        shifts = np.array(taus)

        def shiftsum(states, WT_, direction):
            acc = np.zeros_like(uW)
            for i, tau in enumerate(shifts):
                if tau >= 28:
                    continue
                if direction < 0:
                    acc[tau:] += states[:28-tau] @ WT_[i]
                else:
                    acc[:28-tau] += states[tau:] @ WT_[i]
            return acc

        def tensor_pass_own(WT_):
            F = np.empty((28, nb, N), np.float32)
            x_ = np.zeros((nb, N), np.float32)
            hist = np.zeros((int(shifts.max()), nb, N), np.float32)
            sel = shifts - 1
            for t in range(28):
                x_ = np.tanh(uW[t] + np.tensordot(hist[sel], WT_, axes=([0, 2], [0, 2])))
                F[t] = x_
                hist = np.concatenate([x_[None], hist[:-1]], axis=0)
            return F

        if mode == "dir":
            xp = tensor_pass_own(WpT)
            xm = tensor_pass_own(WmT)[::-1]
            x = np.tanh(uW + shiftsum(xp, WaT, -1) + shiftsum(xm, WbT, +1))
        else:
            x = tensor_pass_own(WaT)
        sweeps, delta = 0, 1.0
        for s in range(max_sweeps):
            if mode == "dir":
                xp_new = np.tanh(uW + shiftsum(x, WpT, -1))
                xm_new = np.tanh(uW + shiftsum(x, WmT, +1))
                x_new = np.tanh(uW + shiftsum(xp_new, WaT, -1) + shiftsum(xm_new, WbT, +1))
            else:
                x_new = np.tanh(uW + shiftsum(x, WaT, -1) + shiftsum(x, WbT, +1))
            delta = float(np.max(np.abs(x_new - x)))
            x = x + damp * (x_new - x)
            sweeps = s + 1
            if delta < tol:
                break
        if mode == "dir":
            xp = np.tanh(uW + shiftsum(x, WpT, -1))
            xm = np.tanh(uW + shiftsum(x, WmT, +1))
            out[lo:lo+nb] = np.hstack([xp[27], xm[0], x[27]])
        else:
            out[lo:lo+nb] = np.hstack([x[0], x[9], x[18], x[27]])
        outx[lo:lo+nb] = x[27]
        all_sw.append(sweeps); all_de.append(delta)
    return out, outx, max(all_sw), max(all_de)

RADII = {"narma30": [0.5, 0.9, 0.99], "silverbox": [0.5, 0.9], "wh": [0.5, 0.9], "mnist": [0.9, 0.99]}
SPLITS = {"narma30": (200, 2000), "silverbox": (500, 10000), "wh": (500, 10000)}
MAXSWEEPS = {"narma30": 12, "silverbox": 10, "wh": 10, "mnist": 6}

task, config = sys.argv[1], sys.argv[2]
mode = "dir" if "_dir_" in config else "flat"
taus = list(range(1, 11)) if config.endswith("_uni") else ELECTED[task]
K = len(taus)
per = (4 if mode == "dir" else 2) * K
N = int((BUDGET / per) ** 0.5)
cell = results.setdefault(config, {}).setdefault(
    task, {"mode": mode, "taus": taus, "N": N, "budget": per * N * N,
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
            Ftr, Ftrx, sw, de = coupled2_mnist(Xtr, N, taus, rho, s, mode, MAXSWEEPS[task])
            Fte, Ftex, _, _ = coupled2_mnist(Xte, N, taus, rho, s, mode, MAXSWEEPS[task])
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
            recs.append({"rho": rho, "val_x": vx, "test_x": tx, "val_xpm": vp, "test_xpm": tp,
                         "sweeps": sw, "delta": de})
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
            x, xp, xm, sw, de = coupled2_seq(u_ext, N, taus, rho, s, mode, MAXSWEEPS[task])
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
            Fpm = np.hstack([xp[:T], xm[:T], x[:T]]) if mode == "dir" else None
            vx, tx = nrmse(Fx)
            vp, tp = nrmse(Fpm) if mode == "dir" else (None, None)
            recs.append({"rho": rho, "val_x": vx, "test_x": tx, "val_xpm": vp, "test_xpm": tp,
                         "sweeps": sw, "delta": de})
            cell["seeds"][str(s)] = recs
            json.dump(results, open(RJ, "w", encoding="utf-8"), indent=1)
            print(f"  rho={rho} x={recs[-1]['test_x']:.4f}"
                  + (f" xpm={recs[-1]['test_xpm']:.4f}" if mode == "dir" else "")
                  + f" sweeps={sw} delta={de:.1e} ({time.time()-t0:.0f}s)", flush=True)
            if time.time() - t0 > TIME_LIM:
                print("time budget : reprendre", flush=True)
                break

full = [s for s in SEEDS if len(cell["seeds"].get(str(s), [])) == NR]
if len(full) == len(SEEDS):
    for var in ("test_x", "test_xpm"):
        if var == "test_xpm" and mode != "dir" and task != "mnist":
            continue
        vk = "val_" + var[5:]
        recsl = [[r for r in cell["seeds"][str(s)] if r.get(vk) is not None] for s in SEEDS]
        if any(len(v) == 0 for v in recsl):
            continue
        # 14.2 : rho choisi sur la validation, on rapporte le test a ce rho
        pick = (max if task == "mnist" else min)
        bests = [pick(v, key=lambda r: r[vk])[var] for v in recsl]
        cell[f"{var}_mean"] = float(np.mean(bests)); cell[f"{var}_std"] = float(np.std(bests))
    json.dump(results, open(RJ, "w", encoding="utf-8"), indent=1)
    msg = f"{config}/{task} COMPLETE x={cell.get('test_x_mean', float('nan')):.4f}"
    if "test_xpm_mean" in cell:
        msg += f" xpm={cell['test_xpm_mean']:.4f}"
    print(msg)
else:
    print(f"pas encore complet ({len(full)}/3 seeds)")
