"""Suite de benchmarks tenseur W(i,j,tau) vs ESN standard, Partie 1.
Taches temporelles (T=4000+, readout ridge, NRMSE) + capacite memoire (MC).
Resultats incrementaux dans results_part1.json.

Protocole (correction 14.2 du rapport, 2026-10-29) : le meilleur rho par seed est
choisi sur un segment de VALIDATION (NVAL = 1000 pas juste avant le test), jamais
sur le test ; le readout final est reajuste sur train+validation. Dans le JSON,
seules les cellules marquees "proto": "rho_sur_validation" sont reprises telles
quelles ; les anciennes (choix de rho sur le test) sont recalculees au relancement."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from chemins import BRUT  # noqa: E402
import numpy as np, json, os, sys
from pathlib import Path

OUT = (BRUT / "tensor-suite/part1")
OUT.mkdir(parents=True, exist_ok=True)
RJ = OUT / "results_part1.json"
results = json.load(open(RJ, encoding="utf-8-sig")) if RJ.exists() else {}

# noyau reservoir (comme le canvas, NumPy)
def spectral(W, rho):
    return W * (rho / max(np.abs(np.linalg.eigvals(W)).max(), 1e-12))

def states_all(u, N, rho, taus, seed):
    """u: (T, d_in). Renvoie (T, N)."""
    rng = np.random.default_rng(seed)
    T, d = u.shape
    Win = rng.uniform(-0.5, 0.5, size=(d, N)).astype(np.float32)
    if taus is None:
        W = [spectral(rng.normal(0, 1, (N, N)), rho).astype(np.float32)]
    else:
        W = [spectral(rng.normal(0, 1, (N, N)), rho / np.sqrt(len(taus))).astype(np.float32) for _ in taus]
    tmax = 1 if taus is None else max(taus)
    x = np.zeros((N,), np.float32)
    hist = np.zeros((tmax, N), np.float32)
    F = np.empty((T, N), np.float32)
    for t in range(T):
        drive = u[t] @ Win
        if taus is None:
            drive = drive + x @ W[0]
            x = np.tanh(drive)
        else:
            for i, tau in enumerate(taus):
                drive = drive + hist[tau - 1] @ W[i]
            x = np.tanh(drive)
            hist = np.concatenate([x[None], hist[:-1]], axis=0)
        F[t] = x
    return F

def ridge_states(Xb, N, rho, taus, seed, chunk=4000):
    """Xb: (B,28,28) MNIST-like -> (B,N)."""
    rng = np.random.default_rng(seed)
    Win = (rng.uniform(-0.5, 0.5, size=(28, N))).astype(np.float32)
    if taus is None:
        W = [spectral(rng.normal(0, 1, (N, N)), rho).astype(np.float32)]
    else:
        W = [spectral(rng.normal(0, 1, (N, N)), rho / np.sqrt(len(taus))).astype(np.float32) for _ in taus]
    tmax = 1 if taus is None else max(taus)
    out = np.empty((len(Xb), N), np.float32)
    for lo in range(0, len(Xb), chunk):
        B = Xb[lo:lo+chunk]
        x = np.zeros((len(B), N), np.float32)
        hist = np.zeros((tmax, len(B), N), np.float32)
        for t in range(28):
            drive = B[:, t, :] @ Win
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

def nrmse_score(F, y, wash, ntest, ridge=1e-6):
    Fb = np.hstack([F, np.ones((len(F), 1), np.float32)]).astype(np.float64)
    y = np.asarray(y, np.float64)
    Ftr, ytr = Fb[wash:-ntest], y[wash:-ntest]
    Fte, yte = Fb[-ntest:], y[-ntest:]
    A = Ftr.T @ Ftr + ridge * np.eye(Ftr.shape[1])
    Wout = np.linalg.solve(A, Ftr.T @ ytr)
    pred = Fte @ Wout
    if pred.ndim == 1:
        return float(np.sqrt(np.mean((pred - yte) ** 2)) / np.std(yte))
    # multi-sorties : moyenne des NRMSE par composante
    return float(np.mean(np.sqrt(np.mean((pred - yte) ** 2, 0)) / np.std(yte, 0)))

NVAL = 1000  # segment de validation avant le test : le rho y est choisi (14.2)

def nrmse_val_test(F, y, wash, ntest, nval=NVAL, ridge=1e-6):
    """Choix de rho sur la validation (14.2) : readout ajuste sur le train seul ->
    erreur de validation ; readout final reajuste sur train+validation -> test."""
    Fb = np.hstack([F, np.ones((len(F), 1), np.float32)]).astype(np.float64)
    y = np.asarray(y, np.float64)
    Ftr, ytr = Fb[wash:-ntest-nval], y[wash:-ntest-nval]
    Fva, yva = Fb[-ntest-nval:-ntest], y[-ntest-nval:-ntest]
    A = Ftr.T @ Ftr + ridge * np.eye(Ftr.shape[1])
    Wout = np.linalg.solve(A, Ftr.T @ ytr)
    pred_va = Fva @ Wout
    if pred_va.ndim == 1:
        val = float(np.sqrt(np.mean((pred_va - yva) ** 2)) / np.std(yva))
    else:
        val = float(np.mean(np.sqrt(np.mean((pred_va - yva) ** 2, 0)) / np.std(yva, 0)))
    Ftv, ytv = Fb[wash:-ntest], y[wash:-ntest]
    A2 = Ftv.T @ Ftv + ridge * np.eye(Ftv.shape[1])
    Wout2 = np.linalg.solve(A2, Ftv.T @ ytv)
    Fte, yte = Fb[-ntest:], y[-ntest:]
    pred = Fte @ Wout2
    if pred.ndim == 1:
        test = float(np.sqrt(np.mean((pred - yte) ** 2)) / np.std(yte))
    else:
        test = float(np.mean(np.sqrt(np.mean((pred - yte) ** 2, 0)) / np.std(yte, 0)))
    return val, test

# generateurs
def narma(n, seed):
    """NARMA-n (generalisation standard : alpha=.2, beta=.004, g=1.5, d=.001 pour n>10;
    n=10 : .3/.05/1.5/.1)."""
    rng = np.random.default_rng(seed)
    u = rng.uniform(0, 0.5, size=5200)
    y = np.zeros(5200, np.float32)
    a, b, g, d = (0.3, 0.05, 1.5, 0.1) if n == 10 else (0.2, 0.004, 1.5, 0.001)
    for t in range(n, 5200):
        s = y[t-n:t].sum()
        y[t] = a * y[t-1] + b * y[t-1] * s + g * u[t-n] * u[t-1] + d
    return np.stack([u, np.ones_like(u)], 1)[200:5200].astype(np.float32), y[200:5200]

def mackey_glass(tau, seed, T=5000):
    rng = np.random.default_rng(seed)
    dt = 0.1
    hist_len = int(tau / dt) + 2
    x = np.zeros(60000 + hist_len, np.float32)
    x[:hist_len] = 1.1 + 0.2 * rng.standard_normal(hist_len)
    for t in range(hist_len, len(x)):
        x[t] = x[t-1] + dt * (0.2 * x[t - int(tau/dt)] / (1 + x[t - int(tau/dt)]**10) - 0.1 * x[t-1])
    xs = x[::10]  # dt effectif = 1
    u = xs[:T][..., None]
    return np.concatenate([u, np.ones((T,1), np.float32)], 1), xs[1:T+1].astype(np.float32)

def lorenz(seed, T=4200):
    rng = np.random.default_rng(seed)
    s, r, b = 10.0, 28.0, 8/3
    dt = 0.02
    n = T + 2000
    X = np.zeros((n, 3), np.float32)
    X[0] = rng.standard_normal(3) * 5
    for t in range(1, n):
        x, y, z = X[t-1]
        X[t, 0] = x + dt * s * (y - x)
        X[t, 1] = y + dt * (x * (r - z) - y)
        X[t, 2] = z + dt * (x * y - b * z)
    X = X[2000:]
    return X[:-1], X[1:]

def memory_capacity(seed, T=4200, kmax=20):
    rng = np.random.default_rng(seed)
    u = rng.uniform(-0.5, 0.5, size=T).astype(np.float32)
    return np.stack([u, np.ones_like(u)], 1), u, kmax

def mc_score(F, u, wash, kmax=20, ridge=1e-8):
    """Capacite memoire totale : somme des corr^2 pour k=1..kmax."""
    Fb = np.hstack([F, np.ones((len(F), 1), np.float32)])[wash:].astype(np.float64)
    uu = np.asarray(u, np.float64)[wash:]
    A = Fb.T @ Fb + ridge * np.eye(Fb.shape[1])
    total = 0.0
    for k in range(1, kmax + 1):
        yk = np.roll(u, k)[wash:]
        Wout = np.linalg.solve(A, Fb.T @ yk)
        pred = Fb @ Wout
        c = np.corrcoef(pred, yk)[0, 1]
        total += c * c
    return float(total)

def mc_val(F, u, wash, kmax=20, nval=NVAL, ridge=1e-8):
    """Capacite memoire sur le segment de validation (readout ajuste sur le train
    seul) : sert a choisir rho sans toucher au segment final (14.2)."""
    Fb = np.hstack([F, np.ones((len(F), 1), np.float32)])[wash:-nval].astype(np.float64)
    uu = np.asarray(u, np.float64)
    A = Fb.T @ Fb + ridge * np.eye(Fb.shape[1])
    total = 0.0
    for k in range(1, kmax + 1):
        yk = np.roll(uu, k)[wash:-nval]
        Wout = np.linalg.solve(A, Fb.T @ yk)
        pred = Fb @ Wout
        c = np.corrcoef(pred, yk)[0, 1]
        total += c * c
    return float(total)

# runner
TAUS = list(range(1, 11))
CONFIGS = [
    ("std632", "ESN standard N=632 (=FLOPs)", 632, None),
    ("std200", "ESN standard N=200 (=unités)", 200, None),
    ("tau10", "Tenseur tau=1..10 N=200", 200, TAUS),
    ("log1248", "Tenseur log {1,2,4,8} N=200", 200, [1, 2, 4, 8]),
    ("tau1", "Tenseur tau=1 seul (contrôle)", 200, [1]),
]
RADII = [0.5, 0.9, 0.99]
SEEDS = 3

def run_task(key, gen, scorer, ntest, wash):
    # 14.2 : le meilleur rho par seed est choisi sur la validation, jamais sur le
    # test (l'argument scorer est garde pour compatibilite, la selection est la-
    # dessus faite par nrmse_val_test). Les anciennes cellules (sans marqueur
    # "proto") sont recalculees, les nouvelles sont reprises telles quelles.
    if key in results and all(results[key].get(c, {}).get("proto") == "rho_sur_validation"
                              for c, *_ in CONFIGS):
        print(f"[skip] {key}")
        return
    res = {}
    for cid, name, N, taus in CONFIGS:
        bests = []
        for seed in range(SEEDS):
            u, y = gen(seed)
            best_val, best_test = np.inf, np.inf
            per_rho = {}
            for rho in RADII:
                F = states_all(u, N, rho, taus, seed)
                val, test = nrmse_val_test(F, y, wash, ntest, NVAL)
                per_rho[str(rho)] = {"val": val, "test": test}
                if val < best_val:
                    best_val, best_test = val, test
            bests.append(best_test)
        res[cid] = {"name": name, "mean": float(np.mean(bests)), "std": float(np.std(bests)),
                    "per_seed": [float(b) for b in bests], "per_rho": per_rho,
                    "proto": "rho_sur_validation"}
        print(f"  {name:35s} {res[cid]['mean']:.4f} +/- {res[cid]['std']:.4f}", flush=True)
    results[key] = res
    json.dump(results, open(RJ, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

WASH, NTEST = 200, 2000
# NVAL = 1000 : defini plus haut (choix de rho sur la validation, 14.2)

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "narma10"):
        print("== NARMA-10 ==")
        run_task("narma10", lambda s: narma(10, s), nrmse_score, NTEST, WASH)
    if which in ("all", "narma30"):
        print("== NARMA-30 ==")
        run_task("narma30", lambda s: narma(30, s), nrmse_score, NTEST, WASH)
    if which in ("all", "mg17"):
        print("== Mackey-Glass tau=17 ==")
        run_task("mg17", lambda s: mackey_glass(17, s), nrmse_score, NTEST, WASH)
    if which in ("all", "mg50"):
        print("== Mackey-Glass tau=50 ==")
        run_task("mg50", lambda s: mackey_glass(50, s), nrmse_score, NTEST, WASH)
    if which in ("all", "mg2ts"):
        print("== MG deux echelles (17 + 50) ==")
        def gen2(s):
            _, y1 = mackey_glass(17, s)
            _, y2 = mackey_glass(50, s + 100)
            T = min(len(y1), len(y2))
            u = np.stack([y1[:T], y2[:T], np.ones(T, np.float32)], 1)
            y = (y1 + y2)[:T]
            return u, y
        run_task("mg2ts", gen2, nrmse_score, NTEST, WASH)
    if which in ("all", "lorenz"):
        print("== Lorenz-63 (1 pas, 3D) ==")
        run_task("lorenz", lambda s: lorenz(s), nrmse_score, NTEST, WASH)
    if which in ("all", "mc"):
        print("== Capacite memoire (k=1..20) ==")
        if "mc" in results and all(results["mc"].get(c, {}).get("proto") == "rho_sur_validation"
                                    for c, *_ in CONFIGS):
            print("[skip] mc")
        else:
            res = {}
            for cid, name, N, taus in CONFIGS:
                bests = []
                for seed in range(SEEDS):
                    u, y, kmax = memory_capacity(seed)
                    best_val, best_test = -np.inf, -np.inf
                    per_rho = {}
                    for rho in RADII:
                        F = states_all(u, N, rho, taus, seed)
                        val = mc_val(F, y, WASH, kmax)
                        per_rho[str(rho)] = {"val": val}
                        if val > best_val:
                            best_val = val
                            best_test = mc_score(F, y, WASH, kmax)
                    bests.append(best_test)
                res[cid] = {"name": name, "mean": float(np.mean(bests)), "std": float(np.std(bests)),
                            "per_seed": [float(b) for b in bests], "per_rho": per_rho,
                            "proto": "rho_sur_validation"}
                print(f"  {name:35s} MC = {res[cid]['mean']:.2f} +/- {res[cid]['std']:.2f}", flush=True)
            results["mc"] = res
            json.dump(results, open(RJ, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
