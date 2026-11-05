"""NARMA-10: standard ESN vs tensor-delay ESN W(i,j,tau), tau=1..10.

Protocole (correction 14.2 du rapport, 2026-10-29) : le meilleur rho par seed est
choisi sur un segment de VALIDATION (NVAL pas juste avant le test), jamais sur le
test ; le readout final est reajuste sur train+validation.

Fair comparison at equal per-step FLOPs:
  - tensor ESN with N units costs ~10 * N^2 (10 delay slices) per step
  - standard ESN with N_s units costs N_s^2 -> N_s = N * sqrt(10) ~ 3.16 N
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from chemins import BRUT  # noqa: E402
import numpy as np
from pathlib import Path

rng_global = np.random.default_rng(0)

# NARMA-10
def narma10(T, rng):
    u = rng.uniform(0.0, 0.5, size=T + 20)
    y = np.zeros(T + 20)
    for t in range(10, T + 20):
        s = y[t-10:t].sum()
        y[t] = 0.3 * y[t-1] + 0.05 * y[t-1] * s + 1.5 * u[t-10] * u[t-1] + 0.1
    return u[20:], y[20:]

# models
def run_standard_esn(u, N, spectral_radius, seed, leak=1.0):
    """x(t) = (1-a) x(t-1) + a * tanh(Win u(t) + W x(t-1))"""
    rng = np.random.default_rng(seed)
    Win = rng.uniform(-0.5, 0.5, size=(N, 1))
    W = rng.normal(0, 1, size=(N, N))
    W *= spectral_radius / max(np.abs(np.linalg.eigvals(W)).max(), 1e-12)
    T = len(u)
    X = np.zeros((T, N))
    x = np.zeros(N)
    for t in range(T):
        x = (1 - leak) * x + leak * np.tanh(Win @ np.array([u[t]]) + W @ x)
        X[t] = x
    return X

def run_tensor_esn(u, N, taus, spectral_radius, seed, leak=1.0):
    """x(t) = (1-a) x(t-1) + a * tanh(Win u(t) + sum_tau W_tau @ x(t-tau))
    W_tau is the tau-th slice of the delay tensor W(i,j,tau)."""
    rng = np.random.default_rng(seed)
    Win = rng.uniform(-0.5, 0.5, size=(N, 1))
    tau_max = max(taus)
    W = {tau: rng.normal(0, 1, size=(N, N)) for tau in taus}
    # scale so the combined delay map has target spectral radius:
    # each slice gets rho / sqrt(K) spectral radius -> effective linear radius ~ rho
    for tau in taus:
        r = max(np.abs(np.linalg.eigvals(W[tau])).max(), 1e-12)
        W[tau] *= (spectral_radius / np.sqrt(len(taus))) / r
    T = len(u)
    X = np.zeros((T, N))
    hist = np.zeros((T + tau_max, N))  # ring of past states, pre-filled zeros
    off = tau_max  # hist[off + t] = x(t)
    for t in range(T):
        drive = Win @ np.array([u[t]])
        for tau in taus:
            drive = drive + W[tau] @ hist[off + t - tau]
        x_new = (1 - leak) * hist[off + t - 1] + leak * np.tanh(drive)
        hist[off + t] = x_new
        X[t] = x_new
    return X

def readout(X, y, washout, ridge=1e-6):
    Xb = np.hstack([X, np.ones((X.shape[0], 1))])
    Xtr, ytr = Xb[washout:-2000], y[washout:-2000]
    Xte, yte = Xb[-2000:], y[-2000:]
    A = Xtr.T @ Xtr + ridge * np.eye(Xtr.shape[1])
    Wout = np.linalg.solve(A, Xtr.T @ ytr)
    pred = Xte @ Wout
    nrmse = np.sqrt(np.mean((pred - yte) ** 2)) / np.std(yte)
    return nrmse, pred, yte

NVAL = 1000  # segment de validation avant le test : le rho y est choisi (14.2)

def readout_valtest(X, y, washout, ntest=2000, nval=NVAL, ridge=1e-6):
    """Choix de rho sur la validation : readout ajuste sur le train seul ->
    erreur de validation ; readout final reajuste sur train+validation -> test."""
    Xb = np.hstack([X, np.ones((X.shape[0], 1))])
    Xtr, ytr = Xb[washout:-ntest-nval], y[washout:-ntest-nval]
    Xva, yva = Xb[-ntest-nval:-ntest], y[-ntest-nval:-ntest]
    A = Xtr.T @ Xtr + ridge * np.eye(Xtr.shape[1])
    Wout = np.linalg.solve(A, Xtr.T @ ytr)
    val = float(np.sqrt(np.mean((Xva @ Wout - yva) ** 2)) / np.std(yva))
    Xtv, ytv = Xb[washout:-ntest], y[washout:-ntest]
    A2 = Xtv.T @ Xtv + ridge * np.eye(Xtv.shape[1])
    Wout2 = np.linalg.solve(A2, Xtv.T @ ytv)
    Xte, yte = Xb[-ntest:], y[-ntest:]
    test = float(np.sqrt(np.mean((Xte @ Wout2 - yte) ** 2)) / np.std(yte))
    return val, test

# experiment
T = 6000
WASH = 200
TAUS = list(range(1, 11))
N_TENSOR = 200
N_STD_EQUAL_FLOPS = int(round(N_TENSOR * np.sqrt(10)))  # 632
N_STD_EQUAL_PARAMS = 63   # N_tensor^3 params vs N_std^2 : 200^2*10 = 400k; sqrt = 632 actually params too
RESULTS = {}

def sweep(name, fn, radii, seeds):
    # 14.2 : le meilleur rho par seed est choisi sur la validation, pas sur le test.
    per_seed_best = []
    per_seed_rho = {}
    for seed in seeds:
        rng = np.random.default_rng(1000 + seed)
        u, y = narma10(T, rng)
        best_val, best_test = np.inf, np.inf
        per_rho = {}
        for rho in radii:
            X = fn(u, rho, seed)
            val, test = readout_valtest(X, y, WASH)
            per_rho[str(rho)] = {"val": val, "test": test}
            if val < best_val:
                best_val, best_test = val, test
        per_seed_best.append(best_test)
        per_seed_rho[str(seed)] = per_rho
    per_seed_best = np.array(per_seed_best)
    RESULTS[name] = {"mean": float(np.mean(per_seed_best)),
                     "std": float(np.std(per_seed_best)),
                     "min": float(np.min(per_seed_best)),
                     "per_seed": per_seed_best.tolist(),
                     "per_seed_rho": per_seed_rho,
                     "proto": "rho_sur_validation"}
    print(f"{name:35s} NRMSE = {np.mean(per_seed_best):.4f} +/- {np.std(per_seed_best):.4f}  (best {np.min(per_seed_best):.4f})")

radii = [0.5, 0.7, 0.9, 0.95, 0.99]
seeds = range(10)

sweep("ESN standard N=632 (=FLOPs)",
       lambda u, rho, s: run_standard_esn(u, N_STD_EQUAL_FLOPS, rho, s), radii, seeds)
sweep("ESN standard N=200 (=units)",
       lambda u, rho, s: run_standard_esn(u, N_TENSOR, rho, s), radii, seeds)
sweep("ESN tensoriel W(i,j,tau) N=200",
       lambda u, rho, s: run_tensor_esn(u, N_TENSOR, TAUS, rho, s), radii, seeds)

# ablation: sparse tensor, log-spaced delays
sweep("ESN tensoriel taus log {1,2,4,8}",
       lambda u, rho, s: run_tensor_esn(u, N_TENSOR, [1, 2, 4, 8], rho, s), radii, seeds)
sweep("ESN tensoriel tau=1 seul (controle)",
       lambda u, rho, s: run_tensor_esn(u, N_TENSOR, [1], rho, s), radii, seeds)

# equal FLOPs: tensor with 10 slices vs standard 632: also tensor at equal flops to standard 200:
sweep("ESN tensoriel N=63 (=FLOPs vs N=200 std)",
       lambda u, rho, s: run_tensor_esn(u, 63, TAUS, rho, s), radii, seeds)

out = (BRUT / "tensor-esn-narma10")
out.mkdir(parents=True, exist_ok=True)
import json
with open(out / "results.json", "w") as f:
    json.dump(RESULTS, f, indent=2)
print("\nSaved to results.json")

# save a plot: one representative test fit for the tensor model
rng = np.random.default_rng(1000)
u, y = narma10(T, rng)
X = run_tensor_esn(u, N_TENSOR, TAUS, 0.9, 0)
nrmse, pred, yte = readout(X, y, WASH)
X2 = run_standard_esn(u, N_STD_EQUAL_FLOPS, 0.9, 0)
nrmse2, pred2, _ = readout(X2, y, WASH)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(yte[:300], "k-", label="NARMA-10 cible", linewidth=1.5)
ax.plot(pred[:300], "tab:blue", label=f"ESN tensoriel (NRMSE {nrmse:.3f})", linewidth=1.2)
ax.plot(pred2[:300], "tab:orange", label=f"ESN standard N=632 (NRMSE {nrmse2:.3f})", linewidth=1.2, alpha=0.8)
ax.set_xlabel("pas de temps (test)")
ax.set_ylabel("y")
ax.legend()
fig.tight_layout()
fig.savefig(out / "fit.png", dpi=130)
print("plot saved")

# bonus: standard ESN + input delay line u(t-1..10) fed to readout
def sweep_delayline(name, fn, radii, seeds):
    # 14.2 : meme correction que sweep — choix de rho sur la validation.
    per_seed = []
    per_seed_rho = {}
    for seed in seeds:
        rng = np.random.default_rng(1000 + seed)
        u, y = narma10(T, rng)
        best_val, best_test = np.inf, np.inf
        per_rho = {}
        for rho in radii:
            X = fn(u, rho, seed)
            D = np.zeros((len(u), 11))
            D[:, 0] = u
            for k in range(1, 11):
                D[k:, k] = u[:-k]
            Xf = np.hstack([X, D])
            val, test = readout_valtest(Xf, y, WASH)
            per_rho[str(rho)] = {"val": val, "test": test}
            if val < best_val:
                best_val, best_test = val, test
        per_seed.append(best_test)
        per_seed_rho[str(seed)] = per_rho
    per_seed = np.array(per_seed)
    RESULTS[name] = {"mean": float(np.mean(per_seed)), "std": float(np.std(per_seed)),
                     "min": float(np.min(per_seed)), "per_seed": per_seed.tolist(),
                     "per_seed_rho": per_seed_rho, "proto": "rho_sur_validation"}
    print(f"{name:35s} NRMSE = {np.mean(per_seed):.4f} +/- {np.std(per_seed):.4f}  (best {np.min(per_seed):.4f})")

sweep_delayline("ESN N=632 + ligne de retard u(t-1..10)",
                lambda u, rho, s: run_standard_esn(u, N_STD_EQUAL_FLOPS, rho, s), radii, seeds)
with open(str(BRUT / "tensor-esn-narma10/results.json"), "w") as f:
    json.dump(RESULTS, f, indent=2)
