"""Tau entraines : descente de coordonnees sur alpha_tau (poids de chaque tranche).
x(t) = tanh(Win u(t) + somme_i alpha_i W_i x(t-tau_i)). Objectif : NRMSE validation.
Compare : alpha uniforme (=1) vs alpha appris, en test."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from chemins import BRUT  # noqa: E402
import numpy as np, json, sys
from suite_part2 import states_seq, nrmse_score, narma, _load_phys

OUT = str(BRUT / "tensor-suite/part2/results_alpha.json")
res = json.load(open(OUT)) if __import__("os").path.exists(OUT) else {}

TAUS = [1, 2, 4, 8, 16]
N = 200
RHO = 0.9

def states_alpha(u, W, Win, alphas):
    T = len(u)
    tmax = max(TAUS)
    x = np.zeros((N,), np.float32)
    hist = np.zeros((tmax, N), np.float32)
    F = np.empty((T, N), np.float32)
    for t in range(T):
        drive = u[t] @ Win
        for i, tau in enumerate(TAUS):
            if alphas[i]:
                drive = drive + alphas[i] * (hist[tau - 1] @ W[i])
        x = np.tanh(drive)
        hist = np.concatenate([x[None], hist[:-1]], axis=0)
        F[t] = x
    return F

def split_score(F, y, wash, ntest, nval):
    Fb = np.hstack([F, np.ones((len(F), 1), np.float32)]).astype(np.float64)
    y = np.asarray(y, np.float64)
    tr = slice(wash, -ntest - nval); va = slice(-ntest - nval, -ntest); te = slice(-ntest, None)
    A = Fb[tr].T @ Fb[tr] + 1e-8 * np.eye(Fb.shape[1])
    Wout = np.linalg.solve(A, Fb[tr].T @ y[tr])
    val = float(np.sqrt(np.mean((Fb[va] @ Wout - y[va]) ** 2)) / np.std(y[te]))
    test = float(np.sqrt(np.mean((Fb[te] @ Wout - y[te]) ** 2)) / np.std(y[te]))
    return val, test

def coord_descent(u, y, W, Win, wash, ntest, nval, passes=2, cands=(0.0, 0.5, 1.5, 2.0)):
    alpha = np.ones(len(TAUS)) / np.sqrt(len(TAUS))
    _, test0 = split_score(states_alpha(u, W, Win, alpha), y, wash, ntest, nval)
    best_val, _ = split_score(states_alpha(u, W, Win, alpha), y, wash, ntest, nval)
    history = [list(alpha)]
    for p in range(passes):
        for i in range(len(TAUS)):
            cur = alpha[i]
            for c in cands:
                a = cur * c
                if a < 0: continue
                alpha[i] = a
                v, _ = split_score(states_alpha(u, W, Win, alpha), y, wash, ntest, nval)
                if v < best_val:
                    best_val = v; cur = a
            alpha[i] = cur
        history.append(list(alpha))
    _, test1 = split_score(states_alpha(u, W, Win, alpha), y, wash, ntest, nval)
    return test0, test1, alpha, history

def run(task):
    if task in res: print("[skip]", task); return
    # poids fixes
    rng = np.random.default_rng(0)
    Win = rng.uniform(-0.5, 0.5, size=(u_dim(task), N)).astype(np.float32)
    W = []
    K = len(TAUS)
    for tau in TAUS:
        M = np.random.default_rng(100 + tau).normal(0, 1, (N, N)).astype(np.float32)
        r = max(np.abs(np.linalg.eigvals(M.astype(np.float64))).max(), 1e-12)
        W.append((M * (RHO / np.sqrt(K) / r)).astype(np.float32))
    out = {}
    if task == "narma30":
        for seed in range(2):
            u, y = narma(30, seed)
            t0, t1, alpha, hist = coord_descent(u, y, W, Win, 200, 2000, 1000, passes=2)
            out[f"seed{seed}"] = {"test_uniform": t0, "test_learned": t1,
                                  "alphas": [round(float(a), 4) for a in alpha]}
            print(f"  seed{seed}: uniform {t0:.4f} -> appriss {t1:.4f}  alpha={np.round(alpha,3)}", flush=True)
    else:
        u, y = _load_phys(task)
        t0, t1, alpha, hist = coord_descent(u[:30000], y[:30000], W, Win, 500, 10000, 5000,
                                            passes=1, cands=(0.0, 0.5, 2.0))
        out["seed0"] = {"test_uniform": t0, "test_learned": t1,
                        "alphas": [round(float(a), 4) for a in alpha]}
        print(f"  seed0: uniform {t0:.4f} -> appris {t1:.4f}  alpha={np.round(alpha,3)}", flush=True)
    res[task] = out
    json.dump(res, open(OUT, "w"), indent=2)

def u_dim(task):
    return 2

if __name__ == "__main__":
    run(sys.argv[1])
