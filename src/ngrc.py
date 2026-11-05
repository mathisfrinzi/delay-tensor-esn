"""Partie 2.9 : premiere baseline concurrente : NG-RC (Next-Generation
Reservoir Computing, Gauthier et al. 2021). Modele lineaire-en-parametres :
features = vecteur de retards de l'entree u (profondeur D, comme le tenseur)
+ monomes croises d'ordre 2, readout ridge : PAS de reservoir du tout.
Variantes :
  - u-seul (equitable) : retards de u seulement, comme le tenseur (l'etat
    du reservoir ne voit que u) ;
  - u+y (oracle) : retards de u ET de la sortie passee y : information que
    le tenseur n'a PAS ; borne superieure de la famille "features explicites".
Taches temporelles (NARMA-30, Silverbox, WH ; MNIST non applicable).
Splits identiques a eval_pruned (narma wash 200 / test 2000 ; phys wash 500 /
test 10000). 3 seeds pour NARMA (tirages d'entree), deterministe pour phys.
Budget : le NG-RC a ~10^3 parametres contre 400k connexions recurrentes du
tenseur : la parite se fait sur la PROFONDEUR de memoire (D), pas sur les
parametres. Features preallouees en float32 + equations normales par blocs
(memoire bornee). Checkpoint : results_ngrc.json."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from chemins import BRUT  # noqa: E402
import numpy as np, json, sys
from pathlib import Path
from scipy.io import loadmat

OUT = (BRUT / "tensor-suite/part2")
RJ = OUT / "results_ngrc.json"
results = json.load(open(RJ)) if RJ.exists() else {}

def narma30(seed, T=5200):
    rng = np.random.default_rng(seed)
    u = rng.uniform(0, 0.5, size=T)
    y = np.zeros(T, np.float32)
    a, b, g, d = 0.2, 0.004, 1.5, 0.001
    n = 30
    for t in range(n, T):
        y[t] = a * y[t-1] + b * y[t-1] * y[t-n:t].sum() + g * u[t-n] * u[t-1] + d
    return u.astype(np.float64), y.astype(np.float64)

def load_phys(which):
    if which == "silverbox":
        d = loadmat("/tmp/datasets/silverbox/SilverboxFiles/SNLS80mV.mat")
        u = d["V1"].ravel().astype(np.float64); y = d["V2"].ravel().astype(np.float64)
    else:
        d = loadmat("/tmp/datasets/WH.mat")
        u = d["uBenchMark"].ravel().astype(np.float64); y = d["yBenchMark"].ravel().astype(np.float64)
    s = np.std(y[:100000])
    return u[:30000] / np.std(u[:30000]), y[:30000] / s

def features(u, D, y=None):
    """Matrice (T, P) float32 preallouee : [1, u(t-1..t-D), monomes u_i.u_j,
    (+ y(t-1..t-D), monomes y_i.y_j, croises u_i.y_j), u(t)]."""
    T = len(u)
    n_mon = D * (D + 1) // 2
    extra = (D + n_mon + D * D) if y is not None else 0
    P = 1 + D + n_mon + extra + 1
    Phi = np.zeros((T, P), np.float32)
    Phi[:, 0] = 1.0
    U = np.zeros((T, D), np.float32)
    for d in range(1, D + 1):
        U[d:, d - 1] = u[:-d]
    Phi[:, 1:1 + D] = U
    c = 1 + D
    for i in range(D):
        for j in range(i, D):
            Phi[:, c] = U[:, i] * U[:, j]
            c += 1
    if y is not None:
        Yd = np.zeros((T, D), np.float32)
        for d in range(1, D + 1):
            Yd[d:, d - 1] = y[:-d]
        Phi[:, c:c + D] = Yd
        c += D
        for i in range(D):
            for j in range(i, D):
                Phi[:, c] = Yd[:, i] * Yd[:, j]
                c += 1
        for i in range(D):
            for j in range(D):
                Phi[:, c] = U[:, i] * Yd[:, j]
                c += 1
    Phi[:, c] = u  # terme lineaire u(t) : le tenseur le recoit aussi
    return Phi

task = sys.argv[1]
R = results.setdefault(task, {})
D_LIST = [10, 30]
wash, ntest = (200, 2000) if task == "narma30" else (500, 10000)

for D in D_LIST:
    for mode in ("u", "u+y"):
        key = f"D{D}_{mode}"
        if key in R:
            continue
        tests, nfeat = [], 0
        for seed in ([0, 1, 2] if task == "narma30" else [0]):
            if task == "narma30":
                u, y = narma30(seed)
            else:
                u, y = load_phys(task)
            Phit = features(u, D, y=y if mode == "u+y" else None)
            nfeat = Phit.shape[1]
            lam = 1e-6 if task == "narma30" else 1e-8
            tr0, tr1 = wash, len(u) - ntest
            P = Phit.shape[1]
            A = np.zeros((P, P)); b = np.zeros(P)
            for lo in range(tr0, tr1, 4000):
                Pc = Phit[lo:min(lo + 4000, tr1)].astype(np.float64)
                yc = y[lo:min(lo + 4000, tr1)]
                A += Pc.T @ Pc
                b += Pc.T @ yc
            A += lam * np.eye(P)
            Wout = np.linalg.solve(A, b)
            pred = Phit[-ntest:].astype(np.float64) @ Wout
            tests.append(float(np.sqrt(np.mean((pred - y[-ntest:]) ** 2)) / np.std(y[-ntest:])))
        R[key] = {"n_feat": int(nfeat), "test_mean": float(np.mean(tests)),
                  "test_std": float(np.std(tests)), "tests": tests}
        json.dump(results, open(RJ, "w"), indent=1)
        print(f"{task} NG-RC {key}: test={np.mean(tests):.4f} +/- {np.std(tests):.4f} ({nfeat} features)", flush=True)

print(f"{task} 2.9 NG-RC terminé")
