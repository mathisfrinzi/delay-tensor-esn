"""Partie 8 : 8.1 : spectre de retards d'une tache (oracle sans reservoir).

Oracle de 6.7 (version prod) : features = retards u(t-a), a dans {0..60},
+ produits u(t-a)u(t-b) (a<b) + biais. Importance par ablation :
I(L) = NRMSE(L \\ {L}) - NRMSE(L), ou retirer L enleve u(t-L) ET tous les
produits qui le contiennent. Ablations par blocs de 3 retards consecutifs
aussi (trous mous des systemes physiques, 5.1).

Astuce de calcul : le Gram d'un ensemble ablate est une SOUS-MATRICE du Gram
complet. Une seule passe de donnees par (tache, seed) accumule A_main, A_val,
A_test (+ b et zz scalaires) ; chaque ablation = extraction de sous-matrice +
cholesky. Le NRMSE de validation et de test se recalculent exactement depuis
les Grams : val_mse = Wo^T A_val Wo - 2 Wo^T b_val + zz_val, idem test.

Taches : narma30, narma25 (3 seeds), silverbox, wh (donnee fixee, 1 passe).
Splits type 6.7 : T=50000, wash 500, train [500, 40000), test [40000, 50000),
lambda choisi sur les 20 % finaux du train puis reentraîne sur tout le train.

Sortie : p8/results_p8_spectre.json {cells, agg, meta} (COMPLETE quand tout est fait).
"""
import numpy as np, json, time, sys, subprocess, os
from pathlib import Path
from scipy.linalg import cho_factor, cho_solve
from scipy.io import loadmat

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p6_common import narma_x, LAMBDAS, TIME_BUDGET

P8OUT = Path(__file__).resolve().parent.parent / "results"
P8OUT.mkdir(parents=True, exist_ok=True)


def save_p8(results, name):
    json.dump(results, open(P8OUT / f"results_p8_{name}.json", "w"), indent=1)

T = 50_000
WASH = 500
I_TE = 40_000
LAGS = list(range(0, 61))
NL = len(LAGS)
IU, IV = np.triu_indices(NL, 1)          # paires (a<b) de produits
D = NL + len(IU) + 1                      # 61 + 1830 + 1 = 1892
BIAS = D - 1
CH = 4000
SEEDS_NARMA = [0, 1, 2]
DATA = Path("/tmp/datasets")


def ensure_datasets():
    """/tmp est vide entre les tours : retélécharge si besoin."""
    sb = DATA / "silverbox/SilverboxFiles/SNLS80mV.mat"
    wh = DATA / "WH.mat"
    if not sb.exists():
        DATA.mkdir(parents=True, exist_ok=True)
        zipn = DATA / "SilverboxFiles.zip"
        subprocess.run(["curl", "-sL", "-m", "150",
                        "https://drive.google.com/uc?export=download&id=17iS-6oBUUgrmiAcrZoG9S5sOaljZnDSy",
                        "-o", str(zipn)], check=True)
        subprocess.run(["unzip", "-o", "-q", str(zipn), "-d", str(DATA / "silverbox")], check=True)
    if not wh.exists():
        subprocess.run(["curl", "-sL", "-m", "150",
                        "https://www.ee.kth.se/~hjalmars/ifac_tc11_benchmarks/2009_wienerhammerstein/WienerHammerBenchMark.mat",
                        "-o", str(wh)], check=True)
    return sb, wh


def task_data(task, seed):
    """Retourne (u, y) normalises comme suite_real.py, T=50000."""
    if task == "narma30":
        u, y, gen = narma_x(seed, 30, T)
        return np.asarray(u, np.float64), y
    if task == "narma25":
        u, y, gen = narma_x(seed, 25, T)
        return np.asarray(u, np.float64), y
    sb, wh = ensure_datasets()
    if task == "silverbox":
        d = loadmat(sb)
        u_all = d["V1"].ravel().astype(np.float64)
        y_all = d["V2"].ravel().astype(np.float64)
        scale = np.std(y_all[:100000])
        return u_all / np.std(u_all), y_all / scale
    if task == "wh":
        d = loadmat(wh)
        u_all = d["uBenchMark"].ravel().astype(np.float64)
        y_all = d["yBenchMark"].ravel().astype(np.float64)
        scale = np.std(y_all[:100000])
        return u_all / np.std(u_all), y_all / scale
    raise ValueError(task)


def build_rows(u, t0, t1):
    """(t1-t0, D) : retards + produits + biais, float64."""
    n = t1 - t0
    Xd = np.empty((n, NL), np.float64)
    for ia, a in enumerate(LAGS):
        Xd[:, ia] = u[t0 - a:t1 - a]
    P = Xd[:, :, None] * Xd[:, None, :]
    X = np.concatenate([Xd, P[:, IU, IV]], 1)
    return np.column_stack([X, np.ones(n)])


def accumulate(u, y):
    """Une passe : A_main, A_val, A_test, b_*, zz_*, n_test."""
    m = int(0.8 * (I_TE - WASH)) + WASH     # train_main = [WASH, m), val = [m, I_TE)
    A_main = np.zeros((D, D)); b_main = np.zeros(D)
    A_val = np.zeros((D, D)); b_val = np.zeros(D)
    A_test = np.zeros((D, D)); b_test = np.zeros(D)
    zz_val = 0.0; zz_test = 0.0
    for lo, hi, A, b, zzz in ((WASH, m, A_main, b_main, None),
                              (m, I_TE, A_val, b_val, "val"),
                              (I_TE, T, A_test, b_test, "test")):
        for c0 in range(lo, hi, CH):
            c1 = min(c0 + CH, hi)
            X = build_rows(u, c0, c1)
            z = y[c0:c1]
            A += X.T @ X; b += X.T @ z
            if zzz == "val": zz_val += float(z @ z)
            elif zzz == "test": zz_test += float(z @ z)
    n_test = T - I_TE
    return A_main, b_main, A_val, b_val, A_test, b_test, zz_val, zz_test, n_test


def solve_kept(A_main, b_main, A_val, b_val, zz_val, A_test, b_test, zz_test,
               n_test, keep):
    """Ridge protocole 6.0 sur la sous-matrice 'keep'. Retourne
    (nrmse_validation, nrmse_test, lam) — 14.2 : l'importance I(L) est
    mesuree sur la validation, le test n'est jamais un critere."""
    Am = A_main[np.ix_(keep, keep)]
    Av = A_val[np.ix_(keep, keep)]
    At = A_test[np.ix_(keep, keep)]
    bm = b_main[keep]; bv = b_val[keep]; bt = b_test[keep]
    d = len(keep)
    eye = np.eye(d)
    best = (np.inf, None)
    for lam in LAMBDAS:
        c, low = cho_factor(Am + lam * eye, lower=True)
        Wo = cho_solve((c, low), bm)
        v = float(Wo @ (Av @ Wo) - 2 * Wo @ bv) + zz_val
        if v < best[0]:
            best = (v, lam)
    lam = best[1]
    m_val = I_TE - (int(0.8 * (I_TE - WASH)) + WASH)
    s_zva = b_val[BIAS] / m_val
    var_val = zz_val / m_val - s_zva ** 2
    nrmse_val = float(np.sqrt(best[0] / m_val) / np.sqrt(max(var_val, 1e-300)))
    c, low = cho_factor(Am + Av + lam * eye, lower=True)
    Wo = cho_solve((c, low), bm + bv)
    mse = float(Wo @ (At @ Wo) - 2 * Wo @ bt) + zz_test
    s_zte = b_test[BIAS] / n_test
    var = zz_test / n_test - s_zte ** 2
    return nrmse_val, float(np.sqrt(mse / n_test) / np.sqrt(var)), lam


def keep_mask(remove):
    """remove : ensemble de retards a ablater -> indices de features gardes."""
    keep_del = [a for a in LAGS if a not in remove]
    keep_pairs = [k for k in range(len(IU)) if IU[k] not in remove and IV[k] not in remove]
    idx = list(keep_del) + [NL + k for k in keep_pairs] + [BIAS]
    return np.array(sorted(idx))


def analyse(res):
    """Agregation par tache : I moyen, blocs moyens, L* (seuils 5 %)."""
    agg = {}
    for task, seeds in res["cells"].items():
        ok = [s for s in seeds
              if res["cells"][task][s].get("proto") == "I_sur_validation"]
        if not ok:
            continue  # 14.2 : cellule ancienne (I sur le test), exclue
        sgl = np.mean([res["cells"][task][s]["I_single"] for s in ok], 0)
        blk = np.mean([res["cells"][task][s]["I_block"] for s in ok], 0)
        base = float(np.mean([res["cells"][task][s]["nrmse_full"] for s in ok]))
        L_star = [LAGS[i] for i in np.where(sgl > 0.05 * sgl.max())[0]]
        blocks = [list(range(b, min(b + 3, 61))) for b in range(0, 61, 3)]
        B_star = [blocks[i] for i in np.where(blk > 0.05 * blk.max())[0]]
        agg[task] = {"I_single_mean": sgl.tolist(), "I_block_mean": blk.tolist(),
                     "nrmse_full_mean": base, "L_star": L_star, "B_star": B_star}
    res["agg"] = agg
    res["meta"]["status"] = "COMPLETE" if res["meta"]["done_cells"] == res["meta"]["total_cells"] else "PARTIAL"


def main():
    path = P8OUT / "results_p8_spectre.json"
    if path.exists():
        res = json.load(open(path))
    else:
        res = {"cells": {}, "agg": {}, "meta": {
            "T": T, "wash": WASH, "i_te": I_TE, "lags": LAGS,
            "blocks_3": [list(range(b, min(b + 3, 61))) for b in range(0, 61, 3)],
            "lambda_grid": LAMBDAS, "seeds_narma": SEEDS_NARMA,
            "threshold": "I > 5% de max I (singles) / max bloc (blocs)",
            "predictions": {
                "narma30": "L* = {1, 30} (+ 31-32 via recurrence)",
                "narma25": "L* = {1, 25} (+ 26-27)",
                "silverbox": "bandes larges sans pic net -> critere spectral",
                "wh": "bandes larges sans pic net"},
            "done_cells": 0, "total_cells": 8}}
    
    all_cells = [("narma30", s) for s in SEEDS_NARMA] + [("narma25", s) for s in SEEDS_NARMA] \
        + [("silverbox", 0), ("wh", 0)]
    t0 = time.time()
    for task, seed in all_cells:
        if (task in res["cells"] and str(seed) in res["cells"][task]
                and res["cells"][task][str(seed)].get("proto") == "I_sur_validation"):
            continue  # 14.2 : deja calculee sous le protocole corrige
        u, y = task_data(task, seed)
        (A_main, b_main, A_val, b_val, A_test, b_test,
         zz_val, zz_test, n_test) = accumulate(u, y)
        common = (A_main, b_main, A_val, b_val, zz_val, A_test, b_test, zz_test, n_test)
        full_keep = np.arange(D)
        nrmse_full_val, nrmse_full, lam_full = solve_kept(*common, full_keep)
        # 14.2 : importance I(L) calculee sur la VALIDATION (le test guide la
        # conception via la validation uniquement)
        I_single = []
        for L in LAGS:
            nr_val, _, _ = solve_kept(*common, keep_mask({L}))
            I_single.append(nr_val - nrmse_full_val)
        blocks = [list(range(b, min(b + 3, 61))) for b in range(0, 61, 3)]
        I_block = []
        for B in blocks:
            nr_val, _, _ = solve_kept(*common, keep_mask(set(B)))
            I_block.append(nr_val - nrmse_full_val)
        cell = {"proto": "I_sur_validation",
                "nrmse_full": nrmse_full, "nrmse_full_val": nrmse_full_val,
                "lam_full": lam_full,
                "I_single": [float(x) for x in I_single],
                "I_block": [float(x) for x in I_block]}
        res["cells"].setdefault(task, {})[str(seed)] = cell
        res["meta"]["done_cells"] = sum(len(v) for v in res["cells"].values())
        analyse(res)
        save_p8(res, "spectre")
        print(f"[cell] {task} seed {seed}: full={nrmse_full:.4f} "
              f"I(1)={I_single[1]:.4f} maxI={max(I_single):.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        if time.time() - t0 > TIME_BUDGET:
            print("BUDGET : checkpoint", flush=True)
            return
    analyse(res)
    save_p8(res, "spectre")
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    main()
