"""Les références qui manquaient, et Iacob-Dambre optimisé par CMA-ES. Exploratoire sur les tâches déjà ouvertes.

Cinq familles, toutes à fuite, réglées comme l'ESN à fuite (mêmes tirages de ρ, fuite, échelle d'entrée) :

  self_esn          Xu, Liu et Li (2024) : x(t) dépend aussi de γ·x(t − m) (une tranche identité
                    gratuite, γ ∈ [0 ; 1], m log-uniforme entier sur [2 ; 200]) ; W dense, N = 632.
  lecture_retardee  Holzmann et Hauser (2010) : ESN à fuite N = 632, lecture sur [x(t), x(t − τ)]
                    pour τ dans le S calculé (τ > 1), la lecture a K fois plus de variables.
  cycle             Rodan et Tiňo (2011), réservoir en cycle simple : W = r·(permutation circulaire),
                    entrées de même amplitude et de signes aléatoires ; N = 632 (même taille d'état,
                    632 connexions récurrentes seulement).
  orthogonal        W = ρ·Q, Q orthogonale aléatoire (toutes les valeurs propres de module ρ), N = 632.
  iacob_cma         Iacob et Dambre (2024) avec leur optimisation : positions tirées d'un mélange de
                    K = 4 gaussiennes, et CMA-ES sur le génome (échelle des poids et connectivité par
                    paire de groupes, fuite et biais par groupe, moyennes, écarts-types, corrélations
                    et poids du mélange, échelle d'entrée, vitesse de propagation), délai
                    D = min(200, 1 + ⌊v·distance⌋). λ = 10 candidats × 16 générations = 160
                    évaluations, plus que les 60 des autres familles : avantage de réglage assumé.

    python src/references2.py run [tâche ...]      # IACOB_PROCS processus
    python src/references2.py tableau
Résultats : results/results_references2.json (reprise automatique). ESSAI=1 : version réduite.
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iacob_dambre as I       # noqa: E402  (installe le simulateur à délais de propagation)
import reservoir as R          # noqa: E402
import p7_common as P7         # noqa: E402

PROTO = "references2_1"
FICHIER = (R.RES / "essai" if R.ESSAI else R.RES) / "results_references2.json"
N0 = 632
FAM_REF = ["self_esn", "lecture_retardee", "cycle", "orthogonal"]
CMA_TACHES = ["narma30", "silverbox", "wh", "mc"]
K_GR = 4
LAMBDA, GENERATIONS = (4, 2) if R.ESSAI else (10, 16)
DCAP = 200


def familles(tache):
    return FAM_REF + (["iacob_cma"] if tache in CMA_TACHES else [])


# tirages (mêmes ρ, fuite, échelle que l'ESN à fuite, puis les paramètres propres)
def tirer(famille, rng, tache=None):
    cfg = {"rho": float(rng.uniform(0.3, 1.1)), "fuite": float(rng.uniform(0.05, 1.0)),
           "echelle": float(np.exp(rng.uniform(np.log(0.05), np.log(2.0))))}
    g, m = float(rng.uniform(0, 1)), int(np.rint(np.exp(rng.uniform(np.log(2), np.log(200)))))
    if famille == "self_esn":
        if tache == "mnist":                 # 28 lignes : un délai plus long n'agit jamais (et coûterait 2,5 Go)
            m = 2 + (m - 2) % 26
        cfg.update(gamma=g, m=m)
    return cfg


def construire(famille, cfg, seed, d):
    if famille in ("self_esn", "lecture_retardee"):
        res = R.construire(seed, [1], N0, cfg["rho"], d=d, echelle=cfg["echelle"], fuite=cfg["fuite"])
        if famille == "self_esn":
            m = int(cfg["m"])
            res["taus"] = [1, m]
            res["kinds"][m] = "ident"
            res["tranches"].append((m, "diag", np.full(N0, cfg["gamma"], np.float32)))
        return res
    rng = np.random.default_rng(seed)
    if famille == "cycle":
        Win = (np.sign(rng.uniform(-1, 1, size=(d, N0))) * 0.25 * cfg["echelle"]).astype(np.float32)
        perm = (np.arange(N0) - 1) % N0
        return {"N": N0, "taus": [1], "kinds": {1: "perm"}, "Win": Win,
                "tranches": [(1, "perm", perm, np.float32(cfg["rho"]))], "paire": None, "W2": None,
                "fuite": np.float32(cfg["fuite"]), "rho": cfg["rho"], "echelle": cfg["echelle"]}
    if famille == "orthogonal":
        Win = (rng.uniform(-0.5, 0.5, size=(d, N0)) * cfg["echelle"]).astype(np.float32)
        Q, Rq = np.linalg.qr(rng.normal(0, 1, (N0, N0)))
        Q = Q * np.sign(np.diag(Rq))
        return {"N": N0, "taus": [1], "kinds": {1: "dense"}, "Win": Win,
                "tranches": [(1, "dense", (cfg["rho"] * Q).astype(np.float32))], "paire": None, "W2": None,
                "fuite": np.float32(cfg["fuite"]), "rho": cfg["rho"], "echelle": cfg["echelle"]}
    if famille == "iacob_cma":
        return construire_genome(cfg["genome"], seed, d)
    raise ValueError(famille)


# lecture retardée
def _decaler(F, t):
    return np.vstack([np.zeros((t, F.shape[1]), F.dtype), F[:-t]])


def _lignes(F, taus, a, b):
    """Lignes a..b de [x(t), x(t − τ)…, 1] en float64, sans construire toute la matrice (WH : 30 000 × 6 322)."""
    parts = [F[a:b]]
    for k in taus:
        idx = np.arange(a, b) - k
        blk = np.zeros((b - a, F.shape[1]), F.dtype)
        ok = idx >= 0
        blk[ok] = F[idx[ok]]
        parts.append(blk)
    parts.append(np.ones((b - a, 1), F.dtype))
    return np.hstack(parts).astype(np.float64)


def ridge_paquets(F, taus, y, wash, tr_end, te0, te1, val, ridges=P7.RIDGES, pas=2000):
    """Même lecture ridge que P7.ridge_readout (λ choisi sur les 20 % finaux, réapprentissage sur tout
    le bloc), mais par matrices de Gram accumulées par paquets : mémoire en (K·N)², pas en T·K·N."""
    y = np.asarray(y, np.float64)
    v0 = wash + int(0.8 * (tr_end - wash))
    dim = F.shape[1] * (1 + len(taus)) + 1

    def gram(a, b):
        G, c = np.zeros((dim, dim)), np.zeros(dim)
        for i in range(a, b, pas):
            X = _lignes(F, taus, i, min(i + pas, b))
            G += X.T @ X
            c += X.T @ y[i:i + len(X)]
        return G, c
    G1, c1 = gram(wash, v0)
    G2, c2 = gram(v0, tr_end)
    yv = y[v0:tr_end]
    best_lam, best_val = ridges[0], np.inf
    for lam in ridges:
        w = np.linalg.solve(G1 + lam * np.eye(dim), c1)
        sse = w @ G2 @ w - 2 * w @ c2 + yv @ yv
        v = float(np.sqrt(max(sse, 0.0) / len(yv)) / np.std(yv))
        if v < best_val:
            best_val, best_lam = v, lam
    if val:
        return best_val
    w = np.linalg.solve(G1 + G2 + best_lam * np.eye(dim), c1 + c2)
    pred = np.concatenate([_lignes(F, taus, i, min(i + pas, te1)) @ w for i in range(te0, te1, pas)])
    return float(np.sqrt(np.mean((pred - y[te0:te1]) ** 2)) / np.std(y[te0:te1]))


def evaluer_lecture(res, data, taus, val=False):
    taus = [int(t) for t in taus if t > 1]
    t = data["type"]
    if t == "mnist":
        def feats(X):                        # par paquets : l'état de toutes les lignes est lourd
            out = []
            for i in range(0, len(X), 500):
                St = R.simuler(res, X[i:i + 500].reshape(-1, 28, 28).transpose(1, 0, 2))
                out.append(np.hstack([St[-1]] + [St[-1 - k] for k in taus if k < 28]))
            return np.vstack(out)
        Ftr, Fte = feats(data["Xtr"]), feats(data["Xte"])
        if val:
            v0 = int(0.8 * len(Ftr))
            acc = P7.mnist_readout(Ftr[:v0], data["ytr"][:v0], Ftr[v0:], data["ytr"][v0:])[0]
        else:
            acc = P7.mnist_readout(Ftr, data["ytr"], Fte, data["yte"])[0]
        return 1.0 - acc
    F = R.simuler(res, data["U"])
    if t == "mc":
        F = np.hstack([F] + [_decaler(F, k) for k in taus])
        return float(R.MC_KMAX) - float(R.mc_rapide(F, data["u"], R.MC_KMAX, val=val).sum())
    s = data["splits"]
    if val:
        v0 = s["wash"] + int(0.8 * (s["tr_end"] - s["wash"]))
        return ridge_paquets(F, taus, data["y"], s["wash"], v0, v0, s["tr_end"], False)
    return ridge_paquets(F, taus, data["y"], s["wash"], s["tr_end"], s["te0"], s["te1"], False)


def evaluer(famille, cfg, seed, data, d, S, val=False):
    res = construire(famille, cfg, seed, d)
    if famille == "lecture_retardee":
        return evaluer_lecture(res, data, S, val)
    return R.evaluer(res, data, val=val)


# Iacob-Dambre avec génome et CMA-ES
# gènes dans [0 ; 1], décodés vers leurs bornes
GENES = ([("wscale", (K_GR, K_GR), "log", 0.05, 3.0), ("conn", (K_GR, K_GR), "lin", 0.02, 1.0),
          ("fuite", (K_GR,), "lin", 0.05, 1.0), ("biais", (K_GR,), "lin", 0.0, 1.0),
          ("mu", (K_GR, 2), "lin", 0.0, 1.0), ("sigma", (K_GR, 2), "log", 0.02, 0.5),
          ("corr", (K_GR,), "lin", -0.9, 0.9), ("melange", (K_GR,), "lin", -2.0, 2.0),
          ("echelle", (1,), "log", 0.05, 2.0), ("vitesse", (1,), "log", 1.0, 400.0)])
DIM = sum(int(np.prod(s)) for _, s, _, _, _ in GENES)


def decoder(z):
    z = np.clip(np.asarray(z, float), 0, 1)
    out, i = {}, 0
    for nom, forme, kind, lo, hi in GENES:
        n = int(np.prod(forme))
        v = z[i:i + n]
        i += n
        v = lo * (hi / lo) ** v if kind == "log" else lo + (hi - lo) * v
        out[nom] = v.reshape(forme)
    return out


def construire_genome(z, seed, d):
    g = decoder(z)
    n = N0
    rng = np.random.default_rng(seed)
    w = np.exp(g["melange"]) / np.exp(g["melange"]).sum()
    grp = rng.choice(K_GR, size=n, p=w)
    p = np.zeros((n, 2))
    for k in range(K_GR):
        sx, sy = g["sigma"][k]
        c = g["corr"][k] * sx * sy
        cov = np.array([[sx * sx, c], [c, sy * sy]])
        idx = np.nonzero(grp == k)[0]
        p[idx] = rng.multivariate_normal(g["mu"][k], cov, size=len(idx))
    dist = np.sqrt(((p[:, None, :] - p[None, :, :]) ** 2).sum(-1))
    D = np.minimum(DCAP, 1 + np.floor(g["vitesse"][0] * dist)).astype(int)
    pc = g["conn"][grp[:, None], grp[None, :]]
    mask = rng.uniform(0, 1, (n, n)) < pc
    M = rng.normal(0, 1, (n, n)) * g["wscale"][grp[:, None], grp[None, :]] / np.sqrt(n * pc)
    i, j = np.nonzero(mask)
    L = int(D[i, j].max()) if len(i) else 1
    W_big = sparse.csr_matrix((M[i, j].astype(np.float32), (i, (D[i, j] - 1) * n + j)), shape=(n, L * n))
    Win = rng.uniform(-0.5, 0.5, size=(d, n)) * g["echelle"][0]
    if d == 2:                                                            # entrée [u, 1] : biais par groupe
        Win[-1] = rng.uniform(-0.5, 0.5, size=n) * g["biais"][grp]
    return {"prop": True, "N": n, "L": L, "Win": Win.astype(np.float32), "W_big": W_big, "W2": None,
            "paire": None, "fuite": g["fuite"][grp].astype(np.float32), "connexions": int(mask.sum())}


def cma(f, dim, lam, gens, rng, x0=None, sigma0=0.25):
    """CMA-ES standard (Hansen, 2016), dans [0 ; 1]^dim avec bornes par écrêtage. Renvoie le meilleur vu."""
    mu = lam // 2
    wts = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    wts /= wts.sum()
    mueff = 1 / (wts ** 2).sum()
    cc, cs = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim), (mueff + 2) / (dim + mueff + 5)
    c1 = 2 / ((dim + 1.3) ** 2 + mueff)
    cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
    damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs
    chiN = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
    m = np.full(dim, 0.5) if x0 is None else np.array(x0, float)
    sigma, C, pc, ps = sigma0, np.eye(dim), np.zeros(dim), np.zeros(dim)
    best = (np.inf, m.copy())
    for gen in range(gens):
        Dv, B = np.linalg.eigh(C)
        Dv = np.sqrt(np.maximum(Dv, 1e-20))
        Z = rng.normal(0, 1, (lam, dim))
        Y = Z @ (B * Dv).T
        X = m + sigma * Y
        fx = np.array([f(np.clip(x, 0, 1)) for x in X])
        for x, v in zip(X, fx):
            if v < best[0]:
                best = (float(v), np.clip(x, 0, 1).copy())
        o = np.argsort(fx)[:mu]
        yw = wts @ Y[o]
        m = m + sigma * yw
        Cinv_half = B @ np.diag(1 / Dv) @ B.T
        ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * Cinv_half @ yw
        hs = np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (gen + 1))) / chiN < 1.4 + 2 / (dim + 1)
        pc = (1 - cc) * pc + hs * np.sqrt(cc * (2 - cc) * mueff) * yw
        C = ((1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 - hs) * cc * (2 - cc) * C)
             + cmu * (Y[o].T * wts) @ Y[o])
        sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
        sigma = min(sigma, 0.5)
    return best


def regler(famille, data, seed, d, S, n_eval, tache=None):
    """Réglage sur la validation, puis test. Renvoie (test, val, cfg)."""
    rng = np.random.default_rng(10_000 + seed)
    if famille == "iacob_cma":
        def f(z):
            return evaluer("iacob_cma", {"genome": z.tolist()}, seed, data, d, S, val=True)
        v, z = cma(f, DIM, LAMBDA, GENERATIONS, rng)
        cfg = {"genome": z.tolist(), "evaluations": LAMBDA * GENERATIONS}
    else:
        best = None
        for _ in range(n_eval):
            cfg = tirer(famille, rng, tache)
            v = evaluer(famille, cfg, seed, data, d, S, val=True)
            if best is None or v < best[0]:
                best = (v, cfg)
        v, cfg = best
    test = evaluer(famille, cfg, seed, data, d, S)
    if famille == "iacob_cma":
        res = construire_genome(cfg["genome"], seed, d)
        cfg["connexions"], cfg["L"] = res["connexions"], res["L"]
    return float(test), float(v), cfg


def cellule(job):
    famille, tache, seed = job
    t0 = time.time()
    data = I.charger(tache, seed)
    d = I.d_entree(tache, data)
    S, _ = I.S_calcule(tache, seed)
    test, v, cfg = regler(famille, data, seed, d, S, I.n_eval(tache), tache)
    return (f"{tache}|{famille}|s{seed}",
            {"test": test, "val": v, "cfg": cfg, "S": S, "proto": PROTO}, time.time() - t0)


def run(taches):
    from multiprocessing import Pool
    out = R.lire(FICHIER)
    taches = taches or I.TOUTES
    jobs = [(f, t, s) for t in taches for s in I.seeds(t) for f in familles(t)
            if out["cells"].get(f"{t}|{f}|s{s}", {}).get("proto") != PROTO]
    jobs.sort(key=lambda j: j[0] == "iacob_cma")          # les longs à la fin
    procs = int(os.environ.get("IACOB_PROCS", max(1, min(4, (os.cpu_count() or 2) - 1))))
    print(f"{len(jobs)} cellules à calculer, {procs} processus, début {time.strftime('%H:%M')}", flush=True)
    t0 = time.time()
    with Pool(procs) as pool:
        for k, (key, cell, dt) in enumerate(pool.imap_unordered(cellule, jobs), 1):
            out["cells"][key] = cell
            R.ecrire(out, FICHIER)
            el = time.time() - t0
            print(f"[{k}/{len(jobs)}] {key} test={cell['test']:.4f} ({dt / 60:.1f} min) | "
                  f"{el / 60:.0f} min écoulées, reste ~{el / k * (len(jobs) - k) / 60:.0f} min", flush=True)
    out["meta"] = {"proto": PROTO, "familles": FAM_REF + ["iacob_cma"], "cma_taches": CMA_TACHES,
                   "cma": {"lambda": LAMBDA, "generations": GENERATIONS, "K": K_GR, "dcap": DCAP},
                   "smoke": R.ESSAI, "statut": "exploratoire : tâches déjà ouvertes"}
    R.ecrire(out, FICHIER)


NOMS = {"self_esn": "self-ESN", "lecture_retardee": "lecture retardée", "cycle": "cycle (SCR)",
        "orthogonal": "orthogonal", "iacob_cma": "Iacob-Dambre + CMA-ES"}


def tableau():
    out = R.lire(FICHIER)["cells"]
    ia = R.lire(I.FICHIER)["cells"]
    print("Erreur de test moyenne (capacité sur 300 pour mc) ; entre parenthèses : rapport à l'ESN à fuite, "
          "Wilcoxon apparié.\n")
    for t in I.TOUTES:
        ss = I.seeds(t)
        fams = familles(t)
        if any(f"{t}|{f}|s{s}" not in out for f in fams for s in ss):
            print(f"{t} : pas encore calculé\n")
            continue
        ref = {k: np.array(v) for k, v in I.references(t).items()}
        ref["Iacob-Dambre (positions au hasard)"] = np.array([ia[f"{t}|iacob|s{s}"]["test"] for s in ss])
        fam = {NOMS[f]: np.array([out[f"{t}|{f}|s{s}"]["test"] for s in ss]) for f in fams}
        esn = ref["ESN à fuite"]
        aff = (lambda v: f"{300 - v.mean():.1f}") if t == "mc" else (lambda v: f"{v.mean():.4g}")
        print(f"== {t} ({len(ss)} seeds)")
        for nom, v in {**ref, **fam}.items():
            extra = "" if nom == "ESN à fuite" else f"  (×{v.mean() / esn.mean():.3f}, p = {R.wilcoxon(v, esn):.3g})"
            print(f"  {nom:36s} {aff(v)}{extra}")
        rec = ref["recette"]
        for nom, v in fam.items():
            print(f"  recette contre {nom:22s} : ×{rec.mean() / v.mean():.3f}, p = {R.wilcoxon(rec, v):.3g}")
        if "iacob_cma" in fams:
            c = fam["Iacob-Dambre + CMA-ES"]
            r0 = ref["Iacob-Dambre (positions au hasard)"]
            print(f"  CMA-ES contre positions au hasard : ×{c.mean() / r0.mean():.3f}, p = {R.wilcoxon(c, r0):.3g}")
        print()


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "run":
        run(sys.argv[2:])
    elif cmd == "tableau":
        tableau()
    else:
        print(__doc__)
