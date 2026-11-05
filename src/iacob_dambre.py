"""Iacob et Dambre (2024), et trois mélanges avec la recette. Exploratoire : les tâches sont déjà ouvertes.

Iacob et Dambre placent les neurones dans le plan ; le délai d'une connexion j -> i est
proportionnel à leur distance, et chaque délai entier a ainsi sa propre matrice creuse. Ils
optimisent les positions par CMA-ES ; ici, positions tirées au hasard (leur point de départ).

Quatre familles, toutes à fuite, toutes au budget de 400 000 connexions :

  iacob       une matrice dense W (N = 632), D_ij = 1 + round((dmax − 1)·‖p_i − p_j‖ / √2),
              p_i uniforme dans [0 ; 1]² ; dmax réglé dans [2 ; 40].
  iacob_S     (A, leur modèle adapté) même matrice dense, mais le délai de chaque connexion est
              tiré uniformément dans le S calculé.
  hybride     (B, le mélange) leurs délais de propagation 1..dmax pour la mémoire proche, plus les
              tranches permutées (gratuites) de la recette pour les retards de S plus longs que
              dmax, plus la paire d'ordre 2 si la recette en a une (W₂ dense, N = 447 dans ce cas).
              Rayon ρ/√K par bloc, K = 1 + nombre de tranches permutées (+ 1 pour la paire).
  iacob_spectre (C) leur modèle, avec dmax = max(S calculé) au lieu d'être réglé.

    x(t) = (1 − a)·x(t−1) + a·tanh( W_in·u(t) + Σ_j W_ij·x_j(t − D_ij) [+ tranches] [+ W₂ (x(t−a)⊙x(t−b))] )

S calculé : celui que la recette a retenu sur la validation, à la même seed (results_p12_recette,
results_reel_v2/v3) ; pour NARMA-10, bande et îlots, le S calculé à K = 3 de la partie 3.3.
Réglage : recherche aléatoire sur la validation, mêmes bornes que l'ESN à fuite (ρ, fuite,
échelle d'entrée), plus dmax pour iacob et hybride. 60 évaluations sur les tâches de 4.3,
30 sur les séries de la partie 5 et sur NARMA-10, bande et îlots (comme leurs références).

    python src/iacob_dambre.py run [tâche ...]     # défaut : toutes ; IACOB_PROCS = processus
    python src/iacob_dambre.py tableau
Résultats : results/results_iacob.json (reprise automatique). ESSAI=1 : version réduite.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reservoir as R          # noqa: E402

PROTO = "iacob_2"
FICHIER = (R.RES / "essai" if R.ESSAI else R.RES) / "results_iacob.json"
BUDGET = 400_000
FAMILLES = ["iacob", "iacob_S", "hybride", "iacob_spectre"]
DEV = ["narma30", "silverbox", "wh", "mnist", "mc"]                       # partie 4.3
REEL_V2 = ["electricite_h24", "trafic_h24"]                               # partie 5, test 2
REEL_V3 = ["energie_charge_h24", "energie_prix_h24", "solaire_h24", "meteo_T_h24",
           "meteo_rayonnement_h24"]                                      # partie 5, test 3
SYNTH = ["narma10", "bande|0.5|0", "ilots|0.5|0"]                          # partie 3.3
TOUTES = DEV + SYNTH + REEL_V2 + REEL_V3


def seeds(tache):
    if R.ESSAI:
        return [0, 1]
    return list(range(5)) if tache in SYNTH else list(range(10))


def n_eval(tache):
    if R.ESSAI:
        return 6
    return 60 if tache in DEV else 30


# réservoirs
def _bloc_dense(M, D, n):
    """Entrées de M réparties dans les blocs W_big[:, (D_ij − 1)·n + j]."""
    i, j = np.indices((n, n))
    return M.ravel().astype(np.float32), i.ravel(), ((D - 1) * n + j).ravel()


def construire(famille, cfg, seed, d, S=None, paire=None):
    S = sorted(int(t) for t in (S or [1]))
    rng = np.random.default_rng(seed)
    perm_taus = []
    if famille == "hybride":
        dmax = int(cfg["dmax"])
        perm_taus = [t for t in S if t > dmax]
        avec_paire = paire is not None
        K = 1 + len(perm_taus) + (1 if avec_paire else 0)
        n = R.N_pour_budget(BUDGET, 1 + (1 if avec_paire else 0))
    else:
        dmax = {"iacob": int(cfg.get("dmax", 2)), "iacob_spectre": max(2, max(S)), "iacob_S": max(S)}[famille]
        avec_paire, K, n = False, 1, R.N_pour_budget(BUDGET, 1)
    sc = cfg["rho"] / np.sqrt(K)
    Win = (rng.uniform(-0.5, 0.5, size=(d, n)) * cfg["echelle"]).astype(np.float32)
    M = rng.normal(0, 1, (n, n))
    M *= sc / R.rayon_spectral(M)
    if famille == "iacob_S":
        D = rng.choice(np.array(S), size=(n, n))
    else:
        p = rng.uniform(0, 1, (n, 2))
        dist = np.sqrt(((p[:, None, :] - p[None, :, :]) ** 2).sum(-1))
        D = 1 + np.rint((dmax - 1) * dist / np.sqrt(2)).astype(int)
    L = max([dmax] + S + (list(paire) if avec_paire else []))
    v0, r0, c0 = _bloc_dense(M, D, n)
    vals, rows, cols = [v0], [r0], [c0]
    for t in perm_taus:                       # tranches permutées gratuites : W_τ x = sc · x[perm]
        perm = rng.permutation(n)
        vals.append(np.full(n, sc, np.float32))
        rows.append(np.arange(n))
        cols.append((t - 1) * n + perm)
    W_big = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                              shape=(n, L * n))
    W2 = None
    if avec_paire:
        M2 = rng.normal(0, 1, (n, n))
        W2 = (M2 * (sc / R.rayon_spectral(M2))).astype(np.float32)
    return {"prop": True, "N": n, "L": L, "Win": Win, "W_big": W_big, "W2": W2,
            "paire": tuple(paire) if avec_paire else None, "fuite": np.float32(cfg["fuite"])}


def simuler_prop(res, U, final=False):
    seq = U.ndim == 2
    if seq:
        U = U[:, None, :]
    T, B, _ = U.shape
    n, L, a = res["N"], res["L"], res["fuite"]
    hist = np.zeros((L, B, n), np.float32)          # hist[k] = x(t − 1 − k)
    x = np.zeros((B, n), np.float32)
    sortie = None if final else np.empty((T, B, n), np.float32)
    Wb, W2, pr = res["W_big"], res["W2"], res["paire"]
    for t in range(T):
        drive = U[t] @ res["Win"] + (Wb @ hist.transpose(0, 2, 1).reshape(L * n, B)).T
        if pr is not None:
            drive += (hist[pr[0] - 1] * hist[pr[1] - 1]) @ W2.T
        z = np.tanh(drive)
        x = (1.0 - a) * x + a * z
        hist[1:] = hist[:-1]
        hist[0] = x
        if not final:
            sortie[t] = x
    if final:
        return x
    return sortie[:, 0, :] if seq else sortie


_simuler = R.simuler


def _simuler_tout(res, U, final=False):
    return simuler_prop(res, U, final) if res.get("prop") else _simuler(res, U, final)


R.simuler = _simuler_tout      # R.evaluer passe aussi par ces réservoirs


def tirer(famille, rng):
    cfg = {"rho": float(rng.uniform(0.3, 1.1)), "fuite": float(rng.uniform(0.05, 1.0)),
           "echelle": float(np.exp(rng.uniform(np.log(0.05), np.log(2.0))))}
    dmax = int(rng.integers(2, 41))           # tiré pour toutes : mêmes tirages de ρ, fuite, échelle
    if famille in ("iacob", "hybride"):
        cfg["dmax"] = dmax
    return cfg


# données et S calculé
def charger(tache, seed):
    if tache in DEV:
        return R.charger(tache, seed)
    if tache in SYNTH:
        import controle_conception as CC
        return CC.donnees(tache, seed)
    if tache in REEL_V2:
        import reel_v2
        return reel_v2.charger_v2(tache)
    import reel_v3
    return reel_v3.charger_v2(tache)


def d_entree(tache, data):
    return 28 if tache == "mnist" else data["U"].shape[1]


def S_calcule(tache, seed):
    """(S, paire) retenus sur la validation par la recette, à cette seed (jamais le test)."""
    if tache in DEV:
        c = R.lire(R.RES / "results_p12_recette.json")["cells"][f"{tache}|s{seed}"]["cfg"]
        return c["S"], c.get("paire")
    if tache in SYNTH:
        c = R.lire(R.RES / "results_controle_reglage.json")["cells"]
        return c[f"{tache}|calcule_K3|s{seed}"]["cfg"]["S"], None
    f = "results_reel_v2.json" if tache in REEL_V2 else "results_reel_v3.json"
    c = R.lire(R.RES / f)["cells"][f"{tache}|recette|s{seed}"]["cfg"]
    return c["S"], c.get("paire")


# calcul
def cellule(job):
    """Une famille, une seed, une tâche : réglage sur la validation, puis test."""
    famille, tache, seed = job
    t0 = time.time()
    data = charger(tache, seed)
    d = d_entree(tache, data)
    S, paire = S_calcule(tache, seed)
    rng = np.random.default_rng(10_000 + seed)
    best = None
    for _ in range(n_eval(tache)):
        cfg = tirer(famille, rng)
        v = R.evaluer(construire(famille, cfg, seed, d, S, paire), data, val=True)
        if best is None or v < best[0]:
            best = (v, cfg)
    v, cfg = best
    test = R.evaluer(construire(famille, cfg, seed, d, S, paire), data)
    return (f"{tache}|{famille}|s{seed}",
            {"test": float(test), "val": float(v), "cfg": cfg, "S": S, "paire": paire, "proto": PROTO},
            time.time() - t0)


def run(taches):
    from multiprocessing import Pool
    out = R.lire(FICHIER)
    taches = taches or TOUTES
    jobs = [(f, t, s) for t in taches for s in seeds(t) for f in FAMILLES
            if out["cells"].get(f"{t}|{f}|s{s}", {}).get("proto") != PROTO]
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
    out["meta"] = {"proto": PROTO, "budget": BUDGET, "familles": FAMILLES, "smoke": R.ESSAI,
                   "modele": "Iacob et Dambre 2024 (positions au hasard, sans CMA-ES) et trois mélanges avec la recette",
                   "statut": "exploratoire : tâches déjà ouvertes"}
    R.ecrire(out, FICHIER)


# tableau
def references(tache):
    """{nom: erreurs de test par seed} des familles déjà calculées, mêmes seeds, même budget."""
    ss = seeds(tache)
    if tache in DEV:
        e = R.lire(R.RES / "results_p12_equite.json")["cells"]
        r = R.lire(R.RES / "results_p12_recette.json")["cells"]
        return {"ESN à fuite": [e[f"{tache}|esn_fuite|s{s}"]["test"] for s in ss],
                "recette": [r[f"{tache}|s{s}"]["test"] for s in ss]}
    if tache in SYNTH:
        c = R.lire(R.RES / "results_controle_reglage.json")["cells"]
        return {"ESN à fuite": [c[f"{tache}|esn_fuite|s{s}"]["test"] for s in ss],
                "recette": [c[f"{tache}|calcule_K3|s{s}"]["test"] for s in ss],
                "{1..10}": [c[f"{tache}|uniforme_10|s{s}"]["test"] for s in ss]}
    c = R.lire(R.RES / ("results_reel_v2.json" if tache in REEL_V2 else "results_reel_v3.json"))["cells"]
    return {"ESN à fuite": [c[f"{tache}|esn_fuite|s{s}"]["test"] for s in ss],
            "recette": [c[f"{tache}|recette|s{s}"]["test"] for s in ss]}


def tableau():
    out = R.lire(FICHIER)["cells"]
    noms = {"iacob": "Iacob-Dambre", "iacob_S": "A (délais dans S)", "hybride": "B (mélange)",
            "iacob_spectre": "C (dmax du spectre)"}
    print("Erreur de test moyenne (capacité sur 300 pour mc) ; entre parenthèses : rapport à l'ESN à fuite, "
          "Wilcoxon apparié.\n")
    for t in TOUTES:
        ss = seeds(t)
        if any(f"{t}|{f}|s{s}" not in out for f in FAMILLES for s in ss):
            print(f"{t} : pas encore calculé\n")
            continue
        ref = {k: np.array(v) for k, v in references(t).items()}
        fam = {noms[f]: np.array([out[f"{t}|{f}|s{s}"]["test"] for s in ss]) for f in FAMILLES}
        esn = ref["ESN à fuite"]
        aff = (lambda v: f"{300 - v.mean():.1f}") if t == "mc" else (lambda v: f"{v.mean():.4g}")
        print(f"== {t} ({len(ss)} seeds)")
        for nom, v in {**ref, **fam}.items():
            extra = "" if nom == "ESN à fuite" else f"  (×{v.mean() / esn.mean():.3f}, p = {R.wilcoxon(v, esn):.3g})"
            print(f"  {nom:22s} {aff(v)}{extra}")
        b, rec, ia = fam["B (mélange)"], ref["recette"], fam["Iacob-Dambre"]
        print(f"  B contre la recette : ×{b.mean() / rec.mean():.3f}, p = {R.wilcoxon(b, rec):.3g} ; "
              f"B contre Iacob-Dambre : ×{b.mean() / ia.mean():.3f}, p = {R.wilcoxon(b, ia):.3g}\n")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "run":
        run(sys.argv[2:])
    elif cmd == "tableau":
        tableau()
    else:
        print(__doc__)
