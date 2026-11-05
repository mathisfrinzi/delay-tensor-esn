"""Partie 12 : synthèse.

Corrections de protocole (partie 14, 2026-10-29) :
  - 10 seeds (à 5 paires, le Wilcoxon bilatéral plafonne à p = 0,0625) ;
  - espace de recherche défini A PRIORI (uniformes, puissances de 2, géométrique,
    pairs, bandes, {1, d} sur une grille) : les ensembles « élus » des parties 2 et 5,
    trouvés sur ces mêmes tâches, ne sont plus proposés à la recherche ; les anciennes
    cellules sont conservées sous « |ancien » pour mesurer l'écart ;
  - 12.3 : le réglage se fait sur les seeds 100 à 104, jamais testées ;
  - MC étendu à k = 300 (la mesure saturait à 60) et réglé sur la validation ;
  - recette : critère du profil remplacé par un choix sur la validation entre S
    calculé (K = 2, 3 et 5) et {1, …, 10}, avec ρ (jusqu'à 0,1), fuite et échelle
    d'entrée réglés comme les autres familles (leçon de la partie 12.1).

12.1  equite     Réglage équitable (60 évaluations de recherche aléatoire par famille, sur
                 la validation, 10 seeds), la fuite étant ouverte à toutes les familles :
                 ESN à fuite, DTS-ESN, réservoir à retards avec fuite, réservoir à retards
                 sans fuite. Tranches denses, permutées ou diagonales (budget honnête).
12.2  recette    La recette par le calcul (spectre, S par les sauts, paire d'ordre 2,
                 tranches permutées), choisie sur la validation parmi S calculé (K = 2, 3
                 et 5) et {1, …, 10}, avec ρ, fuite et échelle d'entrée réglés (mêmes
                 tirages pour chaque S) : 60 évaluations, comme les familles réglées.
12.3  budgets    Erreur selon le budget de connexions (25 000 à 1 600 000) et selon le
                 nombre de paramètres de lecture, sur 7 tâches, pour : ESN à fuite et
                 DTS-ESN réglés (30 évaluations), dix retards uniformes, la recette, NG-RC.

    python src/p12_synthese.py equite
    python src/p12_synthese.py recette
    python src/p12_synthese.py budgets
    python src/p12_synthese.py figures
Reprise automatique : results/results_p12_*.json.
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reservoir as R  # noqa: E402
import conception as C  # noqa: E402
import p11_trous_mous as P11  # noqa: E402

RES = R.RES
# 14.1 : 10 seeds (à 5 paires, le Wilcoxon bilatéral plafonne à p = 0,0625)
SEEDS = [0, 1] if R.ESSAI else list(range(10))
# 14.1 : seeds de réglage de 12.3, jamais testées (la seed 0 n'est plus réglée puis testée)
SEEDS_REGLAGE = [0, 1] if R.ESSAI else [100, 101, 102, 103, 104]
PROTO = "14_1_apriori_mc300"   # marqueur de protocole : sans lui, une cellule est recalculée
BUDGET = 400_000
BUDGETS = [25_000, 100_000] if R.ESSAI else [25_000, 100_000, 400_000, 1_600_000]
N_EQUITE = 10 if R.ESSAI else 60
N_BUDGET = 6 if R.ESSAI else 30
TACHES_EQUITE = ["narma30", "silverbox", "wh", "mnist", "mc"]
TACHES_BUDGET = ["narma10", "narma30", "mc", "parite2", "silverbox", "wh", "mnist"]
FICHIER = {k: RES / f"results_p12_{k}.json" for k in ("equite", "recette", "budgets")}

# 14.1 : espace A PRIORI, fixé sans regarder les tâches — uniformes, puissances de 2,
# géométrique, pairs, bandes contiguës, {1, d} sur une grille régulière. Les ensembles
# « élus » ([4, 7], [1, 30, 31], [1, 2, 30, 31], …) trouvés sur NARMA/Silverbox/WH
# dans les parties 2 et 5 sont retirés : la recherche ne part plus avec une
# connaissance de la tâche que l'ESN à fuite n'a pas.
ENSEMBLES = [list(range(1, k + 1)) for k in (2, 3, 5, 10, 20)] + [
    [1, 2, 4, 8, 16], [1, 2, 4, 8, 16, 32], [1, 3, 9], [2, 4, 6, 8, 10],
    list(range(2, 11)), list(range(4, 11)), list(range(2, 21))] + [[1, d] for d in (5, 10, 15, 20, 25, 30, 40)]


# construction
def construire(cfg, budget, seed, d):
    """cfg : S, kind ('dense' | 'perm' | 'diag'), paire, rho, fuite, dts_amin, echelle.
    Avec kind ≠ 'dense', la plus petite tranche reste dense et les autres sont gratuites :
    tout le budget va à cette tranche (et à W₂ s'il y a une paire)."""
    S = sorted(cfg["S"])
    kind = cfg.get("kind", "dense")
    kinds = {t: (kind if (kind != "dense" and t != S[0]) else "dense") for t in S}
    n_dense = sum(1 for t in S if kinds[t] == "dense") + (1 if cfg.get("paire") else 0)
    N = R.N_pour_budget(budget, n_dense)
    return R.construire(seed, S, N, cfg["rho"], kinds=kinds, d=d, echelle=cfg.get("echelle", 1.0),
                        fuite=cfg.get("fuite", 1.0), dts_amin=cfg.get("dts_amin"),
                        paire=tuple(cfg["paire"]) if cfg.get("paire") else None)


def tirer(famille, rng):
    log_u = lambda lo, hi: float(np.exp(rng.uniform(np.log(lo), np.log(hi))))  # noqa: E731
    cfg = {"rho": float(rng.uniform(0.3, 1.1)), "echelle": log_u(0.05, 2.0)}
    if famille == "esn_fuite":
        cfg.update(S=[1], fuite=float(rng.uniform(0.05, 1.0)))
    elif famille == "esn_entree":
        # 14.1/14.3 : référence équitable — même ESN à fuite (mêmes tirages de
        # ρ, fuite et échelle), l'entrée reçoit u(t−τ) pour tout τ du S calculé
        # et le produit de la paire (voir augmenter/reference_entree).
        cfg.update(S=[1], fuite=float(rng.uniform(0.05, 1.0)))
    elif famille == "dts":
        cfg.update(S=[1], dts_amin=log_u(0.01, 0.3))
    elif famille in ("tenseur_fuite", "tenseur_sans_fuite"):
        cfg.update(S=ENSEMBLES[rng.integers(len(ENSEMBLES))], kind=["dense", "perm", "diag"][rng.integers(3)],
                   fuite=float(rng.uniform(0.05, 1.0)) if famille == "tenseur_fuite" else 1.0)
    else:
        raise ValueError(famille)
    return cfg


def d_de(tache, data):
    """Dimension d'entrée : celle des données (augmentées ou non), 28 pour MNIST."""
    return data["U"].shape[1] if tache != "mnist" else R.d_entree(tache)


def augmenter(data, S, paire=None):
    """14.1/14.3, référence équitable : l'entrée reçoit u(t) et u(t−τ) pour tout
    τ ∈ S (zéros avant t = τ), plus le produit u(t−a)·u(t−b) de la paire —
    l'information vieille disponible en entrée, pas seulement dans la récurrence."""
    U = data["U"]
    u = U[:, 0].astype(np.float64)
    cols = [U[:, 0]]
    for tau in sorted(set(int(t) for t in S)):
        c = np.zeros(len(u), np.float64)
        c[tau:] = u[:-tau]
        cols.append(c)
    if paire:
        a, b = int(paire[0]), int(paire[1])
        ca = np.zeros(len(u), np.float64)
        cb = np.zeros(len(u), np.float64)
        ca[a:] = u[:-a]
        cb[b:] = u[:-b]
        cols.append(ca * cb)
    cols.append(np.ones(len(u)))
    out = dict(data)
    out["U"] = np.stack(cols, 1).astype(np.float32)
    return out


def reference_entree(tache):
    """S et paire de la référence d'entrée : les retards calculés par la recette
    (sauts K = 2) sur la partie d'estimation des données — a priori, sans test.
    None pour MNIST (entrée en image, pas de retards temporels)."""
    if tache == "mnist":
        return None, None
    if tache == "mc":
        w = np.ones(61)
        w[0] = 0
        w /= w.sum()
        return C.support_sauts(w, 2, 60, 60)[0], None
    d = R.charger(tache, 0)
    s = d["splits"]
    u, y = d["U"][:s["tr_end"], 0], d["y"][:s["tr_end"]]
    spec = C.spectre(u, y, lmax=40)
    w = C.poids_retards(spec)
    w = C.enveloppe(spec) if w is None else w
    paire = C.paire_produit(spec)
    return C.support_sauts(w, 2, len(w) - 1, min(60, len(w) - 1))[0], paire


def regler(tache, data, famille, budget, seed, n):
    """Recherche aléatoire de n évaluations sur la validation ; renvoie (test, cfg)."""
    rng = np.random.default_rng(10_000 + seed)
    d = d_de(tache, data)
    best = None
    for _ in range(n):
        cfg = tirer(famille, rng)
        v = R.evaluer(construire(cfg, budget, seed, d), data, val=True)
        if best is None or v < best[0]:
            best = (v, cfg)
    return best[1]


def tester(tache, data, cfg, budget, seed):
    return R.evaluer(construire(cfg, budget, seed, d_de(tache, data)), data)


def seeds_reglage(tache):
    """14.1 : seeds de réglage de 12.3, jamais testées. Les tâches dont les données ne
    dépendent pas de la seed (silverbox, wh, mnist) n'ont qu'une réalisation : le
    réglage sur la seed 0 ne voit que son segment de validation, jamais son test."""
    return [0] if tache in ("silverbox", "wh", "mnist") else SEEDS_REGLAGE


def regler_multi(tache, famille, budget, seeds, n):
    """Recherche aléatoire de n évaluations ; critère = moyenne des erreurs de
    validation sur les seeds de réglage (jamais testées)."""
    rng = np.random.default_rng(11_000)
    d = R.d_entree(tache)
    datas = {s: R.charger(tache, s) for s in seeds}
    best = None
    for _ in range(n):
        cfg = tirer(famille, rng)
        v = float(np.mean([R.evaluer(construire(cfg, budget, s, d), datas[s], val=True)
                           for s in seeds]))
        if best is None or v < best[0]:
            best = (v, cfg)
    return best[1]


def recette_regler_multi(tache, seeds, budget, cands):
    """14.1 : choix du candidat de la recette sur la moyenne des validations des
    seeds de réglage (12.3), jamais sur le test."""
    d = R.d_entree(tache)
    datas = {s: R.charger(tache, s) for s in seeds}
    vals = [float(np.mean([R.evaluer(construire(c, budget, s, d), datas[s], val=True)
                           for s in seeds])) for c in cands]
    return cands[int(np.argmin(vals))]


# 12.1 équité
def equite():
    out = R.lire(FICHIER["equite"])
    familles = ["esn_fuite", "esn_entree", "dts", "tenseur_fuite", "tenseur_sans_fuite"]
    for tache in TACHES_EQUITE:
        S_ref, paire_ref = reference_entree(tache)      # référence d'entrée (14.1/14.3)
        # gardé pour le test final et les courbes d'apprentissage, qui doivent
        # reconstruire la même entrée augmentée
        out["cells"][f"{tache}|esn_entree|ref"] = {"S": S_ref, "paire": list(paire_ref) if paire_ref else None,
                                                   "proto": PROTO}
        for seed in SEEDS:
            data = R.charger(tache, seed)
            data_ref = augmenter(data, S_ref, paire_ref) if S_ref else None
            for fam in familles:
                key = f"{tache}|{fam}|s{seed}"
                cell = out["cells"].get(key)
                if isinstance(cell, dict) and cell.get("proto") != PROTO:
                    if fam.startswith("tenseur") or tache == "mc":
                        # 14.1 : espace a priori (familles tenseur) ou métrique MC à
                        # k = 300 : recalcul ; l'ancienne cellule est conservée sous
                        # « |ancien » pour mesurer l'écart avec la version précédente.
                        out["cells"][key + "|ancien"] = cell
                        out["cells"].pop(key, None)
                        cell = None
                    else:
                        cell["proto"] = PROTO      # cellule inchangée, simplement marquée
                        R.ecrire(out, FICHIER["equite"])
                if cell is not None:
                    continue
                dt = data_ref if fam == "esn_entree" else data
                cfg = regler(tache, dt, fam, BUDGET, seed, N_EQUITE)
                out["cells"][key] = {"test": tester(tache, dt, cfg, BUDGET, seed), "cfg": cfg,
                                     "proto": PROTO}
                R.ecrire(out, FICHIER["equite"])
                print(key, round(out["cells"][key]["test"], 4), flush=True)
    out["meta"] = {"budget": BUDGET, "n_evaluations": N_EQUITE, "seeds": SEEDS, "ensembles": ENSEMBLES,
                   "proto": PROTO, "espace": "a priori (14.1), plus aucun élu des parties 2 et 5",
                   "reference_entree": "esn_entree : u(t−τ) pour tout τ du S calculé + produit de la paire (14.1/14.3)",
                   "erreur": "NRMSE ; taux d'erreur (mnist) ; 300 − capacité (mc)"}
    R.ecrire(out, FICHIER["equite"])


# 12.2 recette
def candidats_recette(tache, budget, n_eval=None):
    """n_eval configurations (N_EQUITE par défaut), toutes calculées à partir de la
    tâche (14.1/14.4, parties 12.1 et 12.3) : S par les sauts (K = 2, 3 et 5) et
    {1, …, 10} (spectres étalés), chacun avec les mêmes n_eval/4 tirages de ρ
    (uniforme sur [0,1 ; 1,1]), de la fuite et de l'échelle d'entrée ; tranches
    permutées, paire si le spectre en désigne une. Le critère du profil (partie 11),
    le pire des six méthodes de conception, est retiré ; le choix se fait sur la
    validation, jamais sur le test."""
    n_eval = N_EQUITE if n_eval is None else n_eval
    if tache == "mnist":
        w = np.zeros(28)
        w[1:] = 1.0 / 27                 # couvrir les 27 lignes précédentes
        spec = None
        paire = None
        E = None
    elif tache == "mc":
        w = np.ones(61)
        w[0] = 0
        w /= w.sum()
        spec, paire, E = None, None, w
    else:
        d = R.charger(tache, 0)
        s = d["splits"]
        u, y = d["U"][:s["tr_end"], 0], d["y"][:s["tr_end"]]
        spec = C.spectre(u, y, lmax=40)
        w = C.poids_retards(spec)
        w = C.enveloppe(spec) if w is None else w
        paire = C.paire_produit(spec)
        E = C.enveloppe(spec)
    cands = []
    SS = {f"sauts K={K}": C.support_sauts(w, K, len(w) - 1, min(60, len(w) - 1))[0] for K in (2, 3, 5)}
    SS["uniforme {1..10}"] = list(range(1, 11))     # 14.4 : remplace le critère du profil
    # Partie 12 (12.1) : sans réglage de la fuite et de l'échelle d'entrée, le S calculé
    # perdait sur Silverbox et Wiener-Hammerstein ; réglé comme les autres familles, il y
    # gagne. La recette règle donc ρ, la fuite et l'échelle d'entrée, avec les MÊMES
    # tirages pour chaque S (comparaison appariée), ρ étendu vers 0,1 (14.1). Budget
    # total = n_eval évaluations, comme les familles réglées auxquelles elle est comparée.
    n_tirages = max(1, n_eval // len(SS))
    rng = np.random.default_rng(12_000)
    tirages = [{"rho": float(rng.uniform(0.1, 1.1)),
                "fuite": float(rng.uniform(0.05, 1.0)),
                "echelle": float(np.exp(rng.uniform(np.log(0.05), np.log(2.0))))} for _ in range(n_tirages)]
    for origine, S in SS.items():
        for t in tirages:
            cands.append(dict(t, S=S, kind="perm", paire=list(paire) if paire else None, origine=origine))
    return cands, (spec["test_plein"] if spec else None)


def recette_regler(tache, data, budget, seed, cands):
    d = R.d_entree(tache)
    vals = [R.evaluer(construire(c, budget, seed, d), data, val=True) for c in cands]
    return cands[int(np.argmin(vals))]


def recette():
    out = R.lire(FICHIER["recette"])
    for tache in TACHES_EQUITE:
        mk = f"{tache}|meta"
        if out["cells"].get(mk, {}).get("proto") != PROTO:
            cands, oracle = candidats_recette(tache, BUDGET)
            out["cells"][mk] = {"candidats": cands, "oracle_test": oracle, "proto": PROTO}
            R.ecrire(out, FICHIER["recette"])
        cands = out["cells"][mk]["candidats"]
        for seed in SEEDS:
            key = f"{tache}|s{seed}"
            if out["cells"].get(key, {}).get("proto") == PROTO:
                continue
            data = R.charger(tache, seed)
            cfg = recette_regler(tache, data, BUDGET, seed, cands)
            out["cells"][key] = {"test": tester(tache, data, cfg, BUDGET, seed), "cfg": cfg,
                                 "proto": PROTO}
            R.ecrire(out, FICHIER["recette"])
            print(key, round(out["cells"][key]["test"], 4), cfg.get("origine"), flush=True)
    out["meta"] = {"budget": BUDGET, "seeds": SEEDS, "evaluations": N_EQUITE, "proto": PROTO,
                   "reglage": "ρ ∈ [0,1 ; 1,1], fuite ∈ [0,05 ; 1], échelle log-uniforme [0,05 ; 2], "
                              "mêmes tirages pour chaque S (partie 12.1)",
                   "criteres": "S sauts K=2, 3, 5 et {1..10}, sans profil (14.1/14.4, 12.3)"}
    R.ecrire(out, FICHIER["recette"])


# 12.3 budgets
def budgets():
    out = R.lire(FICHIER["budgets"])
    for tache in TACHES_BUDGET:
        for B in BUDGETS:
            # 14.1 : réglage sur les seeds 100 à 104 (jamais testées), puis test
            # sur les 10 seeds ; la seed 0 n'est plus réglée puis testée.
            mk = f"{tache}|B{B}|meta"
            if (out["cells"].get(mk, {}).get("proto") != PROTO
                    or "tenseur_fuite" not in out["cells"].get(mk, {})):
                sr = seeds_reglage(tache)
                datas = {s: R.charger(tache, s) for s in sr}
                m = {"proto": PROTO, "seeds_reglage": sr}
                # 14.1 (refaire 13.3) : le réservoir à retards réglé avec fuite,
                # qui gagne en 13.1, entre dans les courbes de budget.
                for fam in ("esn_fuite", "dts", "tenseur_fuite"):
                    m[fam] = regler_multi(tache, fam, B, sr, N_BUDGET)
                best = None
                for rho in (0.5, 0.9, 0.99):
                    c = {"S": list(range(1, 11)), "kind": "dense", "rho": rho}
                    v = float(np.mean([R.evaluer(construire(c, B, s, R.d_entree(tache)), datas[s],
                                                 val=True) for s in sr]))
                    if best is None or v < best[0]:
                        best = (v, c)
                m["uniforme_10"] = best[1]
                cands, _ = candidats_recette(tache, B, N_BUDGET)
                m["recette"] = recette_regler_multi(tache, sr, B, cands)
                if datas[sr[0]]["type"] == "seq":
                    bestn = None
                    for D in (5, 10, 20, 30, 40):
                        for prod in (False, True):
                            v = float(np.mean([R.ngrc_evaluer(datas[s], D, prod, val=True) for s in sr]))
                            if bestn is None or v < bestn[0]:
                                bestn = (v, {"D": D, "produits": prod})
                    m["ngrc"] = bestn[1]
                out["cells"][mk] = m
                R.ecrire(out, FICHIER["budgets"])
            m = out["cells"][mk]
            for seed in SEEDS:
                data = R.charger(tache, seed)
                for fam, cfg in m.items():
                    if fam in ("proto", "seeds_reglage"):
                        continue
                    key = f"{tache}|B{B}|{fam}|s{seed}"
                    cell = out["cells"].get(key)
                    if isinstance(cell, dict) and cell.get("proto") != PROTO:
                        # 14.1 : l'ancien réglage se faisait sur la seed 0 (aussi
                        # testée) ; recalcul, ancienne cellule conservée sous « |ancien ».
                        out["cells"][key + "|ancien"] = cell
                        out["cells"].pop(key, None)
                        cell = None
                    if cell is not None:
                        continue
                    if fam == "ngrc":
                        err = R.ngrc_evaluer(data, cfg["D"], cfg["produits"])
                        D = cfg["D"]
                        npar = D + (D * (D + 1) // 2 if cfg["produits"] else 0) + 1
                        out["cells"][key] = {"test": err, "lecture": npar, "connexions": 0,
                                             "proto": PROTO}
                    else:
                        t0 = time.perf_counter()
                        res = construire(cfg, B, seed, R.d_entree(tache))
                        err = R.evaluer(res, data)
                        out["cells"][key] = {"test": err, "lecture": res["N"] + 1,
                                             "connexions": R.connexions(res),
                                             "couts": R.couts(res),
                                             "secondes": round(time.perf_counter() - t0, 2),
                                             "proto": PROTO}
                    R.ecrire(out, FICHIER["budgets"])
            print("budget fait", tache, B, flush=True)
    out["meta"] = {"budgets": BUDGETS, "seeds": SEEDS, "n_reglage": N_BUDGET,
                   "seeds_reglage": "100 à 104 pour les tâches dont les données dépendent de la seed (14.1)",
                   "proto": PROTO,
                   "erreur": "NRMSE ; taux d'erreur (mnist) ; 300 − capacité (mc)"}
    R.ecrire(out, FICHIER["budgets"])


# figures
def figures():
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["svg.fonttype"] = "none"
    import matplotlib.pyplot as plt
    COL = {"esn_fuite": "#eb6834", "esn_entree": "#c8442c", "dts": "#eda100", "tenseur_fuite": "#1baf7a",
           "tenseur_sans_fuite": "#2a78d6", "uniforme_10": "#2a78d6", "recette": "#1baf7a", "ngrc": "#52514e"}
    LAB = {"esn_fuite": "ESN à fuite", "esn_entree": "ESN, retards en entrée", "dts": "DTS-ESN",
           "tenseur_fuite": "retards, avec fuite", "tenseur_sans_fuite": "retards, sans fuite",
           "uniforme_10": "dix retards uniformes", "recette": "recette", "ngrc": "NG-RC"}
    INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#d9d8d3", "#fcfcfb"

    def style(ax):
        ax.set_facecolor(SURF)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=INK2, labelsize=9)
        ax.grid(axis="y", color=GRID, lw=0.8)

    e = R.lire(FICHIER["equite"])
    r = R.lire(FICHIER["recette"])
    if e["cells"]:
        fams = ["esn_fuite", "esn_entree", "dts", "tenseur_sans_fuite", "tenseur_fuite"]
        fig, axs = plt.subplots(1, len(TACHES_EQUITE), figsize=(16, 4.2))
        fig.patch.set_facecolor(SURF)
        for ax, t in zip(axs, TACHES_EQUITE):
            style(ax)
            for i, f in enumerate(fams + ["recette"]):
                src = r if f == "recette" else e
                pat = (lambda s: f"{t}|s{s}") if f == "recette" else (lambda s: f"{t}|{f}|s{s}")
                v = [src["cells"].get(pat(s), {}).get("test") for s in SEEDS]
                v = [x for x in v if x is not None]
                if v:
                    ax.bar(i, np.mean(v), yerr=np.std(v), color=COL[f], ecolor=INK2, capsize=2, label=LAB[f])
            ax.set_xticks([])
            ax.set_title(t, fontsize=10.5, color=INK, loc="left")
        axs[0].legend(fontsize=8, frameon=False)
        fig.tight_layout()
        fig.savefig(RES / "partie12_equite.svg", facecolor=SURF)

    b = R.lire(FICHIER["budgets"])
    if b["cells"]:
        for axe, nom in (("connexions", "partie12_budgets.svg"), ("lecture", "partie12_lecture.svg")):
            fig, axs = plt.subplots(2, 4, figsize=(17, 8))
            fig.patch.set_facecolor(SURF)
            for ax, t in zip(axs.flat, TACHES_BUDGET):
                style(ax)
                for fam in ("esn_fuite", "dts", "tenseur_fuite", "uniforme_10", "recette", "ngrc"):
                    xs, ys = [], []
                    for B in BUDGETS:
                        v = [b["cells"].get(f"{t}|B{B}|{fam}|s{s}") for s in SEEDS]
                        v = [x for x in v if x is not None]
                        if v:
                            xs.append(B if axe == "connexions" else np.mean([x["lecture"] for x in v]))
                            ys.append(np.mean([x["test"] for x in v]))
                    if xs and not (fam == "ngrc" and axe == "connexions"):
                        ax.plot(xs, ys, "-o", color=COL[fam], lw=1.8, ms=4, label=LAB[fam])
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_title(t, fontsize=10.5, color=INK, loc="left")
                ax.set_xlabel("connexions récurrentes" if axe == "connexions" else "paramètres de lecture",
                              fontsize=9, color=INK2)
            axs.flat[-1].axis("off")
            h, l = axs.flat[0].get_legend_handles_labels()
            axs.flat[-1].legend(h, l, fontsize=10, frameon=False, loc="center")
            fig.tight_layout()
            fig.savefig(RES / nom, facecolor=SURF)
    print("figures écrites")


# 14.1 : statistiques pour les tableaux du rapport — moyenne ± écart-type,
# intervalle de confiance bootstrap du rapport apparié, Wilcoxon apparié
# (10 seeds rendent le test valide : plafond p = 0,002 à 10 paires) corrigé par
# Holm au sein de chaque tâche (4 comparaisons contre l'ESN à fuite)
def stats():
    rec = R.lire(FICHIER["recette"])
    out = R.lire(FICHIER["equite"])
    for tache in TACHES_EQUITE:
        ref = [out["cells"].get(f"{tache}|esn_fuite|s{s}", {}).get("test") for s in SEEDS]
        ref = [x for x in ref if x is not None]
        lignes, ps = [], []
        for fam in ("esn_fuite", "esn_entree", "dts", "tenseur_fuite", "tenseur_sans_fuite", "recette"):
            src = rec if fam == "recette" else out
            pat = (lambda s: f"{tache}|s{s}") if fam == "recette" else (lambda s: f"{tache}|{fam}|s{s}")
            v = [src["cells"].get(pat(s), {}).get("test") for s in SEEDS]
            v = [x for x in v if x is not None]
            if not v:
                continue
            lignes.append(f"{tache} | {fam} : {np.mean(v):.4f} ± {np.std(v):.4f} (n = {len(v)})")
            ps.append(None)
            if fam != "esn_fuite" and len(v) == len(ref) and len(v) >= 6:
                rap = np.asarray(v) / np.asarray(ref)
                lo, hi = R.P7.bootstrap_ci(rap)
                lignes[-1] += f" ; rapport apparié {np.mean(rap):.3f}, IC 95 % [{lo:.3f} ; {hi:.3f}]"
                ps[-1] = R.wilcoxon(v, ref)
        idx = [i for i, p in enumerate(ps) if p is not None]
        idx.sort(key=lambda i: ps[i])
        prec = 0.0
        for rang, i in enumerate(idx):
            holm = min(1.0, max(prec, (len(idx) - rang) * ps[i]))
            prec = holm
            lignes[i] += f" ; Wilcoxon p = {ps[i]:.4g}, Holm p = {holm:.4g}"
        for ligne in lignes:
            print(ligne, flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    {"equite": equite, "recette": recette, "budgets": budgets, "figures": figures,
     "stats": stats}.get(cmd, lambda: print(__doc__))()
