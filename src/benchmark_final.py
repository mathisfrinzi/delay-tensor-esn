"""Test final (partie 14) : le test jamais ouvert.

Tous les résultats du projet prennent comme test les 10 000 derniers points des
30 000 de Silverbox et Wiener-Hammerstein, ou les 1 000 derniers pas de NARMA —
des blocs qui ont guidé chaque choix (ensembles élus, recette, critères de
conception). Pour l'article, ce script garde un test final qui n'a jamais été
regardé :

  - Silverbox, Wiener-Hammerstein : le protocole standard de
    nonlinearbenchmark.org — les données d'estimation servent à tout
    apprentissage, le test officiel est ouvert une seule fois, et la métrique
    standard (RMSE en mV : données brutes en volts × 1 000) est rapportée avec
    la NRMSE ;

ATTENTION, Silverbox : tout le projet a utilisé les 30 000 premiers points de
SNLS80mV.mat (p7_common.load_phys), qui sont DANS le test officiel « arrow »
(échantillons 100 à 40 575 dans nonlinear_benchmarks). Les deux tests arrow ont
donc déjà servi au réglage et au test de toutes les parties : ils sont calculés
pour la comparaison avec la littérature, sous les clés « …_deja_vu », mais ne
sont PAS un test jamais ouvert. Seul le test multisine (derniers 25 % du bloc
40 650 à 127 400) l'est. Wiener-Hammerstein n'a pas ce problème : les 30 000
premiers points sont avant le test officiel (échantillons 105 200 à 184 000).
  - NARMA-30 : de nouvelles réalisations (seeds 300 à 309, jamais utilisées
    nulle part ailleurs), ouvertes une seule fois.

Les modèles sont GELÉS : rien n'est réglé ici. Les configurations viennent de
results_p12_equite.json, results_p12_recette.json et results_p12_budgets.json
(partie 13, protocole 14.1) : une cellule sans le marqueur de protocole courant
est refusée — relancer p12_synthese d'abord. Seule la lecture ridge est
réapprise sur les données d'estimation (λ sur les 20 % finaux du bloc, puis
refit sur tout le bloc, comme partout dans le projet), jamais sur le test.

Une cellule déjà présente dans le JSON n'est jamais recalculée : le test final
ne se ré-ouvre pas. Ne jamais supprimer results_final_test.json à la main.

    python src/benchmark_final.py ouvrir [narma30|silverbox|wh]
    python src/benchmark_final.py tableau
    python src/benchmark_final.py critere   (verdict du critère écrit dans rapport/critere_test_final.md)

Les données officielles viennent de la bibliothèque nonlinear_benchmarks
(pip install nonlinear-benchmarks), qui télécharge les enregistrements
d'estimation et de test de nonlinearbenchmark.org. WH : test de 78 800 points,
fenêtre d'initialisation de 50 points exclue de l'erreur ; Silverbox : test
multisine (jamais ouvert) ; arrow complet et sans extrapolation (déjà vus, voir
plus haut), 50 points d'init chacun.
Avec ESSAI=1, mécanique vérifiée sur des troncatures (jamais rapporter).
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reservoir as R        # noqa: E402
import p7_common as P7       # noqa: E402
import p12_synthese as P13   # noqa: E402  (synthèse = partie 13 du rapport)

ESSAI = R.ESSAI
OUT = R.RES / "essai" if ESSAI else R.RES
FICHIER = OUT / "results_final_test.json"
SEEDS = P13.SEEDS
DECALAGE = 300              # nouvelles réalisations NARMA-30 : seed 300+s, jamais utilisées ailleurs
TACHES = ["narma30", "silverbox", "wh"]
PROTO = "final_1"


# données officielles
def charger_benchmark(tache):
    """Estimation et tests officiels via nonlinear_benchmarks. Renvoie
    {'u', 'y', 'tests': {nom: (u, y)}, 'init': points d'initialisation}."""
    try:
        import nonlinear_benchmarks as nb
    except ImportError as e:
        raise RuntimeError("pip install nonlinear-benchmarks : données officielles de "
                           "nonlinearbenchmark.org") from e
    if tache == "wh":
        tr, te = nb.WienerHammerBenchMark()
        tests = {"wh": (te.u, te.y)}
        init = int(getattr(te, "state_initialization_window_length", 50))
    else:
        tr, tes = nb.Silverbox()
        tes = list(tes)
        # arrow : déjà vu par tout le projet (30 000 premiers points de SNLS80mV.mat)
        tests = {"multisine": (tes[0].u, tes[0].y), "arrow_deja_vu": (tes[1].u, tes[1].y),
                 "arrow_sans_extrap_deja_vu": (tes[2].u, tes[2].y)}
        init = int(getattr(tes[0], "state_initialization_window_length", 50))
    u = np.asarray(tr.u).ravel().astype(np.float64)
    y = np.asarray(tr.y).ravel().astype(np.float64)
    tests = {k: (np.asarray(a).ravel().astype(np.float64),
                 np.asarray(b).ravel().astype(np.float64)) for k, (a, b) in tests.items()}
    if ESSAI:
        u, y = u[:3000], y[:3000]
        tests = {k: (a[:1000], b[:1000]) for k, (a, b) in tests.items()}
    return {"u": u, "y": y, "tests": tests, "init": init}


# lecture ridge commune (mêmes splits que le projet)
def biais(F):
    return np.hstack([F, np.ones((len(F), 1), F.dtype)])


def fit_lecture(F, y, wash, tr_end):
    """λ sur les 20 % finaux de [wash:tr_end], puis refit sur [wash:tr_end]."""
    Fb = biais(F).astype(np.float64)
    yy = np.asarray(y, np.float64)
    v0 = wash + int(0.8 * (tr_end - wash))
    I = np.eye(Fb.shape[1])
    errs = []
    for lam in R.RIDGES:
        W = np.linalg.solve(Fb[wash:v0].T @ Fb[wash:v0] + lam * I, Fb[wash:v0].T @ yy[wash:v0])
        r = Fb[v0:tr_end] @ W - yy[v0:tr_end]
        errs.append(np.sqrt((r ** 2).mean()))
    lam = R.RIDGES[int(np.argmin(errs))]
    W = np.linalg.solve(Fb[wash:tr_end].T @ Fb[wash:tr_end] + lam * I,
                        Fb[wash:tr_end].T @ yy[wash:tr_end])
    return W, float(lam)


def evaluer_test(F, y_norm, debut, W, dy, y_brut):
    """Prédiction sur les états du test à partir de 'debut' (fenêtre
    d'initialisation) ; renvoie (RMSE dans l'unité de y_brut × dy/std, NRMSE,
    nombre de points). Pour Silverbox et WH, dy est l'écart-type en volts : la
    conversion en mV se fait à l'appel."""
    pred = biais(F).astype(np.float64) @ W
    r = pred[debut:] - np.asarray(y_norm, np.float64)[debut:]
    rmse = float(np.sqrt((r ** 2).mean()) * dy)
    nrmse = float(rmse / np.std(np.asarray(y_brut, np.float64)[debut:]))
    return rmse, nrmse, len(pred) - debut


def tmax_de(res):
    lags = list(res["taus"]) + ([1] if np.ndim(res["fuite"]) or res["fuite"] != 1.0 else [])
    if res["paire"]:
        lags += list(res["paire"])
    return max(lags)


def ngrc_features(u_norm, D, produits):
    T = len(u_norm)
    lag = np.zeros((T, D))
    for k in range(D):
        lag[k:, k] = u_norm[:T - k]
    feats = [lag]
    if produits:
        iu = np.triu_indices(D)
        feats.append(lag[:, iu[0]] * lag[:, iu[1]])
    return np.hstack(feats).astype(np.float32)


def entree(cfg, U):
    """Entrée du modèle : U telle quelle, ou augmentée des retards et du produit de
    la référence d'entrée (famille esn_entree, même construction que p12_synthese)."""
    ref = cfg.get("_entree")
    if not ref:
        return U
    return P13.augmenter({"U": U}, ref["S"], ref["paire"])["U"]


# modèles gelés
def cfg_de(src, key, fichier):
    c = src["cells"].get(key)
    if not isinstance(c, dict) or c.get("proto") != P13.PROTO:
        raise RuntimeError(f"{fichier} : {key} absent ou protocole antérieur à 14.1 — "
                           "relancer p12_synthese avant d'ouvrir le test final")
    return c


def modeles(tache, equite, recette, budgets):
    """Liste (fam, seed_poids, cfg) des modèles gelés de la partie 13 (14.1)."""
    mds = []
    for fam in ("esn_fuite", "dts", "tenseur_fuite", "tenseur_sans_fuite"):
        for s in SEEDS:
            mds.append((fam, s, cfg_de(equite, f"{tache}|{fam}|s{s}", "equite")["cfg"]))
    for s in SEEDS:
        mds.append(("recette", s, cfg_de(recette, f"{tache}|s{s}", "recette")["cfg"]))
    # référence équitable (14.1/14.3) : ESN à fuite qui reçoit les retards calculés et
    # le produit en entrée ; la référence (S, paire) est celle enregistrée par equite()
    ref = cfg_de(equite, f"{tache}|esn_entree|ref", "equite")
    for s in SEEDS:
        c = dict(cfg_de(equite, f"{tache}|esn_entree|s{s}", "equite")["cfg"])
        c["_entree"] = {"S": ref["S"], "paire": ref["paire"]}
        mds.append(("esn_entree", s, c))
    m = cfg_de(budgets, f"{tache}|B400000|meta", "budgets")
    for s in SEEDS:
        mds.append(("uniforme_10", s, m["uniforme_10"]))
    mds.append(("ngrc", None, m["ngrc"]))
    return mds


def verifier_horodatage():
    """Refuse d'ouvrir le test final tant que le critère n'est pas publié (gist public) et
    que son empreinte SHA-256 ne correspond pas à celle enregistrée."""
    import hashlib
    rap = Path(__file__).resolve().parent.parent / "rapport"
    if not (rap / "horodatage_test_final.md").exists() or not (rap / "critere_test_final.sha256").exists():
        raise SystemExit("Test final refusé : pas de critère publié. Écrire rapport/critere_test_final.md, "
                         "enregistrer son empreinte (critere_test_final.sha256) et son gist (horodatage_test_final.md) d'abord.")
    texte = (rap / "horodatage_test_final.md").read_text(encoding="utf-8")
    if "https://gist.github.com/" not in texte:
        raise SystemExit("Test final refusé : publier d'abord critere_test_final.md dans un gist "
                         "public et noter son adresse dans rapport/horodatage_test_final.md")
    attendu = (rap / "critere_test_final.sha256").read_text(encoding="utf-8").split()[0].lower()
    # fins de ligne normalisees (Windows : CRLF, Linux/GitHub : LF) : l'empreinte porte sur le texte
    reel = hashlib.sha256((rap / "critere_test_final.md").read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    if attendu != reel:
        raise SystemExit("Test final refusé : critere_test_final.md a changé depuis l'empreinte publiée")


def ouvrir(taches):
    if not ESSAI:
        verifier_horodatage()
    out = R.lire(FICHIER)
    equite = R.lire(P13.FICHIER["equite"])
    recette = R.lire(P13.FICHIER["recette"])
    budgets = R.lire(P13.FICHIER["budgets"])
    for tache in taches:
        mds = modeles(tache, equite, recette, budgets)
        if tache == "narma30":
            for fam, s, cfg in mds:
                keys = [f"narma30|realisation|{fam}" + (f"|s{s}" if s is not None else "")]
                if keys[0] in out["cells"]:
                    continue                                   # le test ne se ré-ouvre pas
                seed_new = DECALAGE + (s if s is not None else 0)
                U_new, y_new, _ = P7.narma_x(30, seed_new)
                if fam == "ngrc":
                    # lecture réapprise sur la réalisation d'estimation (seed 0)
                    d0 = R.charger("narma30", 0)
                    sp = d0["splits"]
                    F_est = ngrc_features(d0["U"][:, 0], cfg["D"], cfg["produits"])
                    W, lam = fit_lecture(F_est, d0["y"], sp["wash"], sp["tr_end"])
                    F_new = ngrc_features(U_new[:, 0], cfg["D"], cfg["produits"])
                    tmax = cfg["D"]
                else:
                    d0 = R.charger("narma30", s)
                    sp = d0["splits"]
                    U_e, U_n = entree(cfg, d0["U"]), entree(cfg, U_new)
                    res = P13.construire(cfg, P13.BUDGET, s, U_e.shape[1])
                    F_est = R.simuler(res, U_e)
                    W, lam = fit_lecture(F_est, d0["y"], sp["wash"], sp["tr_end"])
                    F_new = R.simuler(res, U_n)
                    tmax = max([tmax_de(res)] + list(cfg.get("_entree", {}).get("S", [])))
                wash_new = max(200, 3 * tmax)
                rmse, nrmse, n = evaluer_test(F_new, y_new, wash_new, W, 1.0, y_new)
                out["cells"][keys[0]] = {"rmse": rmse, "nrmse": nrmse, "lam": lam,
                                          "points": n, "proto": PROTO}
                R.ecrire(out, FICHIER)
                print(keys[0], "RMSE", round(rmse, 5), "NRMSE", round(nrmse, 5), flush=True)
        else:
            d = charger_benchmark(tache)
            du, dy = float(np.std(d["u"])), float(np.std(d["y"]))
            U_est = np.stack([d["u"] / du, np.ones_like(d["u"])], 1).astype(np.float32)
            y_est = (d["y"] / dy).astype(np.float32)
            T_est = len(U_est)
            for fam, s, cfg in mds:
                for nom_test, (u_te, y_te) in d["tests"].items():
                    key = f"{tache}|{nom_test}|{fam}" + (f"|s{s}" if s is not None else "")
                    if key in out["cells"]:
                        continue
                    U_te = np.stack([u_te / du, np.ones_like(u_te)], 1).astype(np.float32)
                    if fam == "ngrc":
                        F_est = ngrc_features(U_est[:, 0], cfg["D"], cfg["produits"])
                        wash = 500
                        W, lam = fit_lecture(F_est, y_est, wash, T_est)
                        F_te = ngrc_features(U_te[:, 0], cfg["D"], cfg["produits"])
                    else:
                        U_e, U_t = entree(cfg, U_est), entree(cfg, U_te)
                        res = P13.construire(cfg, P13.BUDGET, s, U_e.shape[1])
                        wash = max(500, 3 * tmax_de(res))
                        F_est = R.simuler(res, U_e)
                        W, lam = fit_lecture(F_est, y_est, wash, T_est)
                        F_te = R.simuler(res, U_t)
                    debut = d["init"]   # convention officielle : fenêtre d'initialisation exclue
                    rmse, nrmse, n = evaluer_test(F_te, y_te / dy, debut, W, dy, y_te)
                    rmse *= 1000.0     # données du benchmark en volts → mV
                    out["cells"][key] = {"rmse_mv": rmse, "nrmse": nrmse, "lam": lam,
                                         "points": n, "proto": PROTO,
                                         "deja_vu": nom_test.endswith("_deja_vu")}
                    R.ecrire(out, FICHIER)
                    print(key, "RMSE", round(rmse, 4), "mV ; NRMSE", round(nrmse, 5), flush=True)
    out["meta"] = {"proto": PROTO, "date_ouverture": time.strftime("%Y-%m-%d"),
                   "modeles": "gelés, partie 13 protocole 14.1 (aucun réglage ici)",
                   "unites": "RMSE en mV : données brutes du benchmark (volts) × 1 000",
                   "deja_vu": "Silverbox arrow (complet et sans extrapolation) : contenu dans les 30 000 "
                              "premiers points utilisés par tout le projet — PAS un test jamais ouvert",
                   "narma30": f"nouvelles réalisations, seeds {DECALAGE} à {DECALAGE + max(SEEDS)}",
                   "regle": "une cellule ouverte n'est jamais recalculée",
                   "smoke": ESSAI}
    R.ecrire(out, FICHIER)


def tableau():
    """Moyenne ± écart-type par (tâche, test, famille) — ne touche jamais au test."""
    out = R.lire(FICHIER)
    groupes = {}
    for k, c in out["cells"].items():
        p = k.split("|")
        fam = p[2] if len(p) > 3 else p[-1]
        groupes.setdefault((p[0], p[1], fam), []).append(c)
    for (t, te, fam), cs in sorted(groupes.items()):
        r = [c.get("rmse_mv", c.get("rmse")) for c in cs]
        nz = [c["nrmse"] for c in cs]
        champ = "mV" if t != "narma30" else "unités natives"
        marque = "  [DÉJÀ VU : indicatif, pas un test jamais ouvert]" if te.endswith("_deja_vu") else ""
        print(f"{t} | {te} | {fam} : RMSE {np.mean(r):.4g} ± {np.std(r):.4g} {champ} "
              f"(n = {len(cs)}) ; NRMSE {np.mean(nz):.4g} ± {np.std(nz):.4g}{marque}", flush=True)


# critère principal, écrit avant l'ouverture : rapport/critere_test_final.md
JAMAIS_VU = {"narma30": "realisation", "silverbox": "multisine", "wh": "wh"}


def critere():
    """Verdict du critère principal : réservoir à retards avec fuite (tenseur_fuite)
    contre ESN à fuite (esn_fuite), NRMSE, 10 seeds appariées, Wilcoxon bilatéral.
    Gagne : moyenne plus basse et p < 0,05 ; perd : moyenne plus haute et p < 0,05 ;
    sinon égalité. Confirmé si gagne sur NARMA-30 et Silverbox et ne perd pas sur WH."""
    out = R.lire(FICHIER)
    verdicts = {}
    for t, te in JAMAIS_VU.items():
        a = [out["cells"].get(f"{t}|{te}|tenseur_fuite|s{s}", {}).get("nrmse") for s in SEEDS]
        b = [out["cells"].get(f"{t}|{te}|esn_fuite|s{s}", {}).get("nrmse") for s in SEEDS]
        if any(x is None for x in a + b):
            print(f"{t} : pas encore ouvert")
            continue
        p = R.wilcoxon(a, b)
        ma, mb = float(np.mean(a)), float(np.mean(b))
        v = "gagne" if (ma < mb and p < 0.05) else "perd" if (ma > mb and p < 0.05) else "égalité"
        verdicts[t] = v
        print(f"{t} | {te} : retards avec fuite {ma:.4g}, ESN à fuite {mb:.4g}, "
              f"rapport {ma / mb:.3f}, Wilcoxon p = {p:.4g} -> {v}", flush=True)
    if len(verdicts) == 3:
        ok = verdicts["narma30"] == "gagne" and verdicts["silverbox"] == "gagne" and verdicts["wh"] != "perd"
        print("CRITÈRE PRINCIPAL :", "CONFIRMÉ" if ok else "NON CONFIRMÉ", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    arg = sys.argv[2] if len(sys.argv) > 2 else None
    {"ouvrir": lambda: ouvrir([arg] if arg else TACHES), "critere": critere,
     "tableau": tableau}.get(cmd, lambda: print(__doc__))()
