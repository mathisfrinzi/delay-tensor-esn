"""Nouvelles séries réelles : le réservoir à retards sur des données à retards nets (période 24, 168, 132).

Protocole (écrit avant tout calcul, voir rapport/preenregistrements/test1.md) :
  - prévision à horizon h : entrée u(t) = x(t), cible y(t) = x(t + h), série normalisée avec la
    moyenne et l'écart-type de l'entraînement seul ;
  - découpe chronologique : entraînement [wash ; tr_end), test [te0 ; te1) = la fin de la série.
    La validation (choix de ρ, fuite, échelle d'entrée, λ) est prise DANS l'entraînement
    (80 % / 20 % finaux), le test n'est jamais un critère ;
  - 1) `diagnostic` : spectre de retards sur l'entraînement seul, S calculé, prévision écrite
    dans results/prereg_reel.json (aucun réservoir). À pousser AVANT `run` ;
  - 2) `run` : refuse de partir sans prereg_reel.json. Familles : ESN à fuite, DTS-ESN, retards
    avec fuite (espace a priori de p12_synthese), ESN avec retards en entrée, recette ; 10 seeds
    de réservoir (la série est unique : les seeds ne changent que le tirage) ; 30 évaluations de
    réglage par famille et par seed, sur la validation. Références sans réservoir : persistance,
    ridge sur tous les retards, NG-RC ;
  - 3) `tableau` : tableaux et verdict par rapport aux prévisions.

    python src/reel_nouveau.py diagnostic
    python src/reel_nouveau.py run [tâche ...]
    python src/reel_nouveau.py tableau
Reprise automatique : results/results_reel_nouveau.json. ESSAI=1 : séries tronquées, jamais rapporter.
Données : REEL_DATA (défaut /tmp/datasets_reel), téléchargées au premier lancement.
"""
import hashlib
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reservoir as R          # noqa: E402
import conception as C         # noqa: E402
import p12_synthese as P13     # noqa: E402
import p7_common as P7         # noqa: E402

ESSAI = R.ESSAI
RES = R.RES / "essai" if ESSAI else R.RES
PREREG = RES / "prereg_reel.json"
FICHIER = RES / "results_reel_nouveau.json"
DATA = Path(os.environ.get("REEL_DATA", "/tmp/datasets_reel"))
PROTO = "reel_1"
SEEDS = [0, 1] if ESSAI else list(range(10))
N_EVAL = 6 if ESSAI else 30
BUDGET = 400_000
NEFF_SEUIL = 6.0     # prévision : S calculé gagne si l'effectif de retards N_eff = 1/Σ w² est ≤ 6
FAMILLES = ["esn_fuite", "dts", "tenseur_fuite", "esn_entree", "recette"]

BASE = "https://raw.githubusercontent.com/"
SOURCES = {
    "etth1": (BASE + "zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv", "ETTh1.csv"),
    "etth2": (BASE + "zhouhaoyi/ETDataset/main/ETT-small/ETTh2.csv", "ETTh2.csv"),
    "sunspots": (BASE + "jbrownlee/Datasets/master/monthly-sunspots.csv", "monthly-sunspots.csv"),
    "melbourne": (BASE + "jbrownlee/Datasets/master/daily-min-temperatures.csv", "daily-min-temperatures.csv"),
}
# tâche -> (source, colonne, horizon h, longueur du test, lmax du spectre, période connue)
TACHES = {
    "etth1_h24": ("etth1", "OT", 24, 2880, 200, 24),
    "etth2_h24": ("etth2", "OT", 24, 2880, 200, 24),
    "sunspots_h12": ("sunspots", "Sunspots", 12, 600, 150, 132),
    "melbourne_h7": ("melbourne", "Temp", 7, 730, 60, 365),
}


def telecharger(src):
    url, nom = SOURCES[src]
    DATA.mkdir(parents=True, exist_ok=True)
    p = DATA / nom
    if not p.exists():
        urllib.request.urlretrieve(url, p)
    return p


def lire_serie(src, col):
    import csv
    with open(telecharger(src), newline="", encoding="utf-8-sig") as f:
        lec = csv.DictReader(f)
        return np.array([float(r[col]) for r in lec], np.float64)


def charger_reel(tache):
    """Dict compatible avec reservoir.evaluer : U (T, 2), y, splits, type='seq'."""
    src, col, h, nte, lmax, _ = TACHES[tache]
    x = lire_serie(src, col)
    if ESSAI:
        x = x[:3000]
        nte = 500
    T = len(x) - h                      # cible x(t+h) : on perd les h derniers instants
    tr_end = T - nte
    mu, sd = x[:tr_end].mean(), x[:tr_end].std()      # normalisation : entraînement seul
    xn = (x - mu) / sd
    U = np.stack([xn[:T], np.ones(T)], 1).astype(np.float32)
    y = xn[h:h + T].astype(np.float32)
    wash = 400 if lmax >= 150 else 200
    return {"U": U, "y": y, "type": "seq",
            "splits": {"wash": wash, "tr_end": tr_end, "te0": tr_end, "te1": T}, "lmax": lmax}


def empreinte(d):
    return hashlib.sha256(d["U"].tobytes() + d["y"].tobytes()).hexdigest()[:16]


# diagnostic, sans réservoir
def diagnostic_tache(tache):
    d = charger_reel(tache)
    s = d["splits"]
    lmax = d["lmax"]
    u, y = d["U"][:s["tr_end"], 0], d["y"][:s["tr_end"]]       # entraînement seul
    spec = C.spectre(u, y, lmax=lmax, wash=s["wash"], produits=False)
    w = C.poids_retards(spec)
    if w is None:
        w = C.enveloppe(spec)
    neff = float(1.0 / np.sum(w ** 2))
    top = [int(i) for i in np.argsort(-w)[:6]]
    S = {f"K{K}": [int(t) for t in C.support_sauts(w, K, lmax, lmax)[0]] for K in (2, 3)}
    pred = "retards_devant" if neff <= NEFF_SEUIL else "pas_de_victoire_prevue"
    return {"N_eff": neff, "retards_importants": top, "S_calcule": S, "prevision": pred,
            "erreur_prédicteur_lags_val": spec.get("val_plein"), "serie": empreinte(d),
            "T": int(len(d["y"])), "tr_end": int(s["tr_end"])}


def diagnostic():
    RES.mkdir(parents=True, exist_ok=True)
    out = {"proto": PROTO, "date": time.strftime("%Y-%m-%d %H:%M:%S"), "neff_seuil": NEFF_SEUIL,
           "regle": "prévision « retards_devant » si N_eff ≤ seuil, sinon « pas_de_victoire_prevue »",
           "taches": {}}
    lignes = ["| tâche | N_eff | retards les plus importants | S calculé (K = 2) | S calculé (K = 3) | prévision |",
              "|---|---|---|---|---|---|"]
    for t in TACHES:
        r = diagnostic_tache(t)
        out["taches"][t] = r
        print(t, round(r["N_eff"], 2), r["retards_importants"], r["S_calcule"], r["prevision"], flush=True)
        lignes.append(f"| {t} | {r['N_eff']:.1f} | {r['retards_importants']} | {r['S_calcule']['K2']} | "
                      f"{r['S_calcule']['K3']} | {r['prevision']} |")
    PREREG.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print("\n".join(lignes))
    (RES / "prereg_reel_tableau.md").write_text("\n".join(lignes) + "\n", encoding="utf-8")
    print("prévisions écrites dans", PREREG)


# recette adaptée aux longs retards
def candidats_recette_reel(data, n_eval):
    s = data["splits"]
    lmax = data["lmax"]
    u, y = data["U"][:s["tr_end"], 0], data["y"][:s["tr_end"]]
    spec = C.spectre(u, y, lmax=lmax, wash=s["wash"], produits=False)
    w = C.poids_retards(spec)
    w = C.enveloppe(spec) if w is None else w
    SS = {f"sauts K={K}": [int(t) for t in C.support_sauts(w, K, lmax, lmax)[0]] for K in (2, 3, 5)}
    SS["uniforme {1..10}"] = list(range(1, 11))
    n_t = max(1, n_eval // len(SS))
    rng = np.random.default_rng(12_000)
    tirages = [{"rho": float(rng.uniform(0.1, 1.1)), "fuite": float(rng.uniform(0.05, 1.0)),
                "echelle": float(np.exp(rng.uniform(np.log(0.05), np.log(2.0))))} for _ in range(n_t)]
    return [dict(t, S=S, kind="perm", paire=None, origine=o) for o, S in SS.items() for t in tirages]


def regler_famille(data, famille, seed, n, S_ref=None):
    """Recherche aléatoire de n évaluations sur la validation ; renvoie la meilleure config."""
    rng = np.random.default_rng(10_000 + seed)
    d = data["U"].shape[1]
    best = None
    for _ in range(n):
        cfg = P13.tirer(famille, rng)
        v = R.evaluer(P13.construire(cfg, BUDGET, seed, d), data, val=True)
        if best is None or v < best[0]:
            best = (v, cfg)
    return best[1], best[0]


def lancer_tache(tache, out):
    data = charger_reel(tache)
    pre = json.loads(PREREG.read_text(encoding="utf-8"))["taches"][tache]
    S_ref = pre["S_calcule"]["K2"]
    data_ref = P13.augmenter(data, S_ref, None)
    d1 = data["U"].shape[1]
    t0 = time.time()
    for seed in SEEDS:
        for fam in FAMILLES:
            key = f"{tache}|{fam}|s{seed}"
            if out["cells"].get(key, {}).get("proto") == PROTO:
                continue
            if fam == "recette":
                cands = candidats_recette_reel(data, N_EVAL * 2)
                vals = [R.evaluer(P13.construire(c, BUDGET, seed, d1), data, val=True) for c in cands]
                i = int(np.argmin(vals))
                cfg, val = cands[i], vals[i]
                dt = data
            else:
                dt = data_ref if fam == "esn_entree" else data
                cfg, val = regler_famille(dt, fam, seed, N_EVAL)
            test = R.evaluer(P13.construire(cfg, BUDGET, seed, dt["U"].shape[1]), dt)
            out["cells"][key] = {"test": float(test), "val": float(val), "cfg": cfg, "proto": PROTO}
            R.ecrire(out, FICHIER)
            print(f"{key} test={test:.4f} val={val:.4f} ({time.time() - t0:.0f}s)", flush=True)
    # références sans réservoir (déterministes)
    s = data["splits"]
    lmax = data["lmax"]
    kb = f"{tache}|persistance"
    if kb not in out["cells"]:
        yt = data["y"][s["te0"]:s["te1"]]
        p = data["U"][s["te0"]:s["te1"], 0]
        out["cells"][kb] = {"test": float(np.sqrt(np.mean((p - yt) ** 2)) / np.std(yt)), "proto": PROTO}
    kb = f"{tache}|ridge_lags"
    if kb not in out["cells"]:
        u = data["U"][:, 0].astype(np.float64)
        T = len(u)
        F = np.zeros((T, lmax + 1), np.float32)
        for k in range(lmax + 1):
            F[k:, k] = u[:T - k]
        te, lam, va = P7.ridge_readout(F, data["y"], s["wash"], s["tr_end"], s["te0"], s["te1"])
        out["cells"][kb] = {"test": float(te), "val": float(va), "lam": float(lam), "lmax": lmax, "proto": PROTO}
    kb = f"{tache}|ngrc"
    if kb not in out["cells"]:
        best = None
        for D in (5, 10, 20, 30, 40):
            for prod in (False, True):
                v = R.ngrc_evaluer(data, D, prod, val=True)
                if best is None or v < best[0]:
                    best = (v, D, prod)
        te = R.ngrc_evaluer(data, best[1], best[2])
        out["cells"][kb] = {"test": float(te), "val": float(best[0]), "D": best[1], "produits": best[2],
                            "proto": PROTO}
    R.ecrire(out, FICHIER)
    print(tache, "références sans réservoir écrites", flush=True)


def run(taches):
    if not PREREG.exists():
        raise SystemExit("Refusé : lancer et pousser d'abord `diagnostic` (prévisions écrites avant le test).")
    out = R.lire(FICHIER)
    for t in (taches or list(TACHES)):
        lancer_tache(t, out)
    out["meta"] = {"proto": PROTO, "seeds": SEEDS, "n_eval": N_EVAL, "budget": BUDGET, "smoke": ESSAI,
                   "familles": FAMILLES, "erreur": "NRMSE de test (écart-type du test)"}
    R.ecrire(out, FICHIER)


def tableau():
    out = R.lire(FICHIER)
    pre = json.loads(PREREG.read_text(encoding="utf-8"))["taches"]
    gagne = total = 0
    for t in TACHES:
        if f"{t}|esn_fuite|s{SEEDS[-1]}" not in out["cells"]:
            print(t, ": pas encore calculé")
            continue
        get = lambda fam: np.array([out["cells"][f"{t}|{fam}|s{s}"]["test"] for s in SEEDS])
        ref = get("esn_fuite")
        print(f"== {t} | prévision : {pre[t]['prevision']} (N_eff {pre[t]['N_eff']:.1f}, S K=2 {pre[t]['S_calcule']['K2']})")
        for fam in FAMILLES:
            v = get(fam)
            p = R.wilcoxon(v, ref) if fam != "esn_fuite" else None
            print(f"  {fam:14s} NRMSE {v.mean():.4g} ± {v.std():.2g} (médiane {np.median(v):.4g})"
                  + (f" ; rapport à l'ESN à fuite {v.mean() / ref.mean():.3f}, Wilcoxon p = {p:.4g}" if p is not None else ""))
        for b in ("persistance", "ridge_lags", "ngrc"):
            c = out["cells"].get(f"{t}|{b}")
            if c:
                print(f"  {b:14s} NRMSE {c['test']:.4g} (sans réservoir)")
        rec = get("recette")
        p = R.wilcoxon(rec, ref)
        verdict = "gagne" if (rec.mean() < ref.mean() and p < 0.05) else "perd" if (rec.mean() > ref.mean() and p < 0.05) else "égalité"
        prevu = pre[t]["prevision"] == "retards_devant"
        accord = (verdict == "gagne") == prevu
        total += 1
        gagne += int(accord)
        print(f"  VERDICT recette contre ESN à fuite : {verdict} ; prévision {'tenue' if accord else 'NON tenue'}")
    if total:
        print(f"PRÉVISIONS TENUES : {gagne} sur {total}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "diagnostic":
        diagnostic()
    elif cmd == "run":
        run(sys.argv[2:])
    elif cmd == "tableau":
        tableau()
    else:
        print(__doc__)
