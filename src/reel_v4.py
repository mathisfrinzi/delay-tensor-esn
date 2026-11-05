"""Test pré-enregistré 4 : six séries jamais ouvertes ; recette contre ESN à fuite (règle des tests 2 et 3,
inchangée) et recette contre Iacob et Dambre (2024). Protocole : protocole_test4.md.

    python src/reel_v4.py diagnostic        # sans réservoir, écrit results/prereg_v4.json (à pousser avant run)
    python src/reel_v4.py run [tâche ...]   # refuse sans prereg_v4.json ; IACOB_PROCS processus
    python src/reel_v4.py tableau
ESSAI=1 : début de série seulement (ne recouvre jamais le test réel), jamais rapporter.
Données : REEL_DATA (défaut /tmp/datasets_reel), téléchargées depuis GitHub au premier lancement.
"""
import csv
import datetime as dt
import gzip
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iacob_dambre as I      # noqa: E402  (installe le simulateur à délais de propagation)
import references2 as F2      # noqa: E402
import reel_nouveau as RN     # noqa: E402
import reel_corrige as RC     # noqa: E402
import conception as C        # noqa: E402
import p12_synthese as P13    # noqa: E402
import reservoir as R         # noqa: E402
import p7_common as P7        # noqa: E402

PROTO = "reel_v4"
PREREG = RN.RES / "prereg_v4.json"
FICHIER = RN.RES / "results_reel_v4.json"
SEEDS, N_EVAL, BUDGET = RN.SEEDS, RN.N_EVAL, RN.BUDGET
GAIN_SEUIL = 0.95          # règle des tests 2 et 3, inchangée
P_IACOB = 40               # recette devant Iacob-Dambre prévue si P* > 40 et rapport <= 0,95
NLEN, NTE, H, LMAX = 12000, 2880, 24, 200
G = "https://raw.githubusercontent.com/"
SRC = {"pekin": (G + "jbrownlee/Datasets/master/pollution.csv", "pekin_pollution.csv"),
       "taxi": (G + "numenta/NAB/master/data/realKnownCause/nyc_taxi.csv", "nyc_taxi.csv"),
       "pjm": (G + "panambY/Hourly_Energy_Consumption/master/data/PJME_hourly.csv", "PJME_hourly.csv"),
       "change": (G + "laiguokun/multivariate-time-series-data/master/exchange_rate/exchange_rate.txt.gz",
                  "exchange_rate.txt.gz")}
TACHES = {"pekin_pm25_h24": ("pekin", "pm2.5"), "pekin_T_h24": ("pekin", "TEMP"),
          "pekin_pression_h24": ("pekin", "PRES"), "taxi_h24": ("taxi", None),
          "pjm_charge_h24": ("pjm", None), "change_h24": ("change", None)}
FAM = ["esn_fuite", "recette", "iacob", "iacob_S", "hybride", "iacob_spectre",
       "self_esn", "lecture_retardee", "cycle", "orthogonal", "iacob_cma"]
NOMS = {"esn_fuite": "ESN à fuite", "recette": "recette", "iacob": "Iacob-Dambre", "iacob_S": "A (délais dans S)",
        "hybride": "B (mélange)", "iacob_spectre": "C (dmax = max S)", "self_esn": "self-ESN",
        "lecture_retardee": "lecture retardée", "cycle": "cycle (SCR)", "orthogonal": "orthogonal",
        "iacob_cma": "Iacob-Dambre + CMA-ES"}


def _telecharger(url, nom):
    RN.DATA.mkdir(parents=True, exist_ok=True)
    p = RN.DATA / nom
    if not p.exists():
        urllib.request.urlretrieve(url, p)
    return p


def _remplir(x):
    x = np.array(x, float)
    i0 = int(np.argmax(~np.isnan(x)))
    x = x[i0:]
    for i in range(1, len(x)):
        if np.isnan(x[i]):
            x[i] = x[i - 1]
    return x


def _grille_horaire(ts, v, pas=3600):
    """Moyenne par horodatage, grille régulière, trous comblés par la valeur précédente."""
    d = {}
    for t, x in zip(ts, v):
        d.setdefault(t, []).append(x)
    t0, t1 = min(d), max(d)
    n = int((t1 - t0).total_seconds() // pas) + 1
    out = np.full(n, np.nan)
    for t, xs in d.items():
        out[int((t - t0).total_seconds() // pas)] = np.mean(xs)
    return _remplir(out)


def serie_brute(tache):
    src, col = TACHES[tache]
    p = _telecharger(*SRC[src])
    if src == "pekin":
        with open(p, newline="", encoding="utf-8-sig") as f:
            x = [float(r[col]) if r[col] not in ("NA", "") else np.nan for r in csv.DictReader(f)]
        return _remplir(x)
    if src in ("taxi", "pjm"):
        with open(p, newline="", encoding="utf-8-sig") as f:
            rows = list(csv.reader(f))[1:]
        ts = [dt.datetime.fromisoformat(r[0]) for r in rows]
        v = [float(r[1]) for r in rows]
        if src == "taxi":                         # demi-heures : somme par heure
            ts = [t.replace(minute=0) for t in ts]
            g = _grille_horaire(ts, v)
            return g * 2.0                        # moyenne de deux demi-heures × 2 = somme
        return _grille_horaire(ts, v)
    return np.loadtxt(gzip.open(p), delimiter=",")[:, 0]          # change : première devise, journalier


def charger(tache):
    x = serie_brute(tache)[-NLEN - H:]
    nte = min(NTE, int(0.2 * (len(x) - H)))
    if RN.ESSAI:
        x, nte = x[:3000], 500
    T = len(x) - H
    tr_end = T - nte
    xn = (x - x[:tr_end].mean()) / x[:tr_end].std()
    U = np.stack([xn[:T], np.ones(T)], 1).astype(np.float32)
    return {"U": U, "y": xn[H:H + T].astype(np.float32), "type": "seq", "lmax": LMAX,
            "splits": {"wash": 400, "tr_end": tr_end, "te0": tr_end, "te1": T}}


def ensembles_S(d):
    """Identique au test 3 (reel_v3.ensembles_S)."""
    P, vP = RC.periode(d)
    U = d["U"][:d["splits"]["tr_end"], 0].astype(np.float64)
    s = d["splits"]
    T = len(U)
    v0 = s["wash"] + int(0.8 * (T - s["wash"]))
    Fm = np.zeros((T, 3), np.float32)
    for j in range(3):
        Fm[j:, j] = U[:T - j]
    v012 = P7.ridge_readout(Fm, d["y"][:T], s["wash"], v0, v0, T)[0]
    w = np.zeros(LMAX + 1)
    w[1] = w[2] = 0.25
    mult = list(range(P, LMAX + 1, P))
    for k in mult:
        w[k] = 0.5 / len(mult)
    SS = {f"sauts K={K}": [int(t) for t in C.support_sauts(w, K, LMAX, LMAX)[0]] for K in (2, 3, 5)}
    return P, vP, v012, SS


def diagnostic():
    RN.RES.mkdir(parents=True, exist_ok=True)
    out = {"proto": PROTO, "date": time.strftime("%Y-%m-%d %H:%M:%S"), "gain_seuil": GAIN_SEUIL, "p_iacob": P_IACOB,
           "regle_principale": "recette bat l'ESN à fuite (moyenne plus basse, Wilcoxon p<0.05) si val(P*)/val({0,1,2}) <= 0.95",
           "regle_iacob": "recette bat Iacob-Dambre (moyenne plus basse, p<0.05) si P* > 40 et rapport <= 0.95, sinon pas de victoire",
           "secondaire": "Iacob-Dambre + CMA-ES fait mieux qu'Iacob-Dambre à positions au hasard (p<0.05) sur au moins 4 séries sur 6",
           "taches": {}}
    for t in TACHES:
        d = charger(t)
        P, vP, v012, SS = ensembles_S(d)
        ratio = vP / v012
        out["taches"][t] = {"P_etoile": P, "val_P": vP, "val_012": v012, "rapport": ratio, "S": SS,
                            "longueur": int(len(d["y"])), "test": int(d["splits"]["te1"] - d["splits"]["te0"]),
                            "serie": RN.empreinte(d),
                            "prevision": "retards_devant" if ratio <= GAIN_SEUIL else "pas_de_victoire_prevue",
                            "prevision_iacob": "recette_devant" if (P > P_IACOB and ratio <= GAIN_SEUIL) else "pas_de_victoire_prevue"}
        r = out["taches"][t]
        print(f"{t:20s} P* {P:3d} rapport {ratio:.3f} S(K=3) {SS['sauts K=3']} → {r['prevision']} ; "
              f"contre Iacob : {r['prevision_iacob']}", flush=True)
    PREREG.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")


# calcul : une cellule = (série, famille, seed)
def _cands(SS):
    rng = np.random.default_rng(12_000)
    tir = [{"rho": float(rng.uniform(0.1, 1.1)), "fuite": float(rng.uniform(0.05, 1.0)),
            "echelle": float(np.exp(rng.uniform(np.log(0.05), np.log(2.0))))} for _ in range(max(1, N_EVAL // 3))]
    return [dict(t, S=S, kind="perm", paire=None, origine=o) for o, S in SS.items() for t in tir]


def cellule(job):
    tache, fam, seed = job
    t0 = time.time()
    d = charger(tache)
    pre = json.loads(PREREG.read_text(encoding="utf-8"))["taches"][tache]
    if fam == "esn_fuite":
        cfg, val = RN.regler_famille(d, fam, seed, N_EVAL)
        te = R.evaluer(P13.construire(cfg, BUDGET, seed, 2), d)
    elif fam == "recette":
        cands = _cands(pre["S"])
        vals = [R.evaluer(P13.construire(c, BUDGET, seed, 2), d, val=True) for c in cands]
        i = int(np.argmin(vals))
        cfg, val = cands[i], vals[i]
        te = R.evaluer(P13.construire(cfg, BUDGET, seed, 2), d)
    else:
        rec = R.lire(FICHIER)["cells"][f"{tache}|recette|s{seed}"]["cfg"]
        S, paire = rec["S"], rec.get("paire")
        if fam in I.FAMILLES:
            rng = np.random.default_rng(10_000 + seed)
            best = None
            for _ in range(N_EVAL):
                c = I.tirer(fam, rng)
                v = R.evaluer(I.construire(fam, c, seed, 2, S, paire), d, val=True)
                if best is None or v < best[0]:
                    best = (v, c)
            val, cfg = best
            te = R.evaluer(I.construire(fam, cfg, seed, 2, S, paire), d)
        else:
            te, val, cfg = F2.regler(fam, d, seed, 2, S, N_EVAL)
    return f"{tache}|{fam}|s{seed}", {"test": float(te), "val": float(val), "cfg": cfg, "proto": PROTO}, time.time() - t0


def _pool(jobs, out):
    from multiprocessing import Pool
    procs = int(os.environ.get("IACOB_PROCS", max(1, min(4, (os.cpu_count() or 2) - 1))))
    print(f"{len(jobs)} cellules à calculer, {procs} processus, début {time.strftime('%H:%M')}", flush=True)
    t0 = time.time()
    with Pool(procs) as pool:
        for k, (key, cell, dtm) in enumerate(pool.imap_unordered(cellule, jobs), 1):
            out["cells"][key] = cell
            R.ecrire(out, FICHIER)
            el = time.time() - t0
            print(f"[{k}/{len(jobs)}] {key} test={cell['test']:.4f} ({dtm / 60:.1f} min) | "
                  f"{el / 60:.0f} min écoulées, reste ~{el / k * (len(jobs) - k) / 60:.0f} min", flush=True)


def references_sans_reservoir(tache, out):
    d = charger(tache)
    P = json.loads(PREREG.read_text(encoding="utf-8"))["taches"][tache]["P_etoile"]
    s, u = d["splits"], d["U"][:, 0].astype(np.float64)
    T = len(u)

    def ridge(lags):
        Fm = np.zeros((T, len(lags)), np.float32)
        for j, k in enumerate(lags):
            Fm[k:, j] = u[:T - k]
        te, lam, va = P7.ridge_readout(Fm, d["y"], s["wash"], s["tr_end"], s["te0"], s["te1"])
        return {"test": float(te), "val": float(va), "lam": float(lam), "proto": PROTO}
    yt = d["y"][s["te0"]:s["te1"]]
    out["cells"][f"{tache}|persistance"] = {"test": float(np.sqrt(np.mean((u[s['te0']:s['te1']] - yt) ** 2)) / yt.std()),
                                            "proto": PROTO}
    out["cells"][f"{tache}|ridge_lags"] = ridge(list(range(LMAX + 1)))
    out["cells"][f"{tache}|ridge_periode"] = ridge(sorted({0, 1, 2} | set(range(P, LMAX + 1, P))))
    R.ecrire(out, FICHIER)


def run(taches):
    if not PREREG.exists():
        raise SystemExit("Refusé : lancer et pousser d'abord `diagnostic`.")
    taches = taches or list(TACHES)
    pre = json.loads(PREREG.read_text(encoding="utf-8"))["taches"]
    if not RN.ESSAI:
        for t in taches:
            if RN.empreinte(charger(t)) != pre[t]["serie"]:
                raise SystemExit(f"Refusé : {t} ne correspond pas à la série du diagnostic (empreinte).")
    out = R.lire(FICHIER)
    fait = lambda t, f, s: out["cells"].get(f"{t}|{f}|s{s}", {}).get("proto") == PROTO
    phase1 = [(t, f, s) for t in taches for s in SEEDS for f in ("esn_fuite", "recette") if not fait(t, f, s)]
    _pool(phase1, out)
    phase2 = [(t, f, s) for t in taches for s in SEEDS for f in FAM[2:] if not fait(t, f, s)]
    phase2.sort(key=lambda j: j[1] == "iacob_cma")
    _pool(phase2, out)
    for t in taches:
        references_sans_reservoir(t, out)
    out["meta"] = {"proto": PROTO, "seeds": SEEDS, "n_eval": N_EVAL, "budget": BUDGET, "smoke": RN.ESSAI,
                   "cma": {"lambda": F2.LAMBDA, "generations": F2.GENERATIONS}}
    R.ecrire(out, FICHIER)


def _verdict(a, b):
    p = R.wilcoxon(a, b)
    v = "gagne" if a.mean() < b.mean() and p < 0.05 else "perd" if a.mean() > b.mean() and p < 0.05 else "égalité"
    return v, p


def tableau():
    out = R.lire(FICHIER)["cells"]
    pre = json.loads(PREREG.read_text(encoding="utf-8"))["taches"]
    n1 = n2 = n3 = tot = 0
    for t in TACHES:
        if any(f"{t}|{f}|s{s}" not in out for f in FAM for s in SEEDS):
            print(t, ": pas encore calculé\n")
            continue
        g = {f: np.array([out[f"{t}|{f}|s{s}"]["test"] for s in SEEDS]) for f in FAM}
        esn = g["esn_fuite"]
        pr = pre[t]
        print(f"== {t} | P* {pr['P_etoile']} | rapport {pr['rapport']:.3f} | prévisions : {pr['prevision']}, "
              f"contre Iacob {pr['prevision_iacob']}")
        for f in FAM:
            v = g[f]
            x = "" if f == "esn_fuite" else f"  ×{v.mean() / esn.mean():.3f}  p = {R.wilcoxon(v, esn):.3g}"
            print(f"  {NOMS[f]:24s} {v.mean():.4f} ± {v.std():.4f}{x}")
        for b in ("persistance", "ridge_lags", "ridge_periode"):
            print(f"  {b:24s} {out[f'{t}|{b}']['test']:.4f} (sans réservoir)")
        v1, p1 = _verdict(g["recette"], esn)
        ok1 = (v1 == "gagne") == (pr["prevision"] == "retards_devant")
        v2, p2 = _verdict(g["recette"], g["iacob"])
        ok2 = (v2 == "gagne") == (pr["prevision_iacob"] == "recette_devant")
        v3, p3 = _verdict(g["iacob_cma"], g["iacob"])
        print(f"  RÈGLE 1, recette contre ESN à fuite : {v1} (p = {p1:.3g}) ; prévision {'tenue' if ok1 else 'NON tenue'}")
        print(f"  RÈGLE 2, recette contre Iacob-Dambre : {v2} (×{g['recette'].mean() / g['iacob'].mean():.3f}, "
              f"p = {p2:.3g}) ; prévision {'tenue' if ok2 else 'NON tenue'}")
        print(f"  secondaire, CMA-ES contre positions au hasard : {v3} (×{g['iacob_cma'].mean() / g['iacob'].mean():.3f}, p = {p3:.3g})\n")
        tot += 1
        n1 += ok1
        n2 += ok2
        n3 += v3 == "gagne"
    print(f"RÈGLE 1 TENUE : {n1} sur {tot} ; RÈGLE 2 TENUE : {n2} sur {tot} ; "
          f"secondaire (CMA-ES devant sur ≥ 4) : {n3} sur {tot}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    {"diagnostic": diagnostic, "tableau": tableau}.get(cmd, lambda: None)() if cmd != "run" else run(sys.argv[2:])
    if cmd not in ("diagnostic", "run", "tableau"):
        print(__doc__)
