"""Test pré-enregistré 3 : diagnostic par périodes (règle du test 2, inchangée), cinq nouvelles séries.
Séries : moyenne sur tous les clients / capteurs (laiguokun/multivariate-time-series-data), les 12000 derniers pas
horaires ; horizon 24 h ; test = les 2880 derniers pas. Voir rapport/preenregistrements/test2.md.
    python src/reel_v3.py diagnostic | run [tâche ...] | tableau
ESSAI=1 : séries tronquées, jamais rapporter. Données : REEL_DATA (défaut /tmp/datasets_reel).
"""
import gzip, json, sys, time, urllib.request
import numpy as np
import reel_nouveau as RN
import reel_corrige as RC
import conception as C, p12_synthese as P13, reservoir as R, p7_common as P7

PROTO = "reel_v3"
PREREG = RN.RES / "prereg_v3.json"
FICHIER = RN.RES / "results_reel_v3.json"
SEEDS, N_EVAL, BUDGET = RN.SEEDS, RN.N_EVAL, RN.BUDGET
GAIN_SEUIL = 0.95       # prévision « retards_devant » si val(P*) / val({0,1,2}) <= 0.95
NLEN, NTE, H, LMAX = 12000, 2880, 24, 200
G = "https://raw.githubusercontent.com/"
SRC = {"solaire": (G + "laiguokun/multivariate-time-series-data/master/solar-energy/solar_AL.txt.gz", "solar_AL.txt.gz"),
       "energie": (G + "unit8co/darts/master/datasets/energy_dataset.csv", "energy_dataset.csv"),
       "meteo": (G + "unit8co/darts/master/datasets/weather.csv", "weather_jena.csv")}
TACHES = {"energie_charge_h24": ("energie", "total load actual"), "energie_prix_h24": ("energie", "price actual"),
          "solaire_h24": ("solaire", None), "meteo_T_h24": ("meteo", "T (degC)"),
          "meteo_rayonnement_h24": ("meteo", "SWDR (W/m\u00b2)")}
FAM = ["esn_fuite", "dts", "tenseur_fuite", "esn_entree", "recette"]


def _csv_col(path, col):
    import csv
    with open(path, newline="", encoding="utf-8-sig") as f:
        v = [r[col] for r in csv.DictReader(f)]
    x = np.array([float(a) if a not in ("", "nan", "NaN") else np.nan for a in v])
    for i in range(1, len(x)):          # remplissage avant des rares valeurs manquantes
        if np.isnan(x[i]):
            x[i] = x[i - 1]
    return x


def _horaire(x):
    n = len(x) // 6 * 6
    return x[:n].reshape(-1, 6).mean(1)


def _telecharger(url, nom):
    RN.DATA.mkdir(parents=True, exist_ok=True)
    p = RN.DATA / nom
    if not p.exists():
        urllib.request.urlretrieve(url, p)
    return p


def charger_v2(tache):
    src, col = TACHES[tache]
    url, nom = SRC[src]
    p = _telecharger(url, nom)
    if src == "solaire":
        x = _horaire(np.loadtxt(gzip.open(p), delimiter=",").mean(1))
    elif src == "energie":
        x = _csv_col(p, col)
    else:
        x = _horaire(_csv_col(p, col))
    x = x[-NLEN - H:]
    nte = min(NTE, int(0.2 * (len(x) - H)))
    if RN.ESSAI:
        x, nte = x[:3000], 500          # début de série : ne recouvre jamais le test réel
    T = len(x) - H
    tr_end = T - nte
    xn = (x - x[:tr_end].mean()) / x[:tr_end].std()
    U = np.stack([xn[:T], np.ones(T)], 1).astype(np.float32)
    return {"U": U, "y": xn[H:H + T].astype(np.float32), "type": "seq", "lmax": LMAX,
            "splits": {"wash": 400, "tr_end": tr_end, "te0": tr_end, "te1": T}}


def ensembles_S(d):
    P, vP = RC.periode(d)
    U = d["U"][:d["splits"]["tr_end"], 0].astype(np.float64)
    s = d["splits"]
    T = len(U)
    v0 = s["wash"] + int(0.8 * (T - s["wash"]))
    F = np.zeros((T, 3), np.float32)
    for j in range(3):
        F[j:, j] = U[:T - j]
    v012 = P7.ridge_readout(F, d["y"][:T], s["wash"], v0, v0, T)[0]
    w = np.zeros(LMAX + 1)
    w[1] = w[2] = 0.25
    mult = list(range(P, LMAX + 1, P))
    for k in mult:
        w[k] = 0.5 / len(mult)
    SS = {f"sauts K={K}": [int(t) for t in C.support_sauts(w, K, LMAX, LMAX)[0]] for K in (2, 3, 5)}
    return P, vP, v012, SS


def diagnostic():
    RN.RES.mkdir(parents=True, exist_ok=True)
    out = {"proto": PROTO, "date": time.strftime("%Y-%m-%d %H:%M:%S"), "gain_seuil": GAIN_SEUIL,
           "regle": "retards_devant si val(P*)/val({0,1,2}) <= seuil, sinon pas_de_victoire_prevue",
           "predictions_secondaires": {"vs_dts": "la recette ne bat pas DTS-ESN (p<0.05)",
                                       "vs_ridge": "la recette ne bat pas le ridge sur retards {0,1,2}+multiples de P*"},
           "taches": {}}
    for t in TACHES:
        d = charger_v2(t)
        P, vP, v012, SS = ensembles_S(d)
        ratio = vP / v012
        out["taches"][t] = {"P_etoile": P, "val_P": vP, "val_012": v012, "rapport": ratio, "S": SS,
                            "serie": RN.empreinte(d),
                            "prevision": "retards_devant" if ratio <= GAIN_SEUIL else "pas_de_victoire_prevue"}
        print(t, "P*", P, "rapport", round(ratio, 3), SS, out["taches"][t]["prevision"], flush=True)
    PREREG.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")


def lancer(tache, out):
    d = charger_v2(tache)
    pre = json.loads(PREREG.read_text(encoding="utf-8"))["taches"][tache]
    P, SS = pre["P_etoile"], pre["S"]
    rng = np.random.default_rng(12_000)
    tir = [{"rho": float(rng.uniform(0.1, 1.1)), "fuite": float(rng.uniform(0.05, 1.0)),
            "echelle": float(np.exp(rng.uniform(np.log(0.05), np.log(2.0))))} for _ in range(max(1, N_EVAL // 3))]
    cands = [dict(t, S=S, kind="perm", paire=None, origine=o) for o, S in SS.items() for t in tir]
    dent = P13.augmenter(d, SS["sauts K=3"], None)
    t0 = time.time()
    for seed in SEEDS:
        for fam in FAM:
            k = f"{tache}|{fam}|s{seed}"
            if out["cells"].get(k, {}).get("proto") == PROTO:
                continue
            if fam == "recette":
                vals = [R.evaluer(P13.construire(c, BUDGET, seed, 2), d, val=True) for c in cands]
                i = int(np.argmin(vals))
                cfg, val, dt = cands[i], vals[i], d
            else:
                dt = dent if fam == "esn_entree" else d
                cfg, val = RN.regler_famille(dt, fam, seed, N_EVAL)
            te = R.evaluer(P13.construire(cfg, BUDGET, seed, dt["U"].shape[1]), dt)
            out["cells"][k] = {"test": float(te), "val": float(val), "cfg": cfg, "proto": PROTO}
            R.ecrire(out, FICHIER)
            print(k, round(te, 4), round(val, 4), int(time.time() - t0), "s", flush=True)
    s, u = d["splits"], d["U"][:, 0].astype(np.float64)
    T = len(u)

    def ridge(lags):
        F = np.zeros((T, len(lags)), np.float32)
        for j, k in enumerate(lags):
            F[k:, j] = u[:T - k]
        te, lam, va = P7.ridge_readout(F, d["y"], s["wash"], s["tr_end"], s["te0"], s["te1"])
        return {"test": float(te), "val": float(va), "lam": float(lam), "proto": PROTO}
    yt = d["y"][s["te0"]:s["te1"]]
    out["cells"][f"{tache}|persistance"] = {"test": float(np.sqrt(np.mean((u[s['te0']:s['te1']] - yt) ** 2)) / yt.std()), "proto": PROTO}
    out["cells"][f"{tache}|ridge_lags"] = ridge(list(range(LMAX + 1)))
    out["cells"][f"{tache}|ridge_periode"] = ridge(sorted({0, 1, 2} | set(range(P, LMAX + 1, P))))
    best = None
    for D in (5, 10, 20, 30, 40):
        for prod in (False, True):
            v = R.ngrc_evaluer(d, D, prod, val=True)
            if best is None or v < best[0]:
                best = (v, D, prod)
    out["cells"][f"{tache}|ngrc"] = {"test": float(R.ngrc_evaluer(d, best[1], best[2])), "D": best[1], "proto": PROTO}
    R.ecrire(out, FICHIER)


def run(taches):
    if not PREREG.exists():
        raise SystemExit("Refusé : lancer et pousser d'abord `diagnostic`.")
    out = R.lire(FICHIER)
    for t in (taches or list(TACHES)):
        lancer(t, out)
    out["meta"] = {"proto": PROTO, "seeds": SEEDS, "n_eval": N_EVAL, "budget": BUDGET, "smoke": RN.ESSAI}
    R.ecrire(out, FICHIER)


def tableau():
    out = R.lire(FICHIER)
    pre = json.loads(PREREG.read_text(encoding="utf-8"))["taches"]
    ok = tot = 0
    for t in TACHES:
        if f"{t}|esn_fuite|s{SEEDS[-1]}" not in out["cells"]:
            print(t, ": pas encore calculé"); continue
        g = lambda f: np.array([out["cells"][f"{t}|{f}|s{s}"]["test"] for s in SEEDS])
        ref = g("esn_fuite")
        print(f"== {t} | P* {pre[t]['P_etoile']} S {pre[t]['S']['sauts K=3']} | prévision : {pre[t]['prevision']} (rapport {pre[t]['rapport']:.3f})")
        for f in FAM:
            v = g(f)
            print(f"  {f:14s} {v.mean():.4f} ± {v.std():.4f}" + ("" if f == "esn_fuite" else f"  rapport {v.mean() / ref.mean():.3f}  p = {R.wilcoxon(v, ref):.4g}"))
        for b in ("persistance", "ridge_lags", "ridge_periode", "ngrc"):
            print(f"  {b:14s} {out['cells'][f'{t}|{b}']['test']:.4f} (sans réservoir)")
        rec, dts = g("recette"), g("dts")
        p = R.wilcoxon(rec, ref)
        verdict = "gagne" if rec.mean() < ref.mean() and p < 0.05 else "perd" if rec.mean() > ref.mean() and p < 0.05 else "égalité"
        tenue = (verdict == "gagne") == (pre[t]["prevision"] == "retards_devant")
        pd_ = R.wilcoxon(rec, dts)
        vd = "gagne" if rec.mean() < dts.mean() and pd_ < 0.05 else "perd" if rec.mean() > dts.mean() and pd_ < 0.05 else "égalité"
        rp = out["cells"][f"{t}|ridge_periode"]["test"]
        print(f"  VERDICT principal (recette / ESN à fuite) : {verdict} ; prévision {'tenue' if tenue else 'NON tenue'}")
        print(f"  secondaire : recette contre DTS-ESN : {vd} (p = {pd_:.4g}, prévu : ne bat pas) ; contre ridge périodes : "
              f"{'bat' if rec.mean() < rp else 'ne bat pas'} (prévu : ne bat pas)")
        tot += 1; ok += int(tenue)
    print(f"PRÉVISION PRINCIPALE TENUE : {ok} sur {tot}")


if __name__ == "__main__":
    c = sys.argv[1] if len(sys.argv) > 1 else ""
    {"diagnostic": diagnostic, "tableau": tableau}.get(c, lambda: run(sys.argv[2:]) if c == "run" else print(__doc__))()
