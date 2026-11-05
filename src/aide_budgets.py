"""Aide parallèle à 12.3 (budgets, protocole 14.1) : calcule toutes les cellules
d'une tâche, pour lancer les 7 tâches en parallèle (plusieurs terminaux, ou la
cellule dédiée de pipeline.ipynb).

    python src/aide_budgets.py narma10 narma30 mc parite2 silverbox wh mnist

- S'appuie sur p12_synthese (mêmes tirages, mêmes fonctions regler_multi,
  recette_regler_multi, construire, evaluer) : résultats identiques à la boucle
  budgets(), cellule par cellule.
- Ne modifie JAMAIS results_p12_budgets.json (lu en seul) : écrit
  results/aide_budgets_<tache>.json, reprise automatique. La fusion dans le
  fichier principal se fait avec merge_budgets.py.
- Les cellules déjà marquées proto du fichier principal (fusion précédente) ne
  sont pas recalculées ; une ancienne cellule sans marqueur est conservée sous
  « |ancien » dans le fichier d'aide.
- ESSAI=1 : tailles réduites pour vérifier que le code tourne ; le fichier est
  marqué « essai » et ignoré par merge_budgets.py (jamais fusionné).
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import reservoir as R  # noqa: E402
import p12_synthese as P12  # noqa: E402

RES = R.RES
PROTO = P12.PROTO


def charger_json(chemin):
    if Path(chemin).exists():
        return json.load(open(chemin))
    return {"cells": {}}


def main():
    taches = [t for t in sys.argv[1:] if t in P12.TACHES_BUDGET]
    if not taches:
        print("usage : python src/aide_budgets.py <tache...> parmi", P12.TACHES_BUDGET)
        return
    base = charger_json(RES / "results_p12_budgets.json")["cells"]
    base_proto = {k for k, c in base.items()
                  if isinstance(c, dict) and c.get("proto") == PROTO}
    print("base :", len(base), "cellules,", len(base_proto), "proto 14.1", flush=True)
    for tache in taches:
        fichier = RES / f"aide_budgets_{tache}.json"
        local = charger_json(fichier)
        local["essai"] = bool(R.ESSAI)
        for B in P12.BUDGETS:
            mk = f"{tache}|B{B}|meta"
            m = None
            if isinstance(local["cells"].get(mk), dict) and local["cells"][mk].get("proto") == PROTO:
                m = local["cells"][mk]
            elif mk in base_proto and "tenseur_fuite" in base[mk]:
                m = base[mk]
                local["cells"][mk] = m
                R.ecrire(local, fichier)
            if m is None:
                sr = P12.seeds_reglage(tache)
                datas = {s: R.charger(tache, s) for s in sr}
                m = {"proto": PROTO, "seeds_reglage": sr}
                for fam in ("esn_fuite", "dts", "tenseur_fuite"):
                    m[fam] = P12.regler_multi(tache, fam, B, sr, P12.N_BUDGET)
                    print(tache, B, fam, "regle", flush=True)
                best = None
                for rho in (0.5, 0.9, 0.99):
                    c = {"S": list(range(1, 11)), "kind": "dense", "rho": rho}
                    v = float(np.mean([R.evaluer(P12.construire(c, B, s, R.d_entree(tache)),
                                                 datas[s], val=True) for s in sr]))
                    if best is None or v < best[0]:
                        best = (v, c)
                m["uniforme_10"] = best[1]
                cands, _ = P12.candidats_recette(tache, B, P12.N_BUDGET)
                m["recette"] = P12.recette_regler_multi(tache, sr, B, cands)
                if datas[sr[0]]["type"] == "seq":
                    bestn = None
                    for D in (5, 10, 20, 30, 40):
                        for prod in (False, True):
                            v = float(np.mean([R.ngrc_evaluer(datas[s], D, prod, val=True)
                                               for s in sr]))
                            if bestn is None or v < bestn[0]:
                                bestn = (v, {"D": D, "produits": prod})
                    m["ngrc"] = bestn[1]
                local["cells"][mk] = m
                R.ecrire(local, fichier)
                print(tache, B, "meta fait", flush=True)
            for seed in P12.SEEDS:
                data = R.charger(tache, seed)
                for fam, cfg in m.items():
                    if fam in ("proto", "seeds_reglage"):
                        continue
                    key = f"{tache}|B{B}|{fam}|s{seed}"
                    if key in base_proto:
                        continue
                    cell = local["cells"].get(key)
                    if isinstance(cell, dict) and cell.get("proto") == PROTO:
                        continue
                    rcell = base.get(key)
                    if isinstance(rcell, dict) and rcell.get("proto") != PROTO:
                        ak = key + "|ancien"
                        if ak not in base and ak not in local["cells"]:
                            local["cells"][ak] = rcell
                            R.ecrire(local, fichier)
                    if fam == "ngrc":
                        err = R.ngrc_evaluer(data, cfg["D"], cfg["produits"])
                        D = cfg["D"]
                        npar = D + (D * (D + 1) // 2 if cfg["produits"] else 0) + 1
                        local["cells"][key] = {"test": err, "lecture": npar,
                                              "connexions": 0, "proto": PROTO}
                    else:
                        t0 = time.perf_counter()
                        res = P12.construire(cfg, B, seed, R.d_entree(tache))
                        err = R.evaluer(res, data)
                        local["cells"][key] = {"test": err, "lecture": res["N"] + 1,
                                              "connexions": R.connexions(res),
                                              "couts": R.couts(res),
                                              "secondes": round(time.perf_counter() - t0, 2),
                                              "proto": PROTO}
                    R.ecrire(local, fichier)
                    print(key, round(local["cells"][key]["test"], 4), flush=True)
        n_c = sum(1 for k in local["cells"] if not k.endswith("|ancien"))
        print("tache finie :", tache, "-", n_c, "cellules proto", flush=True)


if __name__ == "__main__":
    main()
