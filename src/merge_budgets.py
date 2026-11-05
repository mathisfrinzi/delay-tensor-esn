"""Fusionne les fichiers d'aide aide_budgets_<tache>.json dans
results_p12_budgets.json (protocole 14.1), après une exécution parallèle de
aide_budgets.py. Les fichiers marqués « essai » (ESSAI=1) sont ignorés.

    python src/merge_budgets.py

Règles (mêmes que merge_equite) :
- jamais supprimer une clé du fichier principal ;
- une cellule proto de l'aide remplace une cellule sans marqueur ;
- une cellule déjà proto fait foi (jamais écrasée) ;
- les clés « |ancien » s'ajoutent si absentes.
Le meta final (mêmes champs que budgets()) est écrit quand tout est complet ;
sinon un meta périmé sans marqueur proto est retiré. Une sauvegarde de
l'ancien fichier est gardée dans results/backup_budgets.json.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p12_synthese as P12  # noqa: E402

RES = P12.RES
PROTO = P12.PROTO
FICHIER = RES / "results_p12_budgets.json"
# familles écrites par budgets() ; ngrc seulement sur les tâches séquentielles
FAMS = ("esn_fuite", "dts", "tenseur_fuite", "uniforme_10", "recette")
TACHES_SEQ = ("narma10", "narma30", "parite2", "silverbox", "wh")


def main():
    if FICHIER.exists():
        merged = json.load(open(FICHIER))
    else:
        merged = {"cells": {}}
    cells = merged.setdefault("cells", {})
    n = 0
    for fichier in sorted(RES.glob("aide_budgets_*.json")):
        aide = json.load(open(fichier))
        if aide.get("essai"):
            print("ignoré (essai) :", fichier.name)
            continue
        for key, cell in aide.get("cells", {}).items():
            if key not in cells:
                cells[key] = cell
                n += 1
            elif key.endswith("|ancien"):
                continue
            elif (isinstance(cell, dict) and cell.get("proto") == PROTO
                  and isinstance(cells[key], dict) and cells[key].get("proto") != PROTO):
                cells[key] = cell
                n += 1
    manque = []
    for tache in P12.TACHES_BUDGET:
        for B in P12.BUDGETS:
            mk = f"{tache}|B{B}|meta"
            if not (isinstance(cells.get(mk), dict) and cells[mk].get("proto") == PROTO):
                manque.append(mk)
                continue
            fams = FAMS + (("ngrc",) if tache in TACHES_SEQ else ())
            for fam in fams:
                if fam not in cells[mk]:
                    manque.append(mk + "|" + fam)
                for seed in P12.SEEDS:
                    key = f"{tache}|B{B}|{fam}|s{seed}"
                    if not (isinstance(cells.get(key), dict) and cells[key].get("proto") == PROTO):
                        manque.append(key)
    if not manque:
        merged["meta"] = {"budgets": P12.BUDGETS, "seeds": P12.SEEDS, "n_reglage": P12.N_BUDGET,
                          "seeds_reglage": "100 à 104 pour les tâches dont les données dépendent de la seed (14.1)",
                          "proto": PROTO,
                          "erreur": "NRMSE ; taux d'erreur (mnist) ; 300 − capacité (mc)"}
        print("meta final écrit")
    else:
        meta = merged.get("meta")
        if not (isinstance(meta, dict) and meta.get("proto") == PROTO):
            merged.pop("meta", None)
        print(len(manque), "cellules manquantes (meta non écrit) :",
              manque[:5], "…" if len(manque) > 5 else "")
    if FICHIER.exists():
        (RES / "backup_budgets.json").write_bytes(FICHIER.read_bytes())
    json.dump(merged, open(FICHIER, "w", encoding="utf-8"), indent=1, ensure_ascii=True)
    json.load(open(FICHIER))  # règle de la partie 14 : JSON complet avant commit
    nproto = sum(1 for c in cells.values() if isinstance(c, dict) and c.get("proto") == PROTO)
    print(f"merge : +{n} cellules, {nproto} proto 14.1, {len(cells)} clés -> results_p12_budgets.json")


if __name__ == "__main__":
    main()
