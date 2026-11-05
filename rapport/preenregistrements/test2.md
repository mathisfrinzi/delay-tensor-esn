# Protocole du test sur séries réelles n° 2 (diagnostic par périodes), écrit et poussé avant tout calcul de réservoir

Origine : 15.1 (le diagnostic par retard isolé rate une information portée par des multiples d'une période) et l'essai exploratoire sur ETTh1 (S = [1, 48, 96] : 0,687 contre 0,711 pour l'ESN à fuite). ETTh1 et ETTh2 sont ouvertes et ne servent plus de test.
Script : `src/reel_v2.py` (diagnostic : `src/reel_corrige.py`). Prévisions : `results/prereg_v2.json`.

Séries (jamais regardées avant le diagnostic) : électricité et trafic de `laiguokun/multivariate-time-series-data`, moyenne sur tous les clients / capteurs, les 12 000 derniers pas horaires ; horizon 24 h ; test = 2 880 derniers pas ; normalisation par l'entraînement.

Diagnostic (entraînement seul) : pour chaque période P de 2 à 100, erreur de validation d'un ridge sur {0, 1, 2} ∪ {kP ≤ 200} ; P* = argmin ; poids moitié sur {1, 2}, moitié sur les multiples de P* ; S = sauts K = 2, 3, 5.
Règle de prévision, fixée avant le calcul : « retards_devant » (la recette bat l'ESN à fuite, Wilcoxon p < 0,05 sur 10 seeds) si val(P*) / val({0,1,2}) ≤ 0,95, sinon « pas_de_victoire_prevue ».
Prévisions secondaires, fixées avant le calcul : la recette ne bat pas DTS-ESN, et ne bat pas le ridge sur {0,1,2} ∪ multiples de P* (observé sur ETTh1).

| tâche | P* | rapport val(P*)/val({0,1,2}) | S (K = 3) | prévision |
|---|---|---|---|---|
| électricité h24 | 24 | 0,972 | [1, 24, 72] | pas de victoire |
| trafic h24 | 4 | 0,455 | [1, 12, 28] | retards devant |

Familles et réglage identiques à `protocole_reel.md` (ESN à fuite, DTS-ESN, tenseur à fuite, ESN avec retards en entrée de S K = 3, recette corrigée avec 3 ensembles S × tirages) ; budget 400 000, 10 seeds, 30 évaluations de réglage ; références sans réservoir : persistance, ridge sur 200 retards, ridge sur périodes, NG-RC.

Divulgation : avant les prévisions, un essai de fumée (séries tronquées aux 3 000 derniers pas, 2 seeds) a été lancé pour vérifier le code. Il a affiché des erreurs de test sur les 500 derniers pas de l'électricité (recette corrigée ≈ 0,46, ESN à fuite ≈ 0,50), qui recouvrent le test réel. Les prévisions ne dépendent que de l'entraînement et de la règle ci-dessus, fixée avant. Ce recouvrement est un point à garder en tête pour l'électricité.
