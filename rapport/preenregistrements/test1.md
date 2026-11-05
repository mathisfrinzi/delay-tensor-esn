# Protocole du test sur séries réelles (écrit et poussé avant tout calcul de réservoir)

Script : `src/reel_nouveau.py`. Étape 1 (`diagnostic`, sans réservoir) : spectre de retards sur l'entraînement seul,
effectif de retards N_eff = 1/Σ w², S calculé (sauts K = 2 et 3). Prévisions : `results/prereg_reel.json`.

Règle de prévision, fixée avant tout calcul : « retards_devant » (la recette bat l'ESN à fuite) si N_eff ≤ 6, sinon « pas_de_victoire_prevue ».

| tâche | série | horizon | test (fin de série) |
|---|---|---|---|
| etth1_h24 | ETTh1, colonne OT | 24 h | 2880 pas |
| etth2_h24 | ETTh2, colonne OT | 24 h | 2880 pas |
| sunspots_h12 | taches solaires mensuelles | 12 mois | 600 mois |
| melbourne_h7 | température min. quotidienne, Melbourne | 7 jours | 730 jours |

Normalisation par moyenne et écart-type de l'entraînement. Réglage (ρ, fuite, échelle, λ) sur les 20 % finaux de l'entraînement,
jamais sur le test. Familles : ESN à fuite, DTS-ESN, tenseur à fuite (espace a priori de la partie 13), ESN avec retards en entrée
(S calculé), recette (S calculé, K ∈ {2, 3, 5} et {1..10}) ; 10 seeds de réservoir, 30 évaluations de réglage par famille et seed,
budget 400 000 connexions. Références sans réservoir : persistance, ridge sur tous les retards (jusqu'à lmax), NG-RC.
Verdict par tâche : recette contre ESN à fuite, Wilcoxon p < 0,05 sur 10 seeds ; la prévision est tenue si le verdict est
« gagne » quand elle dit « retards_devant », et « égalité » ou « perd » quand elle dit « pas_de_victoire_prevue ».
Tous les résultats seront rapportés, pertes comprises. Les séries sont uniques : les 10 seeds ne mesurent que la variance du tirage du réservoir.
