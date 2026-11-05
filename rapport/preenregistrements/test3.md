# Protocole du test sur séries réelles n° 3 (même règle que le n° 2, cinq nouvelles séries), écrit et poussé avant tout calcul de réservoir

Règle, familles, réglage, budget, seeds, références et verdicts : identiques à `protocole_reel_v2.md`, sans aucun changement (seuil 0,95 inclus). Script `src/reel_v3.py`, prévisions `results/prereg_v3.json`. Les séries n'ont pas été regardées avant ce diagnostic ; l'essai de fumée du code a utilisé les 3 000 premiers pas de chaque série, qui ne recouvrent pas le test.

| tâche | série (horizon 24 h, pas horaire) | P* | rapport val(P*)/val({0,1,2}) | prévision |
|---|---|---|---|---|
| energie_charge_h24 | charge électrique réelle, Espagne (darts `energy_dataset`) | 2 | 0,840 | retards devant |
| energie_prix_h24 | prix de l'électricité, Espagne (idem) | 24 | 0,722 | retards devant |
| solaire_h24 | production solaire, Alabama, somme des 137 centrales, moyenne horaire (laiguokun) | 32 | 0,865 | retards devant |
| meteo_T_h24 | température, Jena 2020, moyenne horaire (darts `weather`) | 73 | 0,950 (juste au-dessus du seuil) | pas de victoire |
| meteo_rayonnement_h24 | rayonnement global SWDR, Jena 2020, moyenne horaire | 72 | 0,910 | retards devant |

Longueurs : 12 000 derniers pas pour l'énergie, séries complètes pour le solaire (8 760) et Jena (≈ 8 780) ; test = 20 % (au plus 2 880).
Prévisions secondaires, comme au n° 2 : la recette ne bat ni DTS-ESN ni le ridge sur périodes.
Points faibles connus avant le calcul : les P* trouvés (2, 32, 73…) ne sont pas des cycles lisibles (24, 168) ; la grille retient des multiples qui recouvrent plusieurs cycles. Jena couvre une seule année : le cycle annuel n'est pas appris. La température de Jena est à 0,950 du seuil : sa prévision est fragile.
