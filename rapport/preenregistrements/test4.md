# Protocole du test sur séries réelles n° 4, écrit et poussé avant tout calcul de réservoir

Date d'écriture : 2026-11-04. Script : `src/reel_v4.py`. Prévisions chiffrées : `results/prereg_v4.json`, produit par `python src/reel_v4.py diagnostic` (sans réservoir, entraînement seul) et poussé dans le même commit que ce fichier. `run` refuse de démarrer sans ce fichier.

## Pourquoi ce test

Après les tests 2 et 3, une comparaison faite après coup (exploratoire) a montré que des délais de propagation tirés de la géométrie (Iacob et Dambre, 2024) font aussi bien que la recette sur les séries réelles. Ce test pose la question sur des séries jamais ouvertes, avec la règle écrite avant : quand le calcul de S apporte-t-il quelque chose par rapport à ces délais ?

## Séries (jamais ouvertes dans ce projet)

| tâche | source | pas | série |
|---|---|---|---|
| `pekin_pm25_h24` | `jbrownlee/Datasets`, `pollution.csv` | 1 h | PM2.5 à Pékin, valeurs manquantes comblées par la précédente |
| `pekin_T_h24` | idem | 1 h | température à Pékin |
| `pekin_pression_h24` | idem | 1 h | pression à Pékin |
| `taxi_h24` | `numenta/NAB`, `realKnownCause/nyc_taxi.csv` | 1 h (somme des demi-heures) | passagers des taxis de New York |
| `pjm_charge_h24` | `panambY/Hourly_Energy_Consumption`, `PJME_hourly.csv` | 1 h | consommation électrique PJM Est |
| `change_h24` | `laiguokun/multivariate-time-series-data`, `exchange_rate` | 1 jour | taux de change, première devise |

Seules les métadonnées ont été regardées avant le diagnostic : en-têtes, nombre de lignes, régularité des horodatages, nombre de valeurs manquantes.

**Découpe**, identique au test 3 : les 12 000 derniers pas (ou toute la série si elle est plus courte), horizon 24 pas, test = les 2 880 derniers pas ou 20 % de la série, normalisation par l'entraînement seul, transitoire de 400 pas.

## Familles

Toutes au budget de 400 000 connexions (ou N = 632 neurones pour les réservoirs creux), 10 seeds, 30 évaluations de réglage sur la validation (20 % finaux de l'entraînement), le test ouvert une fois.

- **ESN à fuite.**
- **Recette.** Le S est calculé par le diagnostic par périodes du test 2, sans changement (sauts K = 2, 3 et 5), avec des tranches permutées.
- **Iacob-Dambre**, positions au hasard (`src/iacob_dambre.py`), et ses trois mélanges A, B et C, qui lisent le S retenu par la recette à la même seed.
- **Iacob-Dambre + CMA-ES** (`src/references2.py`) : positions tirées d'un mélange de 4 gaussiennes, génome optimisé par CMA-ES, 160 évaluations. C'est plus que les autres familles : avantage de réglage assumé.
- **self-ESN** (Xu et al., 2024), **lecture retardée** sur [x(t), x(t − τ)] pour τ dans le S de la recette (Holzmann et Hauser, 2010), **réservoir en cycle** (Rodan et Tiňo, 2011), **réservoir orthogonal**.
- Sans réservoir : persistance, ridge sur 200 âges, ridge sur {0, 1, 2} + multiples de P\*.

## Prévisions

Un verdict « gagne » demande une moyenne plus basse et un Wilcoxon apparié p < 0,05 sur les 10 seeds.

1. **Règle 1 (celle des tests 2 et 3, inchangée).** La recette bat l'ESN à fuite si le rapport val(P\*) / val({0, 1, 2}) ≤ 0,95. Sinon, pas de victoire prévue.
2. **Règle 2 (nouvelle).** La recette bat Iacob-Dambre (positions au hasard) si P\* > 40 **et** si le rapport est ≤ 0,95. Sinon, pas de victoire prévue. Justification : leur dmax ne dépasse pas 40, et leurs délais se concentrent vers un tiers de dmax. Une période plus longue ne peut donc s'atteindre qu'en plusieurs sauts, alors que la recette la place exactement. Cette règle n'a été calée que sur les sept séries de la partie 5.5, où elle aurait été tenue 5 fois sur 7.

   **Correction avant tout réservoir, à signaler.** La première version disait seulement « si P\* > 40 ». Au premier passage du diagnostic, `change_h24` a donné P\* = 100 avec un rapport de 1,008 : les multiples de P\* n'y apportent rien, et la règle aurait prévu une victoire de la recette pour une période sans contenu. La condition « rapport ≤ 0,95 » a donc été ajoutée. C'est la seule modification, faite après avoir vu les sorties du diagnostic (entraînement seul) et avant tout calcul de réservoir. Les deux versions donnent la même prévision sur les cinq autres séries.
3. **Secondaire.** Iacob-Dambre + CMA-ES bat Iacob-Dambre à positions au hasard sur au moins 4 séries sur 6.

Aucune prévision pour A, B, C, le self-ESN, la lecture retardée, le cycle et l'orthogonal : ces familles sont rapportées telles quelles.

## Vérifications avant le lancement

Un essai réduit (`ESSAI=1`) a tourné sur `pekin_T_h24` : 2 seeds, sur les 3 000 premiers pas de la fenêtre de 12 000, une partie de l'entraînement qui ne recouvre pas le test. Ses chiffres ne servent qu'à vérifier la chaîne et ne sont pas rapportés. Avant de calculer, `run` vérifie que chaque série téléchargée a la même empreinte que celle du diagnostic.

## Ce qui sera rapporté

Le tableau complet (`python src/reel_v4.py tableau`), pertes comprises, et le décompte de chaque règle. Si un défaut de code est trouvé après le lancement, il sera corrigé et signalé, et le résultat d'avant correction restera visible.
