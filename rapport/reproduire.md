[Sommaire](../README.md#le-rapport)

# Reproduire

## Installation

```bash
pip install -r requirements.txt pytest   # numpy, scipy, pandas, matplotlib
pip install numba                        # facultatif : accélère la partie 6
```

MNIST est téléchargé au premier lancement. Silverbox et Wiener-Hammerstein (nonlinearbenchmark.org) sont lus aux chemins `/tmp/datasets/silverbox/SilverboxFiles/SNLS80mV.mat` et `/tmp/datasets/WH.mat` (`load_phys` dans `src/p7_common.py`). Les séries réelles de la partie 5 sont téléchargées depuis GitHub au premier lancement, dans le dossier `REEL_DATA` (défaut `/tmp/datasets_reel`).

Sous Windows, lancer Python avec `PYTHONUTF8=1`.

## Figures et schémas

Elles se refont à partir des résultats enregistrés dans `results/`, sans relancer les simulations :

```bash
python src/schemas.py         # tous les schémas (results/schemas/)
python src/plot_figures.py    # théorie (2.3), carte des références (4.7), test 4 (5.6), bilan
python src/plot_reel.py       # tests 1 à 3
python src/plot_iacob.py      # parties 4.6 et 5.5
python src/plot_partie1_bilan.py
```

Chaque script `plot_*.py` refait la figure de même nom.

## Scripts par partie

Les noms de scripts (`p4_`, `p6_`, `p7_`, `p8_`, `p12_`) et de figures (`partie2_*`, `partie6_*`, `partie12_*`…) suivent une numérotation de travail, différente de celle des parties. La correspondance :

| Partie | Simulations | Figures |
|---|---|---|
| [1](1_ou_ca_gagne.md) | `esn_tensor.py`, `suite_part1.py`, `suite_part1b.py`, `suite_real.py`, `mnist_esn.py`, `final60k.py`, `fashion.py`, `henon.py`, `suite_newtasks.py`, `narma_tau_sweep*.py`, `phys_tau_sweep.py`, `mnist_tau_sweep.py`, `tau20_sweep.py`, `sweep_ensembles.py`, `prune_tau.py`, `eval_pruned.py`, `alpha_tau10.py`, `band_tau*.py`, `cp_rank.py`, `esn_ref.py`, `p7_4_stats.py` | `plot_partie1_bilan.py`, `plot_newtasks.py`, `plot_sweep.py`, `plot_synthese2.py`, `plot_partie2.py` |
| [2](2_regle.md) | `p6_common.py`, `p6_trous.py`, `p6_planchers.py`, `p6_balayage_d.py`, `p6_dents.py`, `p6_narma_modif.py`, `p6_champ_moyen.py`, `p11_trous_mous.py banc`, `theorie.py` | `plot_p6_*.py`, `plot_partie5_arith.py`, `plot_partie5_renouvellement.py`, `plot_figures.py` |
| [3](3_calculer_S.md) | `p8_spectre.py`, `p8_calcul.py`, `p8_conception.py`, `controle_conception.py` (`reglage`, `boutons`, `nombre`), `p4_tenseur.py`, `p4_cp.py`, `p12_synthese.py recette` | `plot_p8_*.py`, `plot_p4_tenseur.py`, `controle_conception.py figures` |
| [4](4_comparaisons.md) | `p7_common.py`, `p7_1_tranches.py`, `p7_2_entree.py`, `p7_5_ngrc_min.py`, `ngrc.py`, `p12_synthese.py` (`equite`, `budgets`, `stats`), `courbes_apprentissage.py`, `iacob_dambre.py`, `references2.py` | `plot_controles.py`, `p12_synthese.py figures`, `plot_iacob.py`, `plot_figures.py` |
| [5](5_series_reelles.md) | `reel_nouveau.py`, `reel_corrige.py`, `reel_v2.py`, `reel_v3.py`, `reel_v4.py`, `iacob_dambre.py`, `references2.py` | `plot_reel.py`, `plot_iacob.py`, `plot_figures.py` |
| [6](6_reservoir_physique.md) | `p10_tdr.py` | `p10_tdr.py figures` |

## Le moteur commun

Les parties 3 (à partir de 3.3), 4 (à partir de 4.3), 5 et 6 passent par un moteur commun, [`src/reservoir.py`](../src/reservoir.py), et les outils de conception de [`src/conception.py`](../src/conception.py) (spectre, sauts, enveloppe, paire d'ordre 2). Les vérifications d'abord :

```bash
pytest          # tests/ : moteur, théorèmes de la partie 2, familles de référence (version réduite, une seconde)
```

Elles contrôlent que le moteur reproduit exactement l'ESN et les tranches permutées, que la règle des âges atteignables, sa réciproque et la borne par les sauts sont respectées, que le spectre retrouve un noyau planté, que le réservoir physique est causal et a la diversité de décalages attendue, que la lecture retardée par paquets égale la lecture directe et que CMA-ES converge sur une quadratique. Les mêmes tests tournent en intégration continue (`.github/workflows/tests.yml`), avec les schémas et les figures.

```bash
# Partie 3 : réglage égal, boutons, nombre de retards
python src/controle_conception.py reglage
python src/controle_conception.py boutons
python src/controle_conception.py nombre
python src/controle_conception.py figures

# Partie 4 : réglage équitable, recette, budgets
python src/p12_synthese.py equite
python src/p12_synthese.py recette
python src/p12_synthese.py budgets
python src/p12_synthese.py stats
python src/p12_synthese.py figures
python src/courbes_apprentissage.py courbes

# Partie 5 : séries réelles (diagnostic, puis pousser les prévisions, puis run)
python src/reel_v3.py diagnostic       # écrit results/prereg_v3.json, sans réservoir
python src/reel_v3.py run              # refuse de partir sans prereg_v3.json
python src/reel_v3.py tableau

# Parties 4.6 et 5.5 : Iacob et Dambre, et trois mélanges (après les tests précédents : lit leurs S calculés)
IACOB_PROCS=6 python src/iacob_dambre.py run      # 15 tâches, environ 60 h de calcul sur un cœur
python src/iacob_dambre.py tableau
python src/plot_iacob.py

# Partie 2.3 : vérification numérique des théorèmes
python src/theorie.py

# Parties 4.6 (CMA-ES), 4.7 et 5.5 : références manquantes et Iacob-Dambre optimisé
IACOB_PROCS=6 python src/references2.py run
python src/references2.py tableau

# Partie 5.6 : test 4 (diagnostic, pousser les prévisions, puis run)
python src/reel_v4.py diagnostic       # écrit results/prereg_v4.json, sans réservoir
IACOB_PROCS=6 python src/reel_v4.py run   # refuse sans prereg_v4.json ; vérifie l'empreinte des séries
python src/reel_v4.py tableau

# Partie 6 : réservoir physique
python src/p10_tdr.py calibrer
python src/p10_tdr.py balayage
python src/p10_tdr.py conception somme
python src/p10_tdr.py tolerance somme
python src/p10_tdr.py figures
P10_N=200 python src/p10_tdr.py conception somme   # plus de nœuds virtuels
```

Avec `ESSAI=1`, chaque sous-commande tourne en version réduite pour vérifier la chaîne de bout en bout ; ces résultats ne doivent jamais être rapportés. Chaque sous-commande écrit un `results/results_*.json` au format {cells, meta} et reprend là où elle s'est arrêtée.

## Protocole par tâche (moteur commun)

Toutes les tâches : W_in uniforme sur [−0,5 ; 0,5] multipliée par l'échelle d'entrée, entrée [u(t), 1] (28 pixels pour MNIST), état nul avant t = 0, lecture ridge avec λ ∈ {10⁻⁸, 10⁻⁶, 10⁻⁴, 10⁻²}. La validation est faite des 20 % finaux du bloc d'entraînement ; tous les réglages y sont choisis, puis la lecture est réapprise sur tout le bloc.

| tâche | longueur | transitoire | test | normalisation |
|---|---|---|---|---|
| NARMA-10, NARMA-30 | 5 200 pas | 200 pas | 1 000 derniers pas | u uniforme sur [0 ; 0,5] |
| parité L = 2, D = 10 | 5 200 pas | 200 pas | 1 000 derniers pas | bits 0/1 |
| Silverbox, Wiener-Hammerstein | 30 000 premiers points | 500 points | 10 000 derniers | u / std(u), y / std(y[:100 000]) |
| capacité mémoire, k = 1 à 300 | 12 000 pas | 1 000 pas | après les 9 000 premiers pas | u uniforme sur [0 ; 0,5] |
| MNIST lignes | 5 000 images d'entraînement | | 1 000 premières images du test officiel | pixels / 255, lecture sur l'état final |
| noyaux plantés | 6 000 pas | 200 pas | 1 000 derniers pas | entrée AR(1) de lissage φ |
| séries réelles (partie 5) | voir la partie 5 | 200 à 400 pas | fin de la série | moyenne et écart-type de l'entraînement |

Les parties 1, 2 et le début des parties 3 et 4 viennent de codes plus anciens (`esn_tensor.py`, `suite_*.py`, `p6_common.py`, `p7_common.py`), avec les mêmes conventions mais d'autres longueurs (par exemple NARMA sur 6 000 pas dans `esn_tensor.py`, Silverbox et WH sur 40 000 points d'entraînement dans `suite_real.py`).

Les scripts de la campagne de la partie 1 et des fausses pistes écrivent leurs résultats bruts dans `results/brut/` (variable `TENSEUR_BRUT` pour changer de dossier) ; les fichiers utilisés par le rapport sont copiés dans `results/`.

---

[Sommaire](../README.md#le-rapport)
