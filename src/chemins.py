"""Dossier des résultats bruts des scripts des parties 1 et des fausses pistes.

Par défaut results/brut/ dans ce dépôt ; la variable d'environnement TENSEUR_BRUT le change.
"""
import os
from pathlib import Path

BRUT = Path(os.environ.get("TENSEUR_BRUT", Path(__file__).resolve().parents[1] / "results" / "brut"))
