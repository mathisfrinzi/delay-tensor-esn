"""Les tests tournent en version réduite (ESSAI=1) et lisent le code dans src/."""
import os
import sys
from pathlib import Path

os.environ.setdefault("ESSAI", "1")
os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
