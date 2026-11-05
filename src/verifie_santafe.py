"""Compare SF_A.dat (reconstruit depuis reservoirpy) a l'original de McNames (SantaFeA.dat).
Usage : python verifie_santafe.py <original> <SF_A.dat> <results_newtasks.json ...>
Identique : affiche SANTAFE_IDENTIQUE (rien a refaire). Different : remplace SF_A.dat par
l'original (1000 premiers points) et supprime la cellule 'santafe' des JSON donnes, pour que
`suite_newtasks.py santafe` la recalcule ; affiche SANTAFE_DIFFERENT."""
import sys, json
import numpy as np
orig = np.loadtxt(sys.argv[1]).ravel()
sf = np.loadtxt(sys.argv[2], delimiter=",").ravel()
print(f"original : {len(orig)} valeurs, SF_A.dat : {len(sf)} valeurs")
if len(orig) >= 1000 and np.array_equal(orig[:1000], sf[:1000]):
    print("SANTAFE_IDENTIQUE : les 1000 premiers points sont les memes ; le fichier SF_A.dat est bon.")
else:
    n = int(np.sum(orig[:1000] != sf[:len(orig[:1000])])) if len(sf) >= len(orig[:1000]) else -1
    print(f"SANTAFE_DIFFERENT : {n} points differents. On prend l'original.")
    np.savetxt(sys.argv[2], orig[:1000], fmt="%d")
    for p in sys.argv[3:]:
        d = json.load(open(p, encoding="utf-8-sig"))
        if d.pop("santafe", None) is not None:
            open(p, "w", encoding="utf-8").write(json.dumps(d, indent=2, ensure_ascii=False))
            print("cellule santafe supprimee de", p)
