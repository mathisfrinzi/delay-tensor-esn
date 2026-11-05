"""Les familles de référence (partie 4) se construisent et la lecture par paquets égale la lecture directe."""
import numpy as np

import p7_common as P7
import references2 as F
import reservoir as R


def test_lecture_par_paquets_egale_lecture_directe():
    d = R.charger("narma30", 0)
    s = d["splits"]
    res = R.construire(0, [1], 80, 0.9, fuite=0.6)
    taus = [6, 10, 40]
    Fm = R.simuler(res, d["U"])
    Fl = np.hstack([Fm] + [F._decaler(Fm, k) for k in taus])
    a = P7.ridge_readout(Fl, d["y"], s["wash"], s["tr_end"], s["te0"], s["te1"])[0]
    b = F.ridge_paquets(Fm, taus, d["y"], s["wash"], s["tr_end"], s["te0"], s["te1"], False, pas=700)
    assert abs(a - b) < 1e-6


def test_familles_se_construisent():
    rng = np.random.default_rng(0)
    for fam in F.FAM_REF:
        cfg = F.tirer(fam, rng)
        res = F.construire(fam, cfg, 0, 2)
        X = R.simuler(res, rng.uniform(0, 0.5, (50, 2)).astype(np.float32))
        assert X.shape == (50, F.N0) and np.isfinite(X).all()


def test_self_esn_mnist_borne_a_27():
    ms = [F.tirer("self_esn", np.random.default_rng(i), "mnist")["m"] for i in range(300)]
    assert 2 <= min(ms) and max(ms) <= 27


def test_cycle_est_une_permutation_circulaire():
    res = F.construire("cycle", {"rho": 0.5, "fuite": 1.0, "echelle": 1.0}, 0, 2)
    perm = res["tranches"][0][2]
    assert sorted(perm) == list(range(F.N0)) and perm[1] == 0


def test_cma_descend_sur_une_quadratique():
    v, z = F.cma(lambda z: float(((z - 0.3) ** 2).sum()), 5, 8, 30, np.random.default_rng(0))
    assert v < 1e-3 and np.abs(z - 0.3).max() < 0.05
