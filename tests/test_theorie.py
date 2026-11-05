"""Les deux résultats de la partie 2 : la règle et sa réciproque générique, et la borne par les sauts."""
import numpy as np

import reservoir as R
import theorie as Th


def _cas(S, kind, rho, fuite, seed=0, N=60, lmax=60):
    res = R.construire(seed, S, N, rho, kinds={t: (kind if t > 1 else "dense") for t in S}, fuite=fuite)
    u = np.random.default_rng(100 + seed).uniform(0, 0.5, 300 + lmax)
    J = Th.jacobiens(res, u, 200, lmax)
    B, s, Sp = Th.borne(res, lmax)
    return J, B, s, Th.sauts(Sp, lmax)


def test_sensibilite_nulle_hors_du_semi_groupe():
    J, _, _, h = _cas([3, 7], "dense", 0.9, 1.0)
    assert np.all(J[h >= 10 ** 8] == 0.0)


def test_reciproque_sur_plusieurs_tirages():
    for seed in range(5):
        J, _, _, h = _cas([4, 9], "perm", 0.8, 1.0, seed=seed, lmax=40)
        assert J[h < 10 ** 8].min() > 0.0


def test_borne_jamais_violee():
    for S, kind, rho, a in ([3, 7], "dense", 0.9, 1.0), ([1, 30], "perm", 0.9, 0.5), ([1], "dense", 0.4, 1.0):
        J, B, _, _ = _cas(S, kind, rho, a)
        assert np.all(J <= B * (1 + 1e-6) + 1e-300)


def test_borne_geometrique_en_sauts():
    J, B, s, h = _cas([3, 7], "dense", 0.3, 1.0)
    assert s < 1
    ok = h < 10 ** 8
    assert np.all(B[ok] <= B[0] * s ** h[ok] * (1 + 1e-9))


def test_sauts_rendu_de_monnaie():
    h = Th.sauts([3, 7], 15)
    assert list(h[[0, 3, 6, 7, 10, 13, 14]]) == [0, 1, 2, 1, 2, 3, 2]
    assert h[11] >= 10 ** 8
