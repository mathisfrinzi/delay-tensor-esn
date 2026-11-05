"""Vérifications du moteur commun (src/reservoir.py), à lancer avant toute campagne :

    pytest

Chaque test compare le nouveau code à une référence déjà utilisée dans le rapport, ou
à une propriété exacte (règle des sommes de retards).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import reservoir as R  # noqa: E402
import conception as C  # noqa: E402
import p10_tdr as T10  # noqa: E402

P7 = R.P7


def test_esn_identique_a_p7():
    """S = {1}, sans fuite : mêmes états que p7_common.run_states."""
    U, _, _ = P7.narma_x(30, 0, 400)
    res = R.construire(3, [1], 50, 0.9)
    Win, W, taus, _ = P7.make_reservoir(3, [1], 0.9, N=50)
    assert np.allclose(res["Win"], Win)
    F1 = R.simuler(res, U)
    F2 = P7.run_states(U, Win, W, taus)
    assert np.abs(F1 - F2).max() < 1e-4


def test_tranches_gratuites_identiques_a_p7():
    """S = {1, 3, 7} avec tranches permutées : mêmes états que p7_common."""
    U, _, _ = P7.narma_x(30, 1, 300)
    kinds = {3: "perm", 7: "perm"}
    res = R.construire(5, [1, 3, 7], 40, 0.9, kinds=kinds)
    Win, W, taus, _ = P7.make_reservoir(5, [1, 3, 7], 0.9, N=40, kinds=kinds)
    F1 = R.simuler(res, U)
    F2 = P7.run_states(U, Win, W, taus)
    assert np.abs(F1 - F2).max() < 1e-4


def test_fuite_un_vaut_sans_fuite():
    U, _, _ = P7.narma_x(30, 2, 300)
    a = R.simuler(R.construire(1, [1, 4], 30, 0.8, fuite=1.0), U)
    b = R.simuler(R.construire(1, [1, 4], 30, 0.8), U)
    assert np.allclose(a, b)


def test_lots_mnist_egaux_aux_sequences():
    """Simuler B séquences ensemble = les simuler une par une."""
    rng = np.random.default_rng(0)
    X = rng.uniform(size=(28, 3, 28)).astype(np.float32)
    res = R.construire(0, [1, 2, 5], 30, 0.9, d=28, kinds={2: "diag", 5: "perm"}, fuite=0.6)
    lot = R.simuler(res, X, final=True)
    une = np.stack([R.simuler(res, X[:, b, :])[-1] for b in range(3)])
    assert np.abs(lot - une).max() < 1e-5


def test_regle_des_sommes():
    """S = {3, 7} : la trace d'une impulsion est exactement nulle aux âges 1, 2, 4, 5, 8, 11."""
    res = R.construire(0, [3, 7], 40, 0.9)
    e = R.profil_lineaire(res, 20)
    for L in (1, 2, 4, 5, 8, 11):
        assert e[L] == 0.0
    for L in (3, 6, 7, 9, 10, 12, 13):
        assert e[L] > 0.0


def test_mc_rapide_egal_mc_profile():
    U, u = P7.mc_input(0)
    F = R.simuler(R.construire(0, [1, 5], 40, 0.9), U)
    a = R.mc_rapide(F, u, 20)
    b = P7.mc_profile(F, u, kmax=20)
    assert np.allclose(a, [b[str(k)] for k in range(1, 21)], atol=1e-6)


def test_sauts_et_support():
    h = C.sauts([1, 10], 31)
    assert h[30] == 3 and h[29] == 11 and h[1] == 1
    w = np.zeros(41)
    w[[1, 30]] = 0.5
    S, _ = C.support_sauts(w, 2, 40, 40)
    assert S == [1, 30]


def test_spectre_retrouve_un_noyau_plante():
    rng = np.random.default_rng(0)
    u = rng.uniform(0, 0.5, 8000)
    y = np.zeros_like(u)
    y[12:] = 2.0 * u[7:-5] * u[:-12] + 0.5 * u[:-12]
    spec = C.spectre(u, y, lmax=15)
    top = np.argsort(spec["I_lag"])[-2:]
    assert set(top.tolist()) == {5, 12}
    assert C.paire_produit(spec) == (5, 12)


def test_tdr_retard_egal_horloge_est_identite():
    """Un seul retard D = N : chaque nœud ne lit que lui-même un tour plus tôt
    (Δ = 1) ; D = N + 1 : un décalage par âge."""
    N = 10
    assert T10.diversite([N], N, 8) == 1
    assert T10.diversite([N + 1], N, 8) == 8


def test_tdr_simulation_causale():
    """La sortie au tour n ne dépend pas des entrées des tours suivants."""
    rng = np.random.default_rng(0)
    u = rng.uniform(size=30)
    a = T10.simuler(u, [12, 25], 10, 0.5, 0.8, 0.3, 0)
    u2 = u.copy()
    u2[20:] += 1.0
    b = T10.simuler(u2, [12, 25], 10, 0.5, 0.8, 0.3, 0)
    assert np.allclose(a[:20], b[:20])


def test_tdr_gain_de_boucle():
    # « somme » garde le gain de boucle du retard unique ; « quadratique » le dépasse dès K = 2
    for K in (1, 2, 3):
        assert np.isclose(T10.gains(0.95, K, "somme").sum(), 0.95)
    assert T10.gains(0.95, 2, "quadratique").sum() > 1.0
    assert np.isclose(T10.gains(0.95, 1, "quadratique").sum(), 0.95)
