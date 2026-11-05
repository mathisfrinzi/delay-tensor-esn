"""Moteur commun des parties 11 et 12 : réservoir à retards général, tâches, lectures.

Un seul simulateur couvre l'ESN, l'ESN à fuite, le DTS-ESN, le réservoir à retards
(tranches denses, permutées, diagonales ±1 ou identité) et le terme produit de la
partie 7 :

    z(t) = tanh( W_in·[u(t), 1] + Σ_{τ ∈ S} W_τ·x(t−τ) + W₂·(x(t−a) ⊙ x(t−b)) )
    x(t) = (1 − a_i)·x(t−1) + a_i·z(t)          (a_i = 1 : pas de fuite)

Conventions (protocole commun du rapport, introduction) :
  - W_in (d, N) uniforme sur [−0,5 ; 0,5], multipliée par l'échelle d'entrée ;
  - chaque tranche a le rayon spectral exact ρ/√K, K = nombre de tranches
    (le terme produit compte pour une tranche) ;
  - tirages dans l'ordre : W_in, tranches par τ croissant, W₂, fuites ;
  - état nul avant t = 0.

Comptes de budget : une matrice dense N×N coûte N² connexions ; permutation,
diagonale et identité coûtent 0. La lecture compte N + 1 paramètres par sortie.

Les données et les lectures ridge viennent de p7_common.py (mêmes splits et mêmes
grilles de régularisation que la partie 6 du rapport).

Variable d'environnement ESSAI=1 : tailles réduites pour vérifier que le code
tourne. Les résultats obtenus ainsi ne doivent jamais être rapportés.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
RES = ROOT / "results"
sys.path.insert(0, str(SRC))
os.environ.setdefault("P7_OUT", str(RES / "cache_p7"))
if os.environ.get("ESSAI") == "1":
    os.environ.setdefault("P7_SMOKE", "1")
import p7_common as P7  # noqa: E402

ESSAI = os.environ.get("ESSAI") == "1"
RIDGES = P7.RIDGES
MC_KMAX = 300   # 14.1 : capacité mesurée jusqu'à k = 300 (la mesure saturait à 60)
KINDS = ("dense", "perm", "diag", "ident")


# réservoir
def rayon_spectral(M):
    return float(max(np.abs(np.linalg.eigvals(M)).max(), 1e-12))


def construire(seed, taus, N, rho, kinds=None, d=2, echelle=1.0, fuite=1.0, dts_amin=None,
               paire=None):
    """Tire un réservoir. kinds : dict τ -> 'dense' | 'perm' | 'diag' | 'ident'.
    fuite : scalaire dans ]0 ; 1]. dts_amin : si donné, une fuite par neurone tirée
    log-uniformément sur [dts_amin ; 1] (DTS-ESN), fuite est alors ignorée.
    paire : (a, b) pour le terme W₂·(x(t−a) ⊙ x(t−b)), ou None."""
    taus = sorted(int(t) for t in taus)
    kinds = kinds or {}
    rng = np.random.default_rng(seed)
    Win = (rng.uniform(-0.5, 0.5, size=(d, N)) * echelle).astype(np.float32)
    K = len(taus) + (1 if paire else 0)
    sc = rho / np.sqrt(K)
    tranches = []
    for tau in taus:
        kind = kinds.get(tau, "dense")
        if kind == "dense":
            M = rng.normal(0, 1, (N, N))
            tranches.append((tau, "dense", (M * (sc / rayon_spectral(M))).astype(np.float32)))
        elif kind == "perm":
            tranches.append((tau, "perm", rng.permutation(N), np.float32(sc)))
        elif kind == "diag":
            tranches.append((tau, "diag", (rng.choice((-1.0, 1.0), size=N) * sc).astype(np.float32)))
        elif kind == "ident":
            tranches.append((tau, "diag", np.full(N, sc, np.float32)))
        else:
            raise ValueError(kind)
    W2 = None
    if paire:
        M = rng.normal(0, 1, (N, N))
        W2 = (M * (sc / rayon_spectral(M))).astype(np.float32)
    if dts_amin is not None:
        a = np.exp(rng.uniform(np.log(dts_amin), 0.0, size=N)).astype(np.float32)
    else:
        a = np.float32(fuite)
    return {"N": N, "taus": taus, "kinds": {t: kinds.get(t, "dense") for t in taus}, "Win": Win,
            "tranches": tranches, "paire": tuple(paire) if paire else None, "W2": W2, "fuite": a,
            "rho": rho, "echelle": echelle}


def connexions(res):
    """Connexions récurrentes qui coûtent : N² par matrice dense (W₂ compris)."""
    n_dense = sum(1 for tr in res["tranches"] if tr[1] == "dense") + (1 if res["W2"] is not None else 0)
    return n_dense * res["N"] ** 2


def couts(res):
    """Comptabilité mémoire (14.1) : le tampon circulaire garde les max(S)
    derniers états — O(max(S)·N), pas O(K·N) — plus les poids ; opérations par
    pas pour l'état (lecture exclue) : W_in·u (dN), tranches, produit W₂, tanh.
    Ordres de grandeur, pour la ligne « coût » des tableaux à budget égal."""
    N = res["N"]
    lags = list(res["taus"]) + ([1] if np.ndim(res["fuite"]) or res["fuite"] != 1.0 else [])
    if res["paire"]:
        lags += list(res["paire"])
    tmax = max(lags + [1])
    octets = 4 * N * tmax + 4 * res["Win"].size
    ops = res["Win"].size + N                     # W_in·u + tanh
    for tr in res["tranches"]:
        if tr[1] == "dense":
            octets += 4 * N * N
            ops += 2 * N * N
        elif tr[1] == "perm":
            octets += 8 * N + 4                   # indices + échelle
            ops += 2 * N
        else:                                     # diag / ident
            octets += 4 * N
            ops += 2 * N
    if res["W2"] is not None:
        octets += 4 * N * N
        ops += 3 * N * N                          # produit élémentaire + matrice
    return {"octets": int(octets), "ops_par_pas": int(ops)}


def N_pour_budget(budget, n_dense):
    return int(round(np.sqrt(budget / max(n_dense, 1))))


def _applique(tr, xp):
    if tr[1] == "dense":
        return xp @ tr[2].T
    if tr[1] == "perm":
        return tr[3] * xp[:, tr[2]]
    return tr[2] * xp


def simuler(res, U, final=False):
    """U : (T, d) pour une séquence, ou (T, B, d) pour B séquences en parallèle
    (MNIST : T = 28 lignes, d = 28 pixels). Renvoie les états (T, N), (T, B, N),
    ou seulement le dernier état (B, N) si final=True."""
    seq = U.ndim == 2
    if seq:
        U = U[:, None, :]
    T, B, _ = U.shape
    N = res["N"]
    lags = list(res["taus"]) + ([1] if np.ndim(res["fuite"]) or res["fuite"] != 1.0 else [])
    if res["paire"]:
        lags += list(res["paire"])
    tmax = max(lags)
    buf = np.zeros((tmax, B, N), np.float32)
    a = res["fuite"]
    sortie = None if final else np.empty((T, B, N), np.float32)
    x = np.zeros((B, N), np.float32)
    for t in range(T):
        drive = U[t] @ res["Win"]
        for tr in res["tranches"]:
            if t - tr[0] >= 0:
                drive += _applique(tr, buf[(t - tr[0]) % tmax])
        if res["paire"]:
            pa, pb = res["paire"]
            if t - max(pa, pb) >= 0:
                drive += (buf[(t - pa) % tmax] * buf[(t - pb) % tmax]) @ res["W2"].T
        z = np.tanh(drive)
        x = z if (np.ndim(a) == 0 and a == 1.0) else (1.0 - a) * x + a * z
        buf[t % tmax] = x
        if not final:
            sortie[t] = x
    if final:
        return x
    return sortie[:, 0, :] if seq else sortie


def profil_lineaire(res, lmax):
    """Énergie de la trace d'une impulsion d'entrée dans le réservoir linéarisé
    (tanh remplacé par l'identité, terme produit ignoré), pour L = 0..lmax :
        h(0) = W_in[0], h(L) = (1 − a)⊙h(L−1) + a⊙Σ_τ W_τ h(L−τ).
    Calcul exact sur le réservoir tiré, pas de champ moyen. Renvoie e (lmax+1,)."""
    N = res["N"]
    a = res["fuite"]
    h = np.zeros((lmax + 1, 1, N), np.float64)
    h[0, 0] = res["Win"][0]
    for L in range(1, lmax + 1):
        acc = np.zeros((1, N))
        for tr in res["tranches"]:
            if L - tr[0] >= 0:
                acc += _applique(tr, h[L - tr[0]].astype(np.float32))
        h[L] = (1.0 - a) * h[L - 1] + a * acc
    return (h[:, 0, :] ** 2).sum(axis=1)


# tâches
def parite(seed, L=2, D=10, T=5200):
    """XOR de la fenêtre u(t−D−L+1 .. t−D), u binaire i.i.d. (comme la partie 7)."""
    rng = np.random.default_rng(seed)
    u = rng.integers(0, 2, size=T).astype(np.float32)
    y = np.zeros(T, np.float64)
    cs = np.concatenate([[0.0], np.cumsum(u)])
    for t in range(D + L - 1, T):
        y[t] = (cs[t - D + 1] - cs[t - D + 1 - L]) % 2
    return np.stack([u, np.ones_like(u)], 1).astype(np.float32), y


def charger(nom, seed):
    """Renvoie un dict : U, y (ou u pour mc), type ('seq', 'mc', 'mnist'), splits."""
    T_seq = 1600 if ESSAI else 5200
    if nom in ("narma10", "narma30"):
        U, y, _ = P7.narma_x(int(nom[5:]), seed, T_seq)
        return {"U": U, "y": y, "type": "seq", "splits": _splits_seq(T_seq, 30)}
    if nom == "parite2":
        U, y = parite(seed, 2, 10, T_seq)
        return {"U": U, "y": y, "type": "seq", "splits": _splits_seq(T_seq, 30)}
    if nom in ("silverbox", "wh"):
        U, y = P7.load_phys(nom)
        T = len(y)
        nte = 2000 if ESSAI else 10000
        return {"U": U, "y": y, "type": "seq",
                "splits": {"wash": 500, "tr_end": T - nte, "te0": T - nte, "te1": T}}
    if nom == "mc":
        U, u = P7.mc_input(seed)
        return {"U": U, "u": u, "type": "mc"}
    if nom == "mnist":
        Xtr, ytr, Xte, yte = P7.get_mnist(5000, 1000)
        return {"Xtr": Xtr, "ytr": ytr, "Xte": Xte, "yte": yte, "type": "mnist"}
    raise ValueError(nom)


def _splits_seq(T, tmax):
    wash = max(200, 3 * tmax)
    nte = 300 if ESSAI else 1000
    return {"wash": wash, "tr_end": T - nte, "te0": T - nte, "te1": T}


def d_entree(nom):
    return 28 if nom == "mnist" else 2


def evaluer(res, data, val=False):
    """Erreur de test (ou de validation si val=True) d'un réservoir sur une tâche.
    Renvoie une erreur où plus bas = meilleur : NRMSE ; taux d'erreur pour MNIST ;
    300 − capacité totale (k = 1..300, 14.1 : la mesure saturait à 60) pour mc —
    pour mc, val=True mesure la capacité sur le segment de validation : le test
    n'est jamais un critère de réglage."""
    t = data["type"]
    if t == "mnist":
        Ftr = simuler(res, data["Xtr"].reshape(-1, 28, 28).transpose(1, 0, 2), final=True)
        Fte = simuler(res, data["Xte"].reshape(-1, 28, 28).transpose(1, 0, 2), final=True)
        if val:
            n = len(Ftr)
            v0 = int(0.8 * n)
            acc, _, _v = P7.mnist_readout(Ftr[:v0], data["ytr"][:v0], Ftr[v0:], data["ytr"][v0:])
        else:
            acc, _, _v = P7.mnist_readout(Ftr, data["ytr"], Fte, data["yte"])
        return 1.0 - acc
    F = simuler(res, data["U"])
    if t == "mc":
        return float(MC_KMAX) - float(mc_rapide(F, data["u"], MC_KMAX, val=val).sum())
    s = data["splits"]
    if val:
        v0 = s["wash"] + int(0.8 * (s["tr_end"] - s["wash"]))
        return P7.ridge_readout(F, data["y"], s["wash"], v0, v0, s["tr_end"])[0]
    return P7.ridge_readout(F, data["y"], s["wash"], s["tr_end"], s["te0"], s["te1"])[0]


def mc_rapide(F, u, kmax=60, val=False):
    """Même calcul que p7_common.mc_profile (splits, choix de λ par cible sur les
    20 % finaux du fit, r² du segment d'évaluation), mais les kmax cibles u(t−k)
    sont résolues ensemble : une factorisation par valeur de λ au lieu de kmax × 4.
    14.1 : val=True mesure la capacité sur le segment de validation [v0:tr_end]
    (fit sur [wash:v0], λ choisi sur les 20 % finaux de ce fit) — le test n'est
    jamais un critère de réglage."""
    from scipy.linalg import cho_factor, cho_solve
    T = len(u)
    wash = 200 if P7.SMOKE else 1000
    tr_end = wash + (800 if P7.SMOKE else 8000)
    v0 = wash + int(0.8 * (tr_end - wash))
    if val:
        m1, m2, m3 = wash + int(0.8 * (v0 - wash)), v0, tr_end
    else:
        m1, m2, m3 = v0, tr_end, T
    Fb = np.hstack([F, np.ones((T, 1), np.float32)]).astype(np.float64)
    uv = np.asarray(u, np.float64)
    Z = np.zeros((T, kmax))
    for k in range(1, kmax + 1):
        Z[k:, k - 1] = uv[:-k]
    A_a, A_c = Fb[wash:m1].T @ Fb[wash:m1], Fb[wash:m2].T @ Fb[wash:m2]
    b_a, b_c = Fb[wash:m1].T @ Z[wash:m1], Fb[wash:m2].T @ Z[wash:m2]
    I = np.eye(Fb.shape[1])
    crit = []
    for lam in RIDGES:
        W = cho_solve(cho_factor(A_a + lam * I), b_a)
        r = Fb[m1:m2] @ W - Z[m1:m2]
        crit.append(np.sqrt((r ** 2).mean(0)) / Z[m1:m2].std(0))
    choix = np.argmin(np.array(crit), axis=0)
    mc = np.zeros(kmax)
    for i, lam in enumerate(RIDGES):
        ks = np.nonzero(choix == i)[0]
        if not len(ks):
            continue
        W = cho_solve(cho_factor(A_c + lam * I), b_c[:, ks])
        P = Fb[m2:m3] @ W
        for j, k in enumerate(ks):
            if P[:, j].std() < 1e-12:
                continue
            r = np.corrcoef(P[:, j], Z[m2:m3, k])[0, 1]
            mc[k] = max(r * r, 0.0)
    return mc


# NG-RC
def ngrc_evaluer(data, D, produits=True, val=False):
    """NG-RC : variables = [1, u(t), …, u(t−D+1)] et, si produits, tous les
    produits deux à deux de ces retards ; lecture ridge (mêmes splits)."""
    if data["type"] not in ("seq",):
        return None
    u = data["U"][:, 0].astype(np.float64)
    T = len(u)
    lag = np.zeros((T, D))
    for k in range(D):
        lag[k:, k] = u[:T - k]
    feats = [lag]
    if produits:
        iu = np.triu_indices(D)
        feats.append(lag[:, iu[0]] * lag[:, iu[1]])
    F = np.hstack(feats).astype(np.float32)
    s = data["splits"]
    if val:
        v0 = s["wash"] + int(0.8 * (s["tr_end"] - s["wash"]))
        return P7.ridge_readout(F, data["y"], s["wash"], v0, v0, s["tr_end"])[0]
    return P7.ridge_readout(F, data["y"], s["wash"], s["tr_end"], s["te0"], s["te1"])[0]


# sauvegarde
def lire(chemin):
    if Path(chemin).exists():
        return json.load(open(chemin))
    return {"cells": {}, "meta": {}}


def ecrire(resultats, chemin):
    Path(chemin).parent.mkdir(parents=True, exist_ok=True)
    tmp = str(chemin) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(resultats, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    # Windows refuse parfois le remplacement quand un autre programme (antivirus, indexation, synchronisation)
    # tient le fichier ouvert à cet instant : on réessaie, puis on écrit directement en dernier recours.
    import time
    for essai in range(40):
        try:
            os.replace(tmp, chemin)
            return
        except PermissionError:
            time.sleep(0.25 * (1 + essai // 5))
    with open(chemin, "w") as f:
        json.dump(resultats, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    try:
        os.remove(tmp)
    except OSError:
        pass


def wilcoxon(a, b):
    return P7.wilcoxon_paired(a, b)
