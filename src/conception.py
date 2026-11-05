"""Conception d'un réservoir à retards par le calcul (parties 5, 11 et 12).

Quatre briques, sans entraîner de réservoir :

1. spectre(u, y, lmax)      importance de chaque retard, de chaque bloc de 3 retards
                            et de chaque produit u(t−a)·u(t−b), mesurée par ablation
                            dans un prédicteur ridge sur les entrées retardées ;
2. support_sauts(w, K)      S (contenant 1) qui minimise Σ_L w_L · h_S(L), où h_S(L)
                            est le nombre minimal de sauts pour écrire L avec S ;
3. paire_produit(spec)      la paire (a, b), a ≠ b, de produit la plus importante, si elle
                            porte une part suffisante de l'importance totale ;
4. enveloppe(spec) et recouvrement(E, P)
                            où la tâche a besoin de mémoire, et quelle part du profil
                            de mémoire P d'un réservoir tombe dans cette enveloppe.

Toutes les ablations se calculent à partir d'une seule matrice de Gram : retirer des
variables revient à extraire une sous-matrice.
"""
import itertools

import numpy as np
from scipy.linalg import cho_factor, cho_solve

LAMBDAS = (1e-8, 1e-6, 1e-4, 1e-2)


# spectre
def _variables(u, lmax):
    T = len(u)
    lag = np.zeros((T, lmax + 1))
    for k in range(lmax + 1):
        lag[k:, k] = u[:T - k]
    ia, ib = np.triu_indices(lmax + 1, k=0)
    return lag, ia, ib


def spectre(u, y, lmax=40, wash=200, frac_test=0.2, produits=True, bloc=3):
    """Prédicteur ridge sur [u(t−L) pour L = 0..lmax, produits u(t−a)u(t−b) pour
    a ≤ b (carrés compris), constante]. Ablations :
      I_lag[L]    hausse de NRMSE de validation quand u(t−L) et tous les produits
                  qui contiennent L sont retirés ;
      I_bloc[j]   idem pour les retards j·bloc .. j·bloc + bloc − 1 ;
      I_paire     hausse quand le seul produit (a, b) est retiré (les 40 premiers).
    La régularisation est choisie sur la validation (20 % finaux de l'entraînement),
    les importances sont mesurées sur la validation, l'erreur de test est rapportée."""
    u = np.asarray(u, np.float64)
    y = np.asarray(y, np.float64)
    T = len(u)
    lag, ia, ib = _variables(u, lmax)
    blocs = [lag]
    if produits:
        blocs.append(lag[:, ia] * lag[:, ib])
    X = np.hstack(blocs + [np.ones((T, 1))])
    P = X.shape[1]
    te0 = int(T * (1 - frac_test))
    v0 = wash + int(0.8 * (te0 - wash))
    sl_tr, sl_va, sl_te = slice(wash, v0), slice(v0, te0), slice(te0, T)

    def gram(sl):
        Xs, ys = X[sl], y[sl]
        return Xs.T @ Xs, Xs.T @ ys, float(ys @ ys), len(ys), float(ys.mean()), float(ys.std())

    A_tr, b_tr, _, _, _, _ = gram(sl_tr)
    A_va, b_va, yy_va, n_va, m_va, s_va = gram(sl_va)
    A_te, b_te, yy_te, n_te, m_te, s_te = gram(sl_te)

    def nrmse(w, A, b, yy, n, sd):
        mse = (w @ A @ w - 2 * w @ b + yy) / n
        return float(np.sqrt(max(mse, 0.0)) / sd)

    def resoudre(keep, lam):
        A = A_tr[np.ix_(keep, keep)] + lam * np.eye(len(keep))
        return cho_solve(cho_factor(A), b_tr[keep])

    tous = np.arange(P)
    best = None
    for lam in LAMBDAS:
        w = resoudre(tous, lam)
        v = nrmse(w, A_va, b_va, yy_va, n_va, s_va)
        if best is None or v < best[0]:
            best = (v, lam, w)
    val_plein, lam, w_plein = best
    test_plein = nrmse(w_plein, A_te, b_te, yy_te, n_te, s_te)

    def val_sans(retires):
        keep = np.setdiff1d(tous, retires)
        w = resoudre(keep, lam)
        return nrmse(w, A_va[np.ix_(keep, keep)], b_va[keep], yy_va, n_va, s_va)

    nl = lmax + 1
    prod_index = {(int(a), int(b)): nl + i for i, (a, b) in enumerate(zip(ia, ib))} if produits else {}

    def variables_de(lags):
        lags = set(lags)
        r = list(lags)
        r += [j for (a, b), j in prod_index.items() if a in lags or b in lags]
        return np.array(sorted(r), int)

    I_lag = [val_sans(variables_de([L])) - val_plein for L in range(nl)]
    I_bloc = [val_sans(variables_de(range(j, min(j + bloc, nl)))) - val_plein for j in range(0, nl, bloc)]
    I_paire = {}
    if produits:
        coef = np.abs(w_plein[nl:nl + len(ia)])
        for i in np.argsort(-coef)[:40]:
            a, b = int(ia[i]), int(ib[i])
            I_paire[f"{a},{b}"] = val_sans(np.array([prod_index[(a, b)]])) - val_plein
    return {"lmax": lmax, "lam": lam, "val_plein": val_plein, "test_plein": test_plein,
            "I_lag": I_lag, "I_bloc": I_bloc, "bloc": bloc, "I_paire": I_paire,
            "coef_lin": [float(c) for c in w_plein[:nl]]}


def plancher(u, y, lags, wash=200, frac_test=0.2):
    """Meilleure NRMSE de test d'un prédicteur ridge qui ne voit que les entrées
    u(t−L), L ∈ lags, et leurs produits deux à deux (carrés compris) : la limite
    de ce qu'un réservoir dont l'état ne contient que ces âges peut extraire."""
    u = np.asarray(u, np.float64)
    y = np.asarray(y, np.float64)
    T = len(u)
    lags = sorted(set(int(L) for L in lags))
    lag = np.zeros((T, len(lags)))
    for j, k in enumerate(lags):
        lag[k:, j] = u[:T - k]
    ia, ib = np.triu_indices(len(lags), k=0)
    X = np.hstack([lag, lag[:, ia] * lag[:, ib], np.ones((T, 1))])
    te0 = int(T * (1 - frac_test))
    v0 = wash + int(0.8 * (te0 - wash))
    best = None
    for lam in LAMBDAS:
        A = X[wash:v0].T @ X[wash:v0] + lam * np.eye(X.shape[1])
        w = cho_solve(cho_factor(A), X[wash:v0].T @ y[wash:v0])
        r = X[v0:te0] @ w - y[v0:te0]
        v = float(np.sqrt(np.mean(r ** 2)) / np.std(y[v0:te0]))
        if best is None or v < best[0]:
            best = (v, lam)
    A = X[wash:te0].T @ X[wash:te0] + best[1] * np.eye(X.shape[1])
    w = cho_solve(cho_factor(A), X[wash:te0].T @ y[wash:te0])
    r = X[te0:] @ w - y[te0:]
    return float(np.sqrt(np.mean(r ** 2)) / np.std(y[te0:]))


def atteignables(S, lmax):
    """⟨S⟩ ∩ [0, lmax]."""
    h = sauts(S, lmax)
    return [int(L) for L in range(lmax + 1) if np.isfinite(h[L])]


# support par les sauts
def sauts(S, lmax):
    """h[L] = nombre minimal de sauts pour écrire L avec S (inf si impossible)."""
    h = np.full(lmax + 1, np.inf)
    h[0] = 0
    for L in range(1, lmax + 1):
        for s in S:
            if L >= s and h[L - s] + 1 < h[L]:
                h[L] = h[L - s] + 1
    return h


def poids_retards(spec, seuil=0.05):
    """w_L = I_lag normalisé, en ne gardant que les retards au-dessus de seuil × max."""
    I = np.clip(np.array(spec["I_lag"], float), 0, None)
    if I.max() <= 0:
        return None
    w = np.where(I >= seuil * I.max(), I, 0.0)
    w[0] = 0.0                      # l'âge 0 est toujours atteignable (entrée directe)
    return w / w.sum() if w.sum() > 0 else None


def support_sauts(w, K, lmax=None, dmax=60):
    """S = {1} ∪ (K−1 retards dans 2..dmax) qui minimise Σ_L w_L h_S(L).
    Recherche exhaustive pour K ≤ 3, gloutonne au-delà (on ajoute un retard à la fois)."""
    lmax = lmax or len(w) - 1
    cibles = np.nonzero(w)[0]

    def cout(S):
        h = sauts(S, lmax)
        return float(np.sum(w[cibles] * h[cibles]))

    cands = range(2, dmax + 1)
    if K <= 3:
        best = min((cout((1,) + c), (1,) + c) for c in itertools.combinations(cands, K - 1))
        return list(best[1]), best[0]
    S = support_sauts(w, 3, lmax, dmax)[0]
    while len(S) < K:
        best = min((cout(tuple(S) + (d,)), d) for d in cands if d not in S)
        S = sorted(S + [best[1]])
    return S, cout(tuple(S))


def paire_produit(spec, part_min=0.1):
    """Paire (a, b) de produit la plus importante si son importance dépasse
    part_min × (somme des importances positives des retards). Sinon None."""
    paires = {k: v for k, v in spec["I_paire"].items() if k.split(",")[0] != k.split(",")[1]}
    if not paires:
        return None
    k, v = max(paires.items(), key=lambda kv: kv[1])
    total = float(np.clip(spec["I_lag"], 0, None).sum())
    if total <= 0 or v < part_min * total:
        return None
    a, b = (int(x) for x in k.split(","))
    return (a, b)


# enveloppe et profils
def enveloppe(spec):
    """Où la tâche a besoin de mémoire, pour L = 1..lmax : importance du bloc qui
    contient L, répartie sur ses retards, plus l'énergie lissée du coefficient
    linéaire u(t−L). Normalisée à somme 1. L'âge 0 est exclu."""
    lmax = spec["lmax"]
    bloc = spec["bloc"]
    E = np.zeros(lmax + 1)
    Ib = np.clip(np.array(spec["I_bloc"], float), 0, None)
    for j, v in enumerate(Ib):
        lo, hi = j * bloc, min(j * bloc + bloc, lmax + 1)
        E[lo:hi] += v / (hi - lo)
    c2 = np.array(spec["coef_lin"]) ** 2
    c2 = np.convolve(c2, np.ones(3) / 3, mode="same")
    if c2[1:].sum() > 0 and E[1:].sum() > 0:
        E += c2 * (E[1:].sum() / c2[1:].sum())
    E[0] = 0.0
    s = E.sum()
    return E / s if s > 0 else E


def recouvrement(E, P):
    """Σ_L min(E_L, P_L) sur L ≥ 1, E et P normalisés à somme 1 sur L ≥ 1."""
    E = np.asarray(E[1:], float)
    P = np.asarray(P[1:len(E) + 1], float)
    E = E / E.sum() if E.sum() > 0 else E
    P = P / P.sum() if P.sum() > 0 else P
    return float(np.minimum(E, P).sum())
