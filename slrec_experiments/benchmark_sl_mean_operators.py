"""Compare averaging operators on SL(8) at controlled neighbor spread.

Operators, all in float64:
  A_amb : det-retracted ambient mean        M=sum w_i G_i; M/det(M)^(1/8)
  A_tan : normalized tangent (algebra) mean exp(sum w_i X_i / sum w_i)
  A_k1  : one bi-invariant fixed-point step m0=A_tan; m0 exp(sum wbar_i log(m0^-1 G_i))
  A_k*  : converged bi-invariant mean (fixed-point iteration, exact logm)

Metric: F(m) = sum_i wbar_i ||logm(m^-1 G_i)||_F^2   (their Schatten-2 semidistance objective,
one-sided; exact scipy logm). Lower is better. Also: failure stats of the ambient sum,
Frobenius blowup after det normalization, and Gregory-K12 log error vs distance.
"""
import numpy as np
import torch
from scipy.linalg import logm, expm

torch.manual_seed(0)
np.random.seed(0)
N = 8
TRIALS = 60
NEIGH = 12   # typical CF degree scale


def trace_free(X):
    return X - np.trace(X) / N * np.eye(N)


def sample_group(sigma, k):
    Xs = [trace_free(np.random.randn(N, N) * sigma) for _ in range(k)]
    Gs = [expm(X) for X in Xs]
    return Xs, Gs


def obj(m, Gs, w):
    minv = np.linalg.inv(m)
    tot = 0.0
    for wi, G in zip(w, Gs):
        L = logm(minv @ G)
        if np.iscomplexobj(L):
            if np.abs(L.imag).max() > 1e-8:
                return np.inf  # left principal-log domain
            L = L.real
        tot += wi * np.linalg.norm(L, "fro") ** 2
    return tot


def karcher(Gs, w, m0, iters=50, tol=1e-12):
    m = m0.copy()
    for _ in range(iters):
        minv = np.linalg.inv(m)
        xi = np.zeros((N, N))
        ok = True
        for wi, G in zip(w, Gs):
            L = logm(minv @ G)
            if np.iscomplexobj(L):
                if np.abs(L.imag).max() > 1e-6:
                    ok = False
                    break
                L = L.real
            xi += wi * L
        if not ok:
            return m, False
        m = m @ expm(xi)
        if np.linalg.norm(xi, "fro") < tol:
            break
    return m, True


def gregory_log_np(A, terms=12, jitter=1e-7):
    I = np.eye(N)
    Z = np.linalg.solve(A + (1 + jitter) * I, A - I)
    Z2 = Z @ Z
    P = Z.copy()
    S = Z.copy()
    for k in range(1, terms):
        P = P @ Z2
        S = S + P / (2 * k + 1)
    return 2 * S


print(f"{'sigma':>6} | {'detM<=0%':>8} {'|logdetM|med':>12} {'||Gamb||med':>11} "
      f"{'F(amb)':>10} {'F(tan)':>10} {'F(k1)':>10} {'F(k*)':>10} | "
      f"{'amb/k*':>7} {'tan/k*':>7} {'k1/k*':>7}")
for sigma in [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2]:
    stats = dict(neg=0, logdet=[], gnorm=[], fa=[], ft=[], f1=[], fs=[])
    for _ in range(TRIALS):
        Xs, Gs = sample_group(sigma, NEIGH)
        wraw = np.abs(np.random.rand(NEIGH)) * 0.4  # sym-norm-like sub-unit weights
        wbar = wraw / wraw.sum()
        M = sum(wi * G for wi, G in zip(wraw, Gs))
        det = np.linalg.det(M)
        if det <= 0:
            stats['neg'] += 1
            continue
        Gamb = M / det ** (1.0 / N)
        stats['logdet'].append(abs(np.log(abs(det))))
        stats['gnorm'].append(np.linalg.norm(Gamb, 'fro'))
        Gtan = expm(sum(wi * X for wi, X in zip(wbar, Xs)))
        m1, _ = karcher(Gs, wbar, Gtan, iters=1)
        ms, ok = karcher(Gs, wbar, Gtan, iters=80)
        fa, ft = obj(Gamb, Gs, wbar), obj(Gtan, Gs, wbar)
        f1, fs = obj(m1, Gs, wbar), obj(ms, Gs, wbar)
        if not np.isfinite([fa, ft, f1, fs]).all():
            continue
        stats['fa'].append(fa); stats['ft'].append(ft)
        stats['f1'].append(f1); stats['fs'].append(fs)
    med = lambda a: float(np.median(a)) if a else float('nan')
    fa, ft, f1, fs = med(stats['fa']), med(stats['ft']), med(stats['f1']), med(stats['fs'])
    print(f"{sigma:6.2f} | {100*stats['neg']/TRIALS:7.1f}% {med(stats['logdet']):12.3e} "
          f"{med(stats['gnorm']):11.3f} {fa:10.4f} {ft:10.4f} {f1:10.4f} {fs:10.4f} | "
          f"{fa/fs:7.3f} {ft/fs:7.3f} {f1/fs:7.3f}")

print()
print("Gregory K=12 truncated log accuracy vs exact (relative Frobenius error), pairs at distance d:")
print(f"{'target d':>8} {'median relerr':>14} {'max relerr':>12} {'dist underest (approx/exact)':>28}")
for target in [0.5, 1.0, 2.0, 3.0, 4.0, 6.0]:
    rel, ratio = [], []
    for _ in range(40):
        X = trace_free(np.random.randn(N, N))
        X = X / np.linalg.norm(X, 'fro') * target
        A = expm(X)  # relative matrix at geodesic-ish distance ||X||
        exact = logm(A)
        if np.iscomplexobj(exact):
            if np.abs(exact.imag).max() > 1e-8:
                continue
            exact = exact.real
        approx = gregory_log_np(A)
        rel.append(np.linalg.norm(approx - exact, 'fro') / np.linalg.norm(exact, 'fro'))
        ratio.append(np.linalg.norm(approx, 'fro') / np.linalg.norm(exact, 'fro'))
    print(f"{target:8.1f} {np.median(rel):14.3e} {np.max(rel):12.3e} {np.median(ratio):28.4f}")
