"""Retrieval quality of cheap GEMM-shaped surrogates vs the exact SL scorer.

Exact score: one-sided ||log(U^-1 I)||_F (Gregory K=24, float64 = ground truth).
Surrogates (all computable as one big GEMM over flattened 64-dim tables):
  frob : ||G_u - G_i||_F^2   = ||G_u||^2 + ||G_i||^2 - 2<G_u, G_i>
  trace: -tr(G_u^-1 G_i)     = -<G_u^-T, G_i>   (user inverses precomputed once)
Also a mid-cost reference:
  cayley: ||Z||_F with Z from one solve (no polynomial)  [not GEMM, ~2-3x cheaper than exact]

Metric: recall of the exact top-50 within the surrogate top-C, per user.
"""
import torch, sys

from slrec_experiments.geometry import to_sl, matrix_log_gregory

torch.manual_seed(0)
N_ITEMS, N_USERS, N = 8192, 64, 8
TOPK = 50

for clip, std in ((0.75, 0.05), (1.5, 0.1)):
    users = to_sl(torch.randn(N_USERS, N, N, dtype=torch.float64) * std * 8, max_frobenius=clip)
    items = to_sl(torch.randn(N_ITEMS, N, N, dtype=torch.float64) * std * 8, max_frobenius=clip)
    # ground truth exact distances
    rel = torch.linalg.solve(users[:, None], items[None, :])
    exact = torch.linalg.matrix_norm(matrix_log_gregory(rel, terms=24), ord="fro", dim=(-2, -1))
    exact_top = exact.topk(TOPK, dim=1, largest=False).indices

    uflat, iflat = users.reshape(N_USERS, -1), items.reshape(N_ITEMS, -1)
    frob = (uflat.square().sum(1, keepdim=True) + iflat.square().sum(1)[None, :]
            - 2 * uflat @ iflat.T)
    uinv = torch.linalg.inv(users).transpose(-2, -1).reshape(N_USERS, -1)
    trace = -(uinv @ iflat.T)
    eye = torch.eye(N, dtype=torch.float64)
    cayley = torch.linalg.matrix_norm(
        torch.linalg.solve(rel + eye, rel - eye), ord="fro", dim=(-2, -1))

    print(f"\ncoord_clip={clip} (typical spread ||X||<= {clip})  exact d: "
          f"median {exact.median():.3f} p99 {exact.quantile(0.99):.3f}")
    print(f"{'surrogate':>8} | " + " ".join(f"C={c:<5}" for c in (256, 512, 1024, 2048)))
    for name, s in (("frob", frob), ("trace", trace), ("cayley", cayley)):
        row = []
        for C in (256, 512, 1024, 2048):
            cand = s.topk(C, dim=1, largest=False).indices
            hits = (exact_top[:, :, None] == cand[:, None, :]).any(-1).float().mean()
            row.append(f"{hits*100:5.1f}%")
        print(f"{name:>8} | " + " ".join(row))
