"""Why does SL8LHGCN lose to LHGCN?  Test the loss-scale-transfer hypothesis.

Both models optimise the IDENTICAL objective inherited from the released
HGCF/lGCN code,

    relu(d_pos^2 - d_neg^2 + loss_margin).sum(),   loss_margin = 0.1,

but the two geometries live on completely different squared-distance scales:

  LHGCN : sqdist = clamp(K * arcosh(theta)^2, max=50), K = 1/curve = 2.
          Embeddings are re-projected onto the hyperboloid every layer but are
          never radius-capped, and LorentzBatchNorm's learnable gamma lets the
          model choose its own dispersion.  d^2 spans [0, 50].
  SL8   : d = ||log(A^-1 B)||_F, coord_clip = 0.75, score_scale fixed at 1.0
          and learnable_score_scale disabled by the faithful-hinge check.

Part A measures the realised d^2 distributions of both scorers, so the margin
can be read as a fraction of each geometry's own scale.
Part B runs a controlled synthetic bipartite CF task (identical propagation,
loss, optimiser, evaluation) and sweeps SL's margin.

Run:  .venv-slrec/bin/python slrec_experiments/diagnose_margin_scale_transfer.py
"""

import math

import torch
import torch.nn.functional as F

torch.manual_seed(0)
N = 8  # SL(8)


# ---------------------------------------------------------------- geometry --
def trace_free(m):
    n = m.shape[-1]
    tr = m.diagonal(dim1=-2, dim2=-1).sum(-1)
    return m - (tr / n)[..., None, None] * torch.eye(n, dtype=m.dtype)


def cap_frobenius(m, cap):
    if cap is None or cap <= 0:
        return m
    nrm = torch.linalg.matrix_norm(m, ord="fro", dim=(-2, -1), keepdim=True)
    return m * (cap / nrm.clamp_min(1e-12)).clamp(max=1.0)


def to_sl(raw, cap):
    return torch.matrix_exp(cap_frobenius(trace_free(raw), cap))


def gregory12(z):
    identity = torch.eye(z.shape[-1], dtype=z.dtype)
    z2 = z @ z
    z4 = z2 @ z2
    z6 = z4 @ z2
    b0 = identity + z2 / 3 + z4 / 5
    b1 = identity / 7 + z2 / 9 + z4 / 11
    b2 = identity / 13 + z2 / 15 + z4 / 17
    b3 = identity / 19 + z2 / 21 + z4 / 23
    return 2.0 * (z @ (b0 + z6 @ (b1 + z6 @ (b2 + z6 @ b3))))


def sl_dist(left, right):
    z = torch.linalg.solve(right + (1.0 + 1e-7) * left, right - left)
    return torch.linalg.matrix_norm(gregory12(z), ord="fro", dim=(-2, -1))


def lorentz_expmap0(u, c):
    K = 1.0 / c
    sqrt_k = math.sqrt(K)
    x = u[..., 1:]
    xn = x.norm(dim=-1, keepdim=True).clamp_min(1e-15)
    theta = xn / sqrt_k
    return torch.cat(
        (sqrt_k * torch.cosh(theta), sqrt_k * torch.sinh(theta) * x / xn), dim=-1
    )


def lorentz_project(x, c):
    K = 1.0 / c
    sp = x[..., 1:]
    t = torch.sqrt((K + sp.square().sum(-1, keepdim=True)).clamp_min(1e-15))
    return torch.cat((t, sp), dim=-1)


def lorentz_sqdist(x, y, c):
    """Hyperboloid.sqdist, including the repository's max=50 clamp."""
    K = 1.0 / c
    prod = -(x[..., 0:1] * y[..., 0:1].transpose(-2, -1)) + x[..., 1:] @ y[
        ..., 1:
    ].transpose(-2, -1)
    theta = (-prod / K).clamp_min(1.0 + 1e-7)
    return (K * torch.acosh(theta).square()).clamp(max=50.0)


# ------------------------------------------------------- Part A: dynamics ---
print("=" * 74)
print("PART A  realised squared-distance range of the two scorers")
print("=" * 74)
print(
    f"{'model':28s} {'d^2 p50':>9} {'d^2 p95':>9} {'d^2 max':>9} "
    f"{'margin/p95':>11} {'hinge active':>13}"
)

MARGIN = 0.1
n_probe = 4000

for cap, label in ((0.75, "SL8 coord_clip=0.75"), (3.0, "SL8 coord_clip=3.0")):
    raw = torch.randn(n_probe, N, N)
    raw = raw / torch.linalg.matrix_norm(
        raw, ord="fro", dim=(-2, -1), keepdim=True
    ) * cap
    a = to_sl(raw[: n_probe // 2], cap)
    b = to_sl(raw[n_probe // 2 :], cap)
    d2 = sl_dist(a, b).square()
    active = float((d2 - d2.flip(0) + MARGIN > 0).float().mean())
    print(
        f"{label:28s} {d2.median():9.3f} {d2.quantile(.95):9.3f} {d2.max():9.3f} "
        f"{MARGIN / float(d2.quantile(.95)):11.3f} {active:12.1%}"
    )

for spatial, label in (
    (0.5, "LHGCN c=0.5 |x_sp|=0.5"),
    (2.0, "LHGCN c=0.5 |x_sp|=2.0"),
    (6.0, "LHGCN c=0.5 |x_sp|=6.0"),
):
    u = torch.randn(n_probe, 64)
    u = u / u[..., 1:].norm(dim=-1, keepdim=True) * spatial
    x = lorentz_project(lorentz_expmap0(u, 0.5), 0.5)
    d2 = lorentz_sqdist(x[: n_probe // 2], x[n_probe // 2 :], 0.5).diagonal()
    active = float((d2 - d2.flip(0) + MARGIN > 0).float().mean())
    print(
        f"{label:28s} {d2.median():9.3f} {d2.quantile(.95):9.3f} {d2.max():9.3f} "
        f"{MARGIN / float(d2.quantile(.95)):11.3f} {active:12.1%}"
    )


# ------------------------------------------- Part B: controlled CF task -----
print()
print("=" * 74)
print("PART B  synthetic bipartite CF, matched protocol, hinge margin sweep")
print("=" * 74)

NU, NI, LATENT = 600, 900, 16
g = torch.Generator().manual_seed(3)
zu = F.normalize(torch.randn(NU, LATENT, generator=g), dim=-1)
zi = F.normalize(torch.randn(NI, LATENT, generator=g), dim=-1)
pop = torch.rand(NI, generator=g).pow(2.0)
aff = zu @ zi.T + 1.2 * pop[None, :]
K_POS = 12
pos = aff.topk(K_POS, dim=1).indices
perm = torch.randperm(K_POS, generator=g)
train_pos, test_pos = pos[:, perm[:8]], pos[:, perm[8:]]

rows = torch.arange(NU).repeat_interleave(8)
cols = train_pos.reshape(-1) + NU
src = torch.cat((rows, cols))
dst = torch.cat((cols, rows))
deg = torch.bincount(src, minlength=NU + NI).float().clamp_min(1)
val = deg[src].pow(-0.5) * deg[dst].pow(-0.5)
ADJ = torch.sparse_coo_tensor(
    torch.stack((src, dst)), val, (NU + NI, NU + NI)
).coalesce()
TRAIN_MASK = torch.zeros(NU, NI, dtype=torch.bool)
TRAIN_MASK.scatter_(1, train_pos, True)


def recall_at_10(scores):
    scores = scores.masked_fill(TRAIN_MASK, float("-inf"))
    top = scores.topk(10, dim=1).indices
    hit = (top[:, :, None] == test_pos[:, None, :]).any(-1).sum(-1).float()
    return float((hit / test_pos.shape[1]).mean())


def train_model(kind, margin=0.1, cap=0.75, epochs=260, lr=0.005, layers=2):
    torch.manual_seed(0)
    if kind == "lhgcn":
        emb = torch.nn.Parameter(0.1 * (2 * torch.rand(NU + NI, 64) - 1))
        gamma = torch.nn.Parameter(torch.ones(1))
        params = [emb, gamma]
    else:
        bound = math.sqrt(6.0 / (NU + NI + 64))
        emb = torch.nn.Parameter(bound * (2 * torch.rand(NU + NI, 64) - 1))
        params = [emb]
    opt = torch.optim.Adam(params, lr=lr)

    def forward():
        if kind == "lhgcn":
            x = lorentz_project(lorentz_expmap0(emb, 0.5), 0.5)
            for _ in range(layers):
                x = lorentz_project(torch.sparse.mm(ADJ, x), 0.5)
                sp = x[..., 1:]
                scaled = gamma * sp / (sp.norm(dim=-1, keepdim=True).mean() + 1e-7)
                x = lorentz_project(torch.cat((x[..., :1], scaled), -1), 0.5)
            return x[:NU], x[NU:]
        gm = to_sl(emb.reshape(-1, N, N), cap)
        for _ in range(layers):
            flat = torch.sparse.mm(ADJ, gm.reshape(NU + NI, -1)).reshape(-1, N, N)
            sign, logabs = torch.linalg.slogdet(flat)
            colmul = torch.ones_like(flat[..., 0, :])
            colmul[..., -1] = torch.where(sign.lt(0), -1.0, 1.0)
            oriented = flat * colmul.unsqueeze(-2)
            ok = sign.ne(0) & torch.isfinite(logabs)
            safe = torch.where(ok, logabs, torch.zeros_like(logabs))
            gm = oriented * (-safe / N).exp()[:, None, None]
            if not bool(ok.all()):
                gm = torch.where(
                    ok[:, None, None], gm, to_sl(torch.nan_to_num(flat), 1.0)
                )
        return gm[:NU], gm[NU:]

    for _ in range(epochs):
        u = torch.randint(0, NU, (2048,))
        p = train_pos[u, torch.randint(0, 8, (2048,))]
        n = torch.randint(0, NI, (2048,))
        user, item = forward()
        if kind == "lhgcn":
            dp = lorentz_sqdist(user[u], item[p], 0.5).diagonal()
            dn = lorentz_sqdist(user[u], item[n], 0.5).diagonal()
        else:
            dp = sl_dist(user[u], item[p]).square()
            dn = sl_dist(user[u], item[n]).square()
        loss = F.relu(dp - dn + margin).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        user, item = forward()
        if kind == "lhgcn":
            score = -lorentz_sqdist(user, item, 0.5)
            spread = float(
                lorentz_sqdist(user[:200], item[:200], 0.5).diagonal().quantile(0.95)
            )
        else:
            score = -sl_dist(user[:, None], item[None, :]).square()
            spread = float((-score[:200].diagonal()).quantile(0.95))
        active = float((dp - dn + margin > 0).float().mean())
    return recall_at_10(score), spread, active


print(f"{'setup':38s} {'Recall@10':>10} {'d^2 p95':>9} {'hinge active':>13}")
r, d, s = train_model("lhgcn", margin=0.1)
print(f"{'LHGCN (Lorentz, BN gamma, margin .1)':38s} {r:10.4f} {d:9.3f} {s:12.1%}")
for cap in (0.75, 3.0):
    r, d, s = train_model("sl", margin=0.1, cap=cap)
    print(f"{f'SL8 clip={cap} margin=0.1 (current)':38s} {r:10.4f} {d:9.3f} {s:12.1%}")
for margin in (0.02, 0.005, 0.001):
    r, d, s = train_model("sl", margin=margin, cap=0.75)
    print(f"{f'SL8 clip=0.75 margin={margin}':38s} {r:10.4f} {d:9.3f} {s:12.1%}")
r, d, s = train_model("sl", margin=0.005, cap=3.0)
print(f"{'SL8 clip=3.0 margin=0.005':38s} {r:10.4f} {d:9.3f} {s:12.1%}")
