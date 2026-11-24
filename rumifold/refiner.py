"""Frame refiner and confidence head of the folding-only RumiFold-Struct checkpoints.

Two lightweight geometry-attention blocks take the language model's residue states and the frames from the linear
structure head, and update both. The confidence head predicts per-residue CA-lDDT from the refined states.
"""
import torch
from torch import nn


def quaternion_rotation(v):
    q = torch.cat((torch.ones_like(v[..., :1]), v), -1); q = q / q.norm(dim=-1, keepdim=True)
    w, x, y, z = q.unbind(-1)
    return torch.stack((1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
                        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)), -1).reshape(*v.shape[:-1], 3, 3)


class GeometryBlock(nn.Module):
    def __init__(self, width=256, heads=4, dim=32):
        super().__init__()
        self.heads, self.dim = heads, dim
        self.norm, self.qkv, self.pair = nn.LayerNorm(width), nn.Linear(width, 3 * heads * dim), nn.Linear(width, 16)
        self.bias = nn.Sequential(nn.Linear(62, 32), nn.SiLU(), nn.Linear(32, heads))
        self.out = nn.Linear(heads * (dim + 4), width)
        self.transition = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width * 2), nn.SiLU(), nn.Linear(width * 2, width))
        self.update = nn.Linear(width, 6)
        self.register_buffer("centers", torch.linspace(0, 32, 16), persistent=False)

    def forward(self, h, R, t, chain, pos):
        n = len(h); z = self.norm(h)
        q, k, v = self.qkv(z).reshape(n, 3, self.heads, self.dim).unbind(1)
        delta = t[None] - t[:, None]; distance = (delta.square().sum(-1) + 1e-8).sqrt()
        local = torch.einsum("iab,ija->ijb", R, delta) / distance[..., None].clamp_min(1.)
        relative = torch.einsum("iab,jac->ijbc", R, R).reshape(n, n, 9)
        rbf = torch.exp(-((distance[..., None] - self.centers) / 2.2).square())
        same = chain[:, None] == chain[None]
        sep = ((pos[None] - pos[:, None]).float() / 32).clamp(-1, 1) * same
        p = self.pair(z)
        pair = torch.cat((p[:, None].expand(n, n, 16), p[None].expand(n, n, 16), local, relative, rbf, same[..., None].float(), sep[..., None]), -1)
        attn = (torch.einsum("ihd,jhd->hij", q, k) / (self.dim ** .5) + self.bias(pair).permute(2, 0, 1)).softmax(-1)
        values = torch.einsum("hij,jhd->ihd", attn, v)
        context = torch.einsum("hij,ijc->ihc", attn, torch.cat((local, (distance / 32).clamp(max=4)[..., None]), -1))
        h = h + self.out(torch.cat((values, context), -1).reshape(n, -1)); h = h + self.transition(h)
        update = self.update(h)
        t = t + torch.einsum("iab,ib->ia", R, 3 * update[:, :3].tanh())
        return h, R @ quaternion_rotation(.25 * update[:, 3:].tanh()), t


class Refiner(nn.Module):
    def __init__(self, width=4096):
        super().__init__()
        self.project = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 256))
        self.blocks = nn.ModuleList(GeometryBlock() for _ in range(2))

    def forward(self, z, R, t, chain, pos):
        h = self.project(z.float())
        for block in self.blocks: h, R, t = block(h, R, t, chain, pos)
        return R, t, h


class ConfidenceHead(nn.Module):
    def __init__(self, width=256, bins=50):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 128), nn.ReLU(), nn.Linear(128, bins))
        self.register_buffer("centers", (torch.arange(bins).float() + .5) / bins, persistent=False)

    def forward(self, h): return (self.net(h.float()).softmax(-1) * self.centers).sum(-1) * 100
