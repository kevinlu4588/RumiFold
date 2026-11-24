"""Supplementary readout heads on the second folding pass: per-residue function labels and Gene Ontology terms.

Neither is used for text answers. Parameter names match the released checkpoints, so `init_from` restores them.
"""
import math

import torch
from torch import nn
from torch.nn.functional import binary_cross_entropy_with_logits as bce, cross_entropy, gelu, layer_norm


def layers(fold):
    """Every hidden state of the second pass: [layers + 1, residues, width]."""
    return torch.stack([h[0] for h in fold.hidden])


class ResidueHeads(nn.Module):
    """One MLP per label type over its own learned mix of layers, joined with the ESM-C features."""

    def __init__(self, width, vocab, n_layers, esm_width=2560, hidden=512):
        super().__init__()
        self.sizes = dict(active=1, binding=len(vocab["binding"]), mutation=1, ptm=len(vocab["ptm"]), topology=len(vocab["topology"]))
        self.mix = nn.ParameterDict({k: nn.Parameter(torch.zeros(n_layers)) for k in self.sizes})
        self.esm_norm = nn.LayerNorm(esm_width)
        self.mlp = nn.ModuleDict({k: nn.Sequential(nn.Linear(width + esm_width, hidden), nn.GELU(), nn.Linear(hidden, n)) for k, n in self.sizes.items()})
        for m in self.mlp.values(): nn.init.zeros_(m[-1].weight); nn.init.zeros_(m[-1].bias)

    def forward(self, hidden, esm):
        h = layer_norm(hidden.float(), hidden.shape[-1:]); e = self.esm_norm(esm.float())
        return {k: self.mlp[k](torch.cat([torch.einsum("n,nld->ld", torch.softmax(self.mix[k], 0), h), e], -1)) for k in self.sizes}


def residue_targets(labels, length, sizes, device):
    """Label sets (0-based positions) -> {type: (target, mask)} for the types this protein carries. Absent means unknown."""
    out = {}
    if "active" in labels:
        t = torch.zeros(length, 1); t[labels["active"], 0] = 1; out["active"] = (t, torch.ones(length, 1))
    for k in ("binding", "ptm"):
        if k in labels:
            t, m = torch.zeros(length, sizes[k]), torch.zeros(length, sizes[k])
            for c, pos in labels[k].items(): t[pos, int(c)] = 1; m[:, int(c)] = 1
            out[k] = (t, m)
    if "mutation" in labels:
        t, m = torch.zeros(length, 1), torch.zeros(length, 1); eff, none = labels["mutation"]["effect"], labels["mutation"]["none"]
        t[eff, 0] = 1; m[eff + none, 0] = 1; out["mutation"] = (t, m)
    if "topology" in labels:
        t = torch.full((length,), -100, dtype=torch.long)
        for a, b, state in labels["topology"]: t[a:b + 1] = state
        out["topology"] = (t, (t >= 0).float())
    return {k: (t.to(device), m.to(device)) for k, (t, m) in out.items()}


def residue_loss(logits, targets):
    """Mean over the label types present; None when there is nothing to score."""
    parts = []
    for k, (t, m) in targets.items():
        if m.sum() == 0: continue
        if k == "topology": parts.append(cross_entropy(logits[k].float(), t, ignore_index=-100))
        else: parts.append((bce(logits[k].float(), t, reduction="none") * m).sum() / m.sum())
    return torch.stack(parts).mean() if parts else None


class GOHead(nn.Module):
    """Label-wise attention over residues plus a mean-pooled MLP, with an optional organism shift. ESM-C is not an input."""

    def __init__(self, width, prior, n_layers, organisms=0, dim=512, hidden=1024, chunk=4096):
        super().__init__()
        n = len(prior); self.dim, self.chunk = dim, chunk
        self.depth = nn.Parameter(torch.zeros(n_layers)); self.qwen_in, self.fuse_ln = nn.Linear(width, dim), nn.LayerNorm(dim)
        self.pool_mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, n)); nn.init.zeros_(self.pool_mlp[-1].weight)
        with torch.no_grad(): self.pool_mlp[-1].bias.copy_(torch.tensor([math.log(p / (1 - p)) for p in prior]))
        self.label_q = nn.Parameter(torch.randn(n, dim) * dim ** -.5)
        self.label_v, self.label_b = nn.Parameter(torch.zeros(n, dim)), nn.Parameter(torch.zeros(n))
        if organisms:
            self.org_emb, self.org_out = nn.Embedding(organisms, dim), nn.Linear(dim, n, bias=False); nn.init.zeros_(self.org_out.weight)

    def forward(self, hidden, organism=None):
        h = layer_norm(hidden.float(), hidden.shape[-1:])
        h = self.fuse_ln(gelu(self.qwen_in(torch.einsum("n,nld->ld", torch.softmax(self.depth, 0), h))))
        out = self.pool_mlp(h.mean(0))
        if hasattr(self, "org_emb"): out = out + self.org_out(self.org_emb(torch.as_tensor(organism, device=h.device)))
        parts = []
        for s in range(0, len(self.label_q), self.chunk):
            att = torch.softmax(self.label_q[s:s + self.chunk] @ h.T / self.dim ** .5, -1)
            parts.append(((att @ h) * self.label_v[s:s + self.chunk]).sum(-1))
        return out + torch.cat(parts) + self.label_b


def go_loss(logits, target, mask):
    return (bce(logits.float(), target, reduction="none") * mask).sum() / mask.sum() if mask.sum() else logits.sum() * 0.


def attach(model, name, head, checkpoint=None):
    """Add a head to the model before the optimizer is built, restoring its weights when the checkpoint has them."""
    if not hasattr(model, name): setattr(model, name, head.to(model.lm_head_weight.device).float())
    if checkpoint:
        from safetensors import safe_open
        with safe_open(f"{checkpoint}/model.safetensors", framework="pt") as f:
            state = {k[len(name) + 1:]: f.get_tensor(k) for k in f.keys() if k.startswith(name + ".")}
        if state: getattr(model, name).load_state_dict(state)
    return getattr(model, name)
