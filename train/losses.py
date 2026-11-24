"""Folding loss: frame error within and between chains, peptide bonds, and backbone clashes (no per-chain alignment)."""
import torch
from torch.nn.functional import smooth_l1_loss

from rumifold.model import rotation


def masked_mean(x, mask):
    return torch.where(mask, x, torch.zeros_like(x)).sum() / mask.sum().clamp_min(1)


def fold_terms(pred, target, chain_id, residue_index):
    """pred, target: [L, 3 (N, CA, C), 3]; NaN in target marks unresolved atoms."""
    with torch.autocast(device_type=pred.device.type, enabled=False):
        p = pred.float(); valid = torch.isfinite(target).all(-1); t = torch.nan_to_num(target.float())
        frames = lambda bb: rotation(torch.cat((bb[:, 1] - bb[:, 2], bb[:, 0] - bb[:, 1]), -1))
        rp, rt = frames(p), frames(t)
        area = torch.linalg.cross(t[:, 1] - t[:, 2], t[:, 0] - t[:, 1], dim=-1).norm(dim=-1)
        frame_ok = valid.all(-1) & (area > 1e-4)
        local = lambda r, x: torch.einsum("iab,ija->ijb", r, x.reshape(-1, 3)[None] - x[:, 1, None])
        err = ((local(rp, p) - local(rt, t)).square().sum(-1) + 1e-8).sqrt() - 1e-4
        err = .9 * err.clamp(max=10) / 10 + .1 * err / 10
        same = chain_id[:, None] == chain_id.repeat_interleave(3)[None]
        mask = frame_ok[:, None] & valid.reshape(-1)[None]
        intra, inter = masked_mean(err, mask & same), masked_mean(err, mask & ~same)
        # no bonds across unresolved residues or annotated breaks
        connected = (chain_id[1:] == chain_id[:-1]) & (residue_index[1:] == residue_index[:-1] + 1)
        native = (t[:-1, 2] - t[1:, 0]).norm(dim=-1)
        bond = connected & valid[:-1, 2] & valid[1:, 0] & (native > .9) & (native < 1.8)
        peptide = masked_mean(smooth_l1_loss((p[:-1, 2] - p[1:, 0]).norm(dim=-1), native, reduction="none"), bond)
        atom = p.reshape(-1, 3); dist = torch.cdist(atom, atom)
        residue = torch.arange(len(p), device=p.device).repeat_interleave(3); ci = chain_id.repeat_interleave(3)
        excluded = (ci[:, None] == ci[None]) & ((residue[:, None] - residue[None]).abs() <= 1)
        clash = (torch.relu(1.5 - dist).square() * torch.triu(~excluded, diagonal=1)).sum() / max(1, len(atom))
        return dict(intra=intra, inter=inter, peptide=peptide, clash=clash)


def fold_loss(pred, item, weights):
    """The better of the two chain assignments for a homodimer."""
    targets, seq = [item["target"]], item["sequences"]
    if len(seq) == 2 and seq[0] == seq[1]:
        n = len(seq[0]); targets.append(torch.cat((item["target"][n:], item["target"][:n])))
    out = []
    for t in targets:
        parts = fold_terms(pred, t, item["chain_id"], item["residue_index"])
        out.append((sum(parts[k] * weights[k] for k in weights), parts))
    return min(out, key=lambda x: float(x[0].detach()))


def replay_terms(logits, turn):
    """Response-position KL to the teacher (its top-20 plus one bucket for the rest) and cross-entropy on its tokens."""
    device = logits.device; as_t = lambda k, t: torch.as_tensor(turn[k], dtype=t, device=device)
    y, ids, lp, tail = as_t("response_ids", torch.long), as_t("topk_ids", torch.long), as_t("topk_logprobs", torch.float32), as_t("tail_mass", torch.float32)
    x = logits[len(turn["input_ids"]) - 1:len(turn["input_ids"]) - 1 + len(y)].float()
    norm = torch.logsumexp(x, -1); top = x.gather(-1, ids) - norm[:, None]
    rest = torch.logsumexp(x.scatter(-1, ids, float("-inf")), -1) - norm
    tail_term = torch.where(tail > 0, tail * (tail.clamp_min(torch.finfo(tail.dtype).tiny).log() - rest), torch.zeros_like(tail))
    return dict(kl=((lp.exp() * (lp - top)).sum(-1) + tail_term).mean(), ce=torch.nn.functional.cross_entropy(x, y))
