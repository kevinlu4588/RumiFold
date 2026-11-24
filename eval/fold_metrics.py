"""CA-only structure metrics with known residue correspondence: TM-score, GDT-TS, CA-lDDT and RMSD.

Inputs are [L, 3] CA coordinates; the native may have NaN rows (unresolved). TM-score and GDT-TS use a seeded,
iteratively refined superposition search, normalised by the number of resolved native residues.
"""
import numpy as np


def kabsch(p, t):
    pc, tc = p.mean(0), t.mean(0); u, _, vh = np.linalg.svd((p - pc).T @ (t - tc))
    R = u @ np.diag([1, 1, np.sign(np.linalg.det(u @ vh))]) @ vh
    return R, tc - pc @ R


def superpose_search(p, t):
    """(TM-score, GDT-TS): the best over seed fragments, each refined with that quantity's own distance cutoff."""
    L = len(t); d0 = max(0.5, 1.24 * max(L - 15, 1) ** (1 / 3) - 1.8) if L > 21 else 0.5
    cuts = {"tm": d0, 1: 1.0, 2: 2.0, 4: 4.0, 8: 8.0}; best = {k: 0.0 for k in cuts}
    lens, f = [], L
    while f >= 4: lens.append(f); f //= 2
    for fl in lens or [L]:
        step = max(1, fl // 2) if fl < L else 1
        for s in range(0, L - fl + 1, step):
            for cut0 in cuts.values():
                sel = np.arange(s, s + fl)
                for _ in range(20):
                    if len(sel) < 3: break
                    R, tr = kabsch(p[sel], t[sel]); d = np.linalg.norm(p @ R + tr - t, axis=1)
                    best["tm"] = max(best["tm"], float((1 / (1 + (d / d0) ** 2)).sum() / L))
                    for c in (1, 2, 4, 8): best[c] = max(best[c], float((d <= c).sum() / L))
                    cut = cut0; new = np.nonzero(d < cut)[0]
                    while len(new) < 3 and cut < 50: cut += 0.5; new = np.nonzero(d < cut)[0]
                    if np.array_equal(new, sel): break
                    sel = new
    return best["tm"], float(np.mean([best[c] for c in (1, 2, 4, 8)]))


def lddt_ca(p, t, radius=15.0):
    dp, dt = (np.linalg.norm(x[:, None] - x[None], axis=-1) for x in (p, t))
    local = (dt < radius) & ~np.eye(len(t), dtype=bool)
    if not local.any(): return float("nan")
    err = np.abs(dp - dt)[local]
    return float(np.mean([(err < c).mean() for c in (0.5, 1.0, 2.0, 4.0)]))


def all_metrics(pred, native):
    """A prediction that covers only part of the resolved residues is scaled by its coverage (TM-score and GDT-TS)."""
    res = np.isfinite(native).all(1); ok = res & np.isfinite(pred).all(1)
    if ok.sum() < 3: return dict(tm=0.0, gdt_ts=0.0, lddt=0.0, rmsd=float("nan"))
    p, t = pred[ok].astype(np.float64), native[ok].astype(np.float64); frac = ok.sum() / res.sum()
    tm, gdt = superpose_search(p, t); R, tr = kabsch(p, t)
    return dict(tm=tm * frac, gdt_ts=gdt * frac, lddt=lddt_ca(p, t), rmsd=float(np.sqrt(((p @ R + tr - t) ** 2).sum(1).mean())))
