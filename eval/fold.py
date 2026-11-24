"""TM-score, CA-lDDT and GDT-TS of predicted against native chains (residues in order, fixed correspondence).

  python eval/fold.py --predicted DIR --native DIR [--out scores.csv]     .npz files with `backbone` [L, 3 (N, CA, C), 3], paired by name

A target with no prediction scores 0 and stays in the mean.
"""
import argparse
import csv
import os
import sys

import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fold_metrics import all_metrics


def ca(path): return np.load(path)["backbone"][:, 1].astype(np.float64)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predicted", required=True); ap.add_argument("--native", required=True); ap.add_argument("--out"); a = ap.parse_args()
    targets = sorted(f for f in os.listdir(a.native)
                     if f.endswith(".npz") and os.path.isfile(os.path.join(a.native, f)))
    if not targets:
        ap.error(f"no native .npz targets found in {a.native}")
    matched = {f for f in targets if os.path.isfile(os.path.join(a.predicted, f))}
    if not matched:
        ap.error(f"no matching .npz predictions in {a.predicted} for {len(targets)} native targets")
    missing = [f for f in targets if f not in matched]
    if missing:
        examples = ", ".join(missing[:5])
        print(f"Warning: missing predictions for {len(missing)} of {len(targets)} targets "
              f"(including {examples}). These targets score zero.", file=sys.stderr)
    rows = []
    for f in targets:
        n = ca(f"{a.native}/{f}"); p = f"{a.predicted}/{f}"
        s = all_metrics(ca(p), n) if f in matched else dict(rmsd=float("nan"), tm=0.0, lddt=0.0, gdt_ts=0.0)
        rows.append(dict(target=f[:-4], length=len(n), **s))
    if a.out:
        with open(a.out, "w", newline="") as fh: w = csv.DictWriter(fh, rows[0].keys()); w.writeheader(); w.writerows(rows)
    print(len(rows), "targets", {k: round(float(np.nanmean([r[k] for r in rows])), 3) for k in ("tm", "lddt", "gdt_ts")})


if __name__ == "__main__": main()
