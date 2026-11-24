"""Fold one protein from its ESM-C features: python scripts/fold.py --model DIR --features f.safetensors --sequence SEQ --out out.pdb"""
import argparse
import os
import sys

import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rumifold import RumiFold, load_features, write_pdb

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--model", required=True, help="checkpoint folder, or a released name such as rumifold-156k")
p.add_argument("--features", required=True); p.add_argument("--offset", type=int, default=0); p.add_argument("--end", type=int)
p.add_argument("--sequence"); p.add_argument("--qwen", default="Qwen/Qwen3-8B"); p.add_argument("--out", default="fold.pdb", help="write a .npz backbone for evaluation or a PDB for visualization")
p.add_argument("--chain-lengths", type=int, nargs="+", help="residues per chain, for complexes")
a = p.parse_args()
model = RumiFold.from_pretrained(a.model, qwen=a.qwen)
fold = model.fold(load_features(a.features, a.offset, a.end), a.chain_lengths)
if a.out.endswith(".npz"):
    np.savez_compressed(a.out, backbone=fold.backbone.detach().float().cpu().numpy())
else:
    write_pdb(a.out, fold.backbone, a.sequence, a.chain_lengths)
print("wrote", a.out)
