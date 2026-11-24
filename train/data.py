"""Reading RumiFold_Data: manifest parts, packed embeddings, PDB tars (unpacked) and AlphaFold backbone shards."""
import glob
import gzip
import json
from functools import lru_cache

import numpy as np
import torch
from safetensors import safe_open


def read_manifest(root, split="train"):
    for part in sorted(glob.glob(f"{root}/manifest/part-*.jsonl.gz")):
        for line in gzip.open(part, "rt"):
            row = json.loads(line)
            if row["split"] == split: yield row


class Embeddings:
    def __init__(self, root):
        self.root, self.index = root, {}
        for part in sorted(glob.glob(f"{root}/embeddings/index/part-*.jsonl")):
            for line in open(part):
                r = json.loads(line); self.index[r["sequence_sha256"]] = (r["shard"], r["residue_offset"], r["residue_end"])

    def __call__(self, sha):
        shard, a, b = self.index[sha]
        with safe_open(f"{self.root}/embeddings/{shard}", framework="pt") as f: return f.get_slice("embeddings")[a:b]


@lru_cache(maxsize=4)
def _shard(path):
    with np.load(path) as z: return z["backbone"], z["plddt"] if "plddt" in z else None


def backbone(root, row):
    path = f"{root}/{row['coordinate_path']}"
    if "coordinate_start" not in row:
        with np.load(path) as z: return z["backbone"]
    bb, plddt = _shard(path); a, b = row["coordinate_start"], row["coordinate_end"]; bb = bb[a:b].copy()
    if "coordinate_plddt_min" in row: bb[plddt[a:b] < row["coordinate_plddt_min"]] = np.nan
    return bb


def example(root, row, embeddings, reverse=False):
    bb = backbone(root, row); offset = np.cumsum([0] + row["lengths"])
    order = list(range(len(row["chains"])))[::-1 if reverse else 1]
    feats, target, chain, index, seqs = [], [], [], [], []
    for new, old in enumerate(order):
        c = row["chains"][old]; n = len(c["sequence"]); e = embeddings(c["sequence_sha256"])
        if tuple(e.shape) != (n, 2560): raise ValueError(f"{row['id']}: embedding does not match the sequence")
        feats.append(e); target.append(torch.from_numpy(bb[offset[old]:offset[old + 1]].copy()))
        chain.append(torch.full((n,), new)); index.append(torch.arange(n)); seqs.append(c["sequence"])
    return dict(id=row["id"], esm=torch.cat(feats), target=torch.cat(target), chain_id=torch.cat(chain),
                residue_index=torch.cat(index), sequences=seqs)


class ClusterSampler:
    """Draw a kind (monomer, homodimer, heterodimer), then a sequence cluster, then an assembly, each uniformly."""

    def __init__(self, rows):
        self.groups = {}
        for r in rows:
            c = r["chains"]
            kind = "monomer" if len(c) == 1 else "homodimer" if c[0]["sequence"] == c[1]["sequence"] else "heterodimer"
            self.groups.setdefault(kind, {}).setdefault(r["sequence_pair_cluster"], []).append(r)
        self.keys = {k: sorted(g) for k, g in self.groups.items()}

    def draw(self, rng, complex_fraction, heterodimer_fraction=.5):
        kind = "monomer"
        if rng.random() < complex_fraction: kind = "heterodimer" if rng.random() < heterodimer_fraction else "homodimer"
        return rng.choice(self.groups[kind][rng.choice(self.keys[kind])])
