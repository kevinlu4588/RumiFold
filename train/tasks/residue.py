"""Task (supplementary): per-residue function labels from the fold, plus a fold loss when an AlphaFold model exists.

Reads RumiFold_Functional: pool/, residue_labels/ and features/. Labels that are absent are unknown, never negative.
"""
import json
import os

import torch
from safetensors import safe_open

from ..data import backbone
from ..heads import ResidueHeads, attach, layers, residue_loss, residue_targets
from ..losses import fold_loss


class Residue:
    def __init__(self, cfg):
        self.cfg, self.root = cfg, cfg["data"]; root = self.root; limit = cfg.get("max_length", 1024)
        self.index = {r["sequence_sha256"]: r for r in map(json.loads, open(f"{root}/features/index.jsonl"))}
        st = f"{root}/residue_labels/afdb/structures.jsonl"
        self.structure = {r["sequence_sha256"]: r for r in map(json.loads, open(st))} if os.path.exists(st) else {}
        self.vocab = json.load(open(f"{root}/residue_labels/vocab.json"))
        self.rows = [r for r in map(json.loads, open(f"{root}/residue_labels/train.jsonl"))
                     if r["labels"] and r["length"] <= limit and r["sequence_sha256"] in self.index]

    def setup(self, model):
        q = model.fold_model.qwen.config
        attach(model, "residue_heads", ResidueHeads(q.hidden_size, self.vocab, q.num_hidden_layers + 1), self.cfg.get("init_from"))

    def draw(self, rng, step):
        r = rng.choice(self.rows); e = self.index[r["sequence_sha256"]]
        with safe_open(f"{self.root}/features/{e['shard']}", framework="pt") as f: x = f.get_slice("embeddings")[e["residue_offset"]:e["residue_end"]]
        n = len(x); item = dict(esm=x, chain_id=torch.zeros(n, dtype=torch.long), residue_index=torch.arange(n), sequences=[r["sequence"]], labels=r["labels"])
        st = self.structure.get(r["sequence_sha256"])
        if st:
            t = torch.from_numpy(backbone(self.root, st).copy())
            if tuple(t.shape) == (n, 3, 3) and int(torch.isfinite(t).all(-1).all(-1).sum()) >= 16: item["target"] = t
        return item

    def loss(self, model, item):
        fold = model.fold_model(item["esm"], item["chain_id"], item["residue_index"]); hidden = layers(fold)
        r = residue_loss(model.residue_heads(hidden, item["esm"]), residue_targets(item["labels"], len(item["esm"]), model.residue_heads.sizes, hidden.device))
        total = fold.backbone.sum() * 0.; parts = {}
        if r is not None: total = total + self.cfg.get("residue_weight", 1.) * r; parts["residue"] = float(r.detach())
        if "target" in item: f, _ = fold_loss(fold.backbone, item, self.cfg["loss"]); total = total + f; parts["fold"] = float(f.detach())
        return total, parts
