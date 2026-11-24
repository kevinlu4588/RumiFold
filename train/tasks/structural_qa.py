"""Task (supplementary): one question per item about the model's own predicted structure, from RumiFold_Functional/structural_qa.

The paper's recipe also generated questions on the fly, asked several per prompt and used further question families; this task
covers the published question bank (seven families). Structures are read from RumiFold_Data.
"""
import glob
import itertools
import json
import random

import torch

from ..data import Embeddings, example, read_manifest
from ..losses import fold_loss
from ..structural_qa import Phrasing, addresses, prepare, qa_loss


class StructuralQa:
    def __init__(self, cfg):
        self.cfg, self.root = cfg, cfg["data"]; self.embeddings = Embeddings(self.root); limit = cfg.get("max_residues", 512)
        self.phrasing = Phrasing(f"{cfg['questions']}/phrasings.json"); want = {}
        for f in sorted(glob.glob(f"{cfg['questions']}/train/shard-*.jsonl")):
            for line in open(f):
                r = json.loads(line); want.setdefault(r["protein_id"], []).append((r["family"], r["params"]))
        self.items = {}
        for r in read_manifest(self.root):
            if r["id"] in want and r["total_length"] <= limit and all(c["sequence_sha256"] in self.embeddings.index for c in r["chains"]): self.items[r["id"]] = (r, want[r["id"]])
        self.ids = sorted(self.items)

    def draw(self, rng, step):
        while True:
            row, questions = self.items[rng.choice(self.ids)]; family, params = rng.choice(questions)
            try: names = addresses(row)
            except ValueError: continue
            item = example(self.root, row, self.embeddings); item.update(question=dict(family=family, params=params), names=names, rng_seed=rng.getrandbits(32))
            return item

    def loss(self, model, item):
        fold = model.fold_model(item["esm"], item["chain_id"], item["residue_index"])
        q, a, spans, inline = self.phrasing.render(item["question"], item["names"], fold.backbone[:, 1].detach().float().cpu().numpy(), random.Random(item["rng_seed"]))
        ids, at, n, kind, soft, digit = prepare(model.tok, q, a, spans, inline)
        logits = model.text_logits(torch.tensor(ids, device=model.lm_head_weight.device), fold, at)
        l = qa_loss(logits, ids, n, kind, soft, digit, self.cfg.get("other_weight", .3)); parts = {"qa": float(l.detach())}; total = self.cfg.get("qa_weight", 1.) * l
        f, _ = fold_loss(fold.backbone, item, self.cfg["loss"]); total = total + self.cfg.get("fold_weight", 1.) * f; parts["fold"] = float(f.detach())
        return total, parts
