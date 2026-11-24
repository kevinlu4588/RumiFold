"""Task: describe a protein's function (Prot2Text), answering from the fold. No structure loss on these items."""
import bisect
import collections
import glob
import itertools
import json

import torch
from safetensors import safe_open

from ..text import answer_loss

SYSTEM = ("You are a scientific assistant specialized in protein function predictions. Given the sequence embeddings of a "
          "protein, describe its function clearly and concisely in professional language.")


class Prot2Text:
    def __init__(self, cfg):
        self.cfg, self.shard = cfg, {}
        for m in sorted(glob.glob(f"{cfg['data']}/{cfg.get('features', 'features_final')}/shard_*.json")):
            for sha in json.load(open(m))["lengths"]: self.shard[sha] = m[:-5] + ".safetensors"
        rows = map(json.loads, open(f"{cfg['data']}/splits/train.jsonl")); limit = cfg.get("max_length", 1024)
        self.rows = [r for r in rows if len(r["sequence"]) <= limit and (r.get("function") or "").strip() and r["sequence_sha256"] in self.shard]
        # draw weight = copies(description) ** -balance: 0 is uniform over proteins, 1 uniform over descriptions
        copies = collections.Counter(r["function"].strip() for r in self.rows)
        self.cum = list(itertools.accumulate(copies[r["function"].strip()] ** -cfg.get("balance", .5) for r in self.rows))

    def draw(self, rng, step):
        r = self.rows[bisect.bisect_left(self.cum, rng.random() * self.cum[-1])]
        with safe_open(self.shard[r["sequence_sha256"]], framework="pt") as f: x = f.get_tensor(r["sequence_sha256"])
        n = len(x); return dict(esm=x, chain_id=torch.zeros(n, dtype=torch.long), residue_index=torch.arange(n), function=r["function"].strip())

    def loss(self, model, item):
        ids = model.tok.encode(item["function"], add_special_tokens=False)
        answer = item["function"] if len(ids) <= self.cfg.get("max_answer", 256) else model.tok.decode(ids[:self.cfg.get("max_answer", 256)])
        fold = model.fold_model(item["esm"], item["chain_id"], item["residue_index"])
        l = answer_loss(model, fold, [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "sequence embeddings: ..."}], answer)
        return l, {"text": float(l.detach())}
