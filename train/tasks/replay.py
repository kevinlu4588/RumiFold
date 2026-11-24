"""Task: fold an assembly and, in the same item, keep the base model's language by matching Qwen3-8B's own answers.

With probability `protein_visible` the text can attend to the folded assembly; otherwise it is text only.
"""
import numpy as np
import torch

from ..losses import fold_loss, replay_terms
from ..replay_data import Replay as Store
from .fold import Fold


class Replay:
    def __init__(self, cfg):
        self.cfg, self.store, self.fold = cfg, Store(cfg["data"], cfg.get("max_tokens", 2048)), Fold(cfg["fold"])

    def draw(self, rng, step):
        while True:
            turns = self.store.turns(rng.choice(self.store.train))
            if turns: break
        return {**self.fold.draw(rng, step), "turn": rng.choice(turns), "visible": rng.random() < self.cfg.get("protein_visible", .25)}

    def loss(self, model, item):
        t = item["turn"]; ids = torch.as_tensor(np.concatenate([t["input_ids"], t["response_ids"]]), dtype=torch.long, device=model.lm_head_weight.device)
        fold = model.fold_model(item["esm"], item["chain_id"], item["residue_index"])
        f, parts = fold_loss(fold.backbone, item, self.fold.cfg["loss"])
        r = replay_terms(model.text_logits(ids, fold if item["visible"] else None), t)
        parts = {**parts, **r}
        return f + self.cfg.get("kl", 2.5) * r["kl"] + self.cfg.get("ce", .1) * r["ce"], {k: float(v.detach()) for k, v in parts.items()}
