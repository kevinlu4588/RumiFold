"""Task: fold an assembly from ESM-C features."""
from ..data import ClusterSampler, Embeddings, example, read_manifest
from ..losses import fold_loss


class Fold:
    def __init__(self, cfg):
        self.cfg, self.root = cfg, cfg["data"]
        self.embeddings = Embeddings(self.root)
        limit = cfg.get("max_residues")
        self.sampler = ClusterSampler(r for r in read_manifest(self.root) if not limit or r["total_length"] <= limit)

    def draw(self, rng, step):
        c = self.cfg; ramp = min(1., step / c["complex_ramp_steps"])
        fraction = c["complex_fraction_start"] + (c["complex_fraction_end"] - c["complex_fraction_start"]) * ramp
        row = self.sampler.draw(rng, fraction, c.get("heterodimer_fraction", .5))
        return example(self.root, row, self.embeddings, reverse=rng.random() < .5)

    def loss(self, model, item):
        fold = model.fold_model(item["esm"], item["chain_id"], item["residue_index"])
        loss, parts = fold_loss(fold.backbone, item, self.cfg["loss"])
        return loss, {k: float(v.detach()) for k, v in parts.items()}
