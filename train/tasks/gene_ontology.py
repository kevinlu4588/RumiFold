"""Task: name a protein's Gene Ontology terms, one question per labelled aspect, plus a fold loss when an AlphaFold model exists.

Supplementary, off by default: "head": true trains the Gene Ontology head next to the text answers, and "residue_labels": true
trains the per-residue heads on proteins of the training split that have labels. The paper's observation items are not included.
"""
import hashlib
import json
import os

import torch
from safetensors import safe_open

from ..data import backbone
from ..go import Ontology
from ..heads import GOHead, ResidueHeads, attach, go_loss, layers, residue_loss, residue_targets
from ..losses import fold_loss
from ..text import answer_loss

ASPECTS = {"mf": "molecular function", "bp": "biological process", "cc": "cellular component"}
QUESTIONS = ("Which Gene Ontology {} terms describe this protein? List the most specific ones, separated by semicolons.",
             "Predict the GO {} annotations of this protein. Give the most specific terms, separated by semicolons.",
             "What {} GO terms does this protein have? Answer with the most specific term names, separated by semicolons.")


def organism(r):
    o = (r.get("organism") or "").strip(); i = o.find(" ("); j = o.find(")", i) if i >= 0 else -1
    return o[:j + 1] if j >= 0 else o


class GeneOntology:
    def __init__(self, cfg):
        self.cfg, root = cfg, cfg["data"]; self.root = root; sha = lambda s: hashlib.sha256(s.encode()).hexdigest()
        self.onto = Ontology(f"{root}/ontology/go-basic.obo")
        self.ia = {t: float(w) for t, w in (l.split()[:2] for l in open(f"{root}/ontology/IA.txt") if l.strip())}
        self.index = {r["sequence_sha256"]: r for r in map(json.loads, open(f"{root}/features/index.jsonl"))}
        self.structure = {r["sequence_sha256"]: r for r in map(json.loads, open(f"{root}/afdb/structures.jsonl"))}
        rows = [json.loads(l) for l in open(f"{root}/proteins.jsonl")]; test = {sha(r["seq"]) for r in rows if r["split"] == "test"}
        self.rows = []
        for r in rows:
            s = sha(r["seq"])
            if r["split"] != "train" or s in test or len(r["seq"]) > cfg.get("max_length", 2048) or s not in self.index: continue      # no exact test sequences
            go = {a: [t for t in r["labels"]["go_" + a] if t in self.onto.space] for a in ASPECTS}
            if any(go.values()): self.rows.append(dict(sha=s, seq=r["seq"], go=go, organism=organism(r), taxid=r.get("taxid")))
        self.labels = {}
        if cfg.get("residue_labels"):
            self.labels = {r["sequence_sha256"]: r["labels"] for r in map(json.loads, open(f"{root}/residue_labels/train.jsonl")) if r["labels"]}
        if cfg.get("head"):
            v = json.load(open(f"{root}/go_vocab.json")); self.vocab = v["terms"]; self.column = {t["go"]: t["index"] for t in self.vocab}
            self.slot = {t: i for i, t in enumerate(v["organism_slots"]["taxa"])}

    def setup(self, model):
        """Add the supplementary heads before the optimizer is built."""
        q, init = model.fold_model.qwen.config, self.cfg.get("init_from")
        if self.cfg.get("head"):
            labelled = {a: sum(1 for r in self.rows if r["go"][a]) for a in ASPECTS}; count = {}
            for r in self.rows:
                for a in ASPECTS:
                    for t in r["go"][a]: count[t] = count.get(t, 0) + 1
            prior = [min(max(count.get(t["go"], 0) / max(labelled[t["aspect"]], 1), 1e-4), .5) for t in self.vocab]
            attach(model, "gov2_head", GOHead(q.hidden_size, prior, q.num_hidden_layers + 1, organisms=len(self.slot) + 1), init)
        if self.cfg.get("residue_labels"):
            vocab = json.load(open(self.cfg["residue_vocab"])); attach(model, "residue_heads", ResidueHeads(q.hidden_size, vocab, q.num_hidden_layers + 1), init)

    def targets(self, go, device):
        """(target, mask) over the head's terms; an aspect without labels is unknown."""
        t, m = torch.zeros(len(self.vocab)), torch.zeros(len(self.vocab)); have = {a for a in ASPECTS if go[a]}
        for x in self.vocab:
            if x["aspect"] in have: m[x["index"]] = 1
        for a in have:
            for g in go[a]:
                if g in self.column: t[self.column[g]] = 1
        return t.to(device), m.to(device)

    def answer(self, terms):
        leaves = sorted(self.onto.leaves(terms), key=lambda t: (-self.ia.get(t, 0.), t))[:self.cfg.get("max_terms", 100000)]
        return "; ".join(self.onto.name[t] for t in leaves) + "."

    def draw(self, rng, step):
        r = rng.choice(self.rows); e = self.index[r["sha"]]; qa = []
        for a, name in ASPECTS.items():
            if not r["go"][a]: continue
            q = rng.choice(QUESTIONS).format(name)
            if r["organism"] and rng.random() >= self.cfg.get("organism_dropout", .1): q = f"Organism: {r['organism']}.\n{q}"
            qa.append((q, self.answer(r["go"][a])))
        with safe_open(f"{self.root}/features/{e['shard'].split('/')[-1]}", framework="pt") as f: x = f.get_slice("embeddings")[e["residue_offset"]:e["residue_end"]]
        n = len(x); item = dict(esm=x, chain_id=torch.zeros(n, dtype=torch.long), residue_index=torch.arange(n), sequences=[r["seq"]], qa=qa, go=r["go"], organism=self.slot.get(r["taxid"], len(self.slot)) if self.cfg.get("head") else None, labels=self.labels.get(r["sha"]))
        st = self.structure.get(r["sha"])
        if st:
            base = self.root if os.path.exists(f"{self.root}/{st['coordinate_path']}") else os.path.dirname(os.path.abspath(self.root))     # some point into RumiFold_Functional
            t = torch.from_numpy(backbone(base, st).copy())
            if tuple(t.shape) == (n, 3, 3) and int(torch.isfinite(t).all(-1).all(-1).sum()) >= 16: item["target"] = t
        return item

    def loss(self, model, item):
        fold = model.fold_model(item["esm"], item["chain_id"], item["residue_index"])
        text = sum(answer_loss(model, fold, q, a) for q, a in item["qa"]); parts = {"text": float(text.detach())}; total = text
        if "target" in item: f, _ = fold_loss(fold.backbone, item, self.cfg["loss"]); total = total + self.cfg.get("fold_weight", 1.) * f; parts["fold"] = float(f.detach())
        hidden = layers(fold) if self.cfg.get("head") or item["labels"] else None
        if self.cfg.get("head"):
            h = go_loss(model.gov2_head(hidden, item["organism"]), *self.targets(item["go"], hidden.device)); total = total + self.cfg.get("head_weight", 3.) * h; parts["head"] = float(h.detach())
        if item["labels"]:
            r = residue_loss(model.residue_heads(hidden, item["esm"]), residue_targets(item["labels"], len(item["esm"]), model.residue_heads.sizes, hidden.device))
            if r is not None: total = total + self.cfg.get("residue_weight", 1.) * r; parts["residue"] = float(r.detach())
        return total, parts
