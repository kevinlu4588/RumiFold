"""Fold a bundled protein and ask about it. Needs only Qwen3-8B and one RumiFold checkpoint (no ESM-C):

  python scripts/demo.py [--protein lumazine_synthase|lysozyme] [--model rumifold-156k] [--out demo.pdb]
"""
import argparse
import json
import os
import sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from rumifold import RumiFold, load_features, write_pdb

QUESTIONS = ["Which residues form the active site of this protein?",
             "First list this protein's Gene Ontology molecular function terms (most specific first) and EC number if any, "
             "then describe its function in one or two sentences, as a Swiss-Prot FUNCTION entry would.",
             "What is the capital of France?"]

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--protein", default="lumazine_synthase", choices=["lumazine_synthase", "lysozyme"]); p.add_argument("--model", default="rumifold-156k")
p.add_argument("--qwen", default="Qwen/Qwen3-8B"); p.add_argument("--out", default="demo.pdb")
a = p.parse_args()
info = json.load(open(f"{ROOT}/examples/data/proteins.json"))[a.protein]
model = RumiFold.from_pretrained(a.model, qwen=a.qwen)
fold = model.fold(load_features(f"{ROOT}/examples/data/{a.protein}.safetensors"))
write_pdb(a.out, fold.backbone, info["sequence"])
print(f"{a.protein} ({info['uniprot']}, {info['length']} residues): backbone written to {a.out}")
for q in QUESTIONS: print(f"\nQ: {q}\nA: {model.ask(q, fold)}")
