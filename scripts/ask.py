"""Ask about a protein: python scripts/ask.py --model DIR --features f.safetensors "Which residues form the active site?" """
import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rumifold import RumiFold, load_features

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("question"); p.add_argument("--model", required=True, help="checkpoint folder, or a released name such as rumifold-156k")
p.add_argument("--features", help="ESM-C features; leave out for a plain language question")
p.add_argument("--offset", type=int, default=0); p.add_argument("--end", type=int)
p.add_argument("--qwen", default="Qwen/Qwen3-8B"); p.add_argument("--max-new-tokens", type=int, default=256)
a = p.parse_args()
model = RumiFold.from_pretrained(a.model, qwen=a.qwen)
fold = model.fold(load_features(a.features, a.offset, a.end)) if a.features else None
print(model.ask(a.question, fold, a.max_new_tokens))
