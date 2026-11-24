"""Prot2Text predictions, as scored by eval/prot2text.py.

  python eval/predict_prot2text.py --model rumifold-prot2text --split test.jsonl --features FEATURE_DIR --out pred.json

FEATURE_DIR holds the shards and one .json manifest per shard, as in the RumiFold_Prot2Text dataset.
"""
import argparse
import glob
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from safetensors import safe_open
from rumifold import RumiFold

SYSTEM = ("You are a scientific assistant specialized in protein function predictions. Given the sequence embeddings of a "
          "protein, describe its function clearly and concisely in professional language.")


def feature_index(folder):
    """sequence hash -> shard; every protein is its own tensor in the shard, keyed by the hash."""
    out = {}
    for manifest in sorted(glob.glob(f"{folder}/shard_*.json")):
        for sha in json.load(open(manifest))["lengths"]: out[sha] = manifest[:-5] + ".safetensors"
    return out


def reference(tok, text, limit=256):
    """The Prot2Text-V2 convention: score against the label truncated to 256 tokens."""
    return tok.decode(tok.encode(text, add_special_tokens=False)[:limit])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for x in ("model", "split", "features", "out"): ap.add_argument("--" + x, required=True)
    ap.add_argument("--qwen", default="Qwen/Qwen3-8B"); ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--max-length", type=int, default=1024, help="longer proteins are truncated")
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--shards", type=int, default=1); ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    index = feature_index(a.features); rows = [json.loads(l) for l in open(a.split)][a.shard::a.shards][:a.limit]
    model = RumiFold.from_pretrained(a.model, qwen=a.qwen); out = {}
    for k, r in enumerate(rows):
        with safe_open(index[r["sequence_sha256"]], framework="pt") as f: fold = model.fold(f.get_tensor(r["sequence_sha256"])[:a.max_length])
        reply = model.ask([{"role": "system", "content": SYSTEM}, {"role": "user", "content": "sequence embeddings: ..."}], fold, a.max_new_tokens, addresses=False)
        out[r.get("id") or r["sequence_sha256"]] = {"pred": reply, "true": reference(model.tok, r["function"])}
        if k % 50 == 0: print(f"{k}/{len(rows)}", flush=True)
    json.dump(out, open(a.out, "w"))


if __name__ == "__main__": main()
