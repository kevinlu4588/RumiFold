"""Gene Ontology predictions for the GO-GPT v2 test set, as scored by eval/go.py.

  python eval/predict_go.py --model rumifold-go --data RumiFold_GO --features RumiFold_GO/features --out pred.json [--shard 0 --shards 1]

FEATURE_DIR holds index.jsonl (sequence_sha256, shard, residue_offset, residue_end) and the shards it names.
"""
import argparse
import hashlib
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rumifold import RumiFold, load_features

ASPECTS = {"mf": "molecular function", "bp": "biological process", "cc": "cellular component"}
QUESTION = "Which Gene Ontology {} terms describe this protein? List the most specific ones, separated by semicolons."


def organism(r):
    """'Danio rerio (Zebrafish) (Brachydanio rerio)' -> 'Danio rerio (Zebrafish)'."""
    o = (r.get("organism") or "").strip(); i = o.find(" ("); j = o.find(")", i) if i >= 0 else -1
    return o[:j + 1] if j >= 0 else o


def ontology(path):
    names, spaces, parents, synonyms, obsolete = {}, {}, {}, {}, set(); cur = None
    for l in open(path):
        l = l.rstrip("\n")
        if l.startswith("["): cur = None
        elif l.startswith("id: GO:"): cur = l[4:]
        elif cur and l.startswith("name: "): names[cur] = l[6:]
        elif cur and l.startswith("namespace: "): spaces[cur] = {"molecular_function": "mf", "biological_process": "bp", "cellular_component": "cc"}[l[11:]]
        elif cur and l.startswith("is_a: "): parents.setdefault(cur, set()).add(l[6:].split(" ! ")[0])
        elif cur and l.startswith("relationship: part_of "): parents.setdefault(cur, set()).add(l.split()[2])
        elif cur and l.startswith('synonym: "') and " EXACT" in l: synonyms.setdefault(cur, []).append(l.split('"')[1])
        elif cur and l.startswith("is_obsolete: true"): obsolete.add(cur)
    lookup = {a: {} for a in ASPECTS}
    live = [t for t in spaces if t not in obsolete]
    for t in live:
        for s in synonyms.get(t, []): lookup[spaces[t]].setdefault(s.lower(), t)
    for t in live: lookup[spaces[t]][names[t].lower()] = t
    return lookup, parents


def parse(reply, lookup, parents):
    out = set()
    for piece in reply.replace("\n", ";").split(";"):
        t = lookup.get(piece.strip().strip(".").strip().lower())
        if t: out.add(t)
    todo = list(out)
    while todo:
        for p in parents.get(todo.pop(), ()):
            if p not in out: out.add(p); todo.append(p)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for x in ("model", "data", "features", "out"): ap.add_argument("--" + x, required=True)
    ap.add_argument("--test-set", default="proteins"); ap.add_argument("--qwen", default="Qwen/Qwen3-8B"); ap.add_argument("--max-new-tokens", type=int, default=768)
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--shards", type=int, default=1); ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    index = {r["sequence_sha256"]: r for r in map(json.loads, open(f"{a.features}/index.jsonl"))}
    rows = [r for r in map(json.loads, open(f"{a.data}/proteins.jsonl")) if r["split"] == "test" and r["test_set"] == a.test_set and r["scored_aspects"]]
    rows = sorted(rows, key=lambda r: r["id"])[a.shard::a.shards][:a.limit]
    lookup, parents = ontology(f"{a.data}/ontology/go-basic.obo"); model = RumiFold.from_pretrained(a.model, qwen=a.qwen); out = {}
    for k, r in enumerate(rows):
        e = index[hashlib.sha256(r["seq"].encode()).hexdigest()]
        fold = model.fold(load_features(os.path.join(a.features, os.path.basename(e["shard"])), e["residue_offset"], e["residue_end"]))
        out[r["id"]] = {"go_" + x: {} for x in ASPECTS}
        for x, name in ASPECTS.items():
            if x.upper() not in r["scored_aspects"]: continue
            q = QUESTION.format(name)
            if organism(r): q = f"Organism: {organism(r)}.\n{q}"
            reply = model.ask(q, fold, a.max_new_tokens, addresses=False)
            out[r["id"]]["go_" + x] = {t: 1.0 for t in sorted(parse(reply, lookup[x], parents))}
        if k % 50 == 0: print(f"{k}/{len(rows)}", flush=True)
    json.dump(out, open(a.out, "w"))


if __name__ == "__main__": main()
