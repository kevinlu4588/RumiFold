"""Information-weighted F1 for GO predictions on the GO-GPT v2 test set (cafaeval, IA weights, OBO propagation).

  python eval/go.py --predictions pred.json --data DIR [--test-set proteins]

pred.json: {protein id: {"go_mf"|"go_bp"|"go_cc": {GO id: score}}}; hard predictions use score 1.0.
DIR is the RumiFold_GO dataset (proteins.jsonl, ontology/IA.txt, ontology/go-basic.obo).
"""
import argparse
import json
import os
import tempfile

from cafaeval.evaluation import cafa_eval

ASPECTS = {"MF": "go_mf", "BP": "go_bp", "CC": "go_cc"}
NAMESPACE = {"MF": "molecular_function", "BP": "biological_process", "CC": "cellular_component"}


def f1(obo, ia, truth, pred, aspect):
    if not truth or not pred: return 0.0
    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(f"{tmp}/p")
        with open(f"{tmp}/gt.tsv", "w") as f: f.writelines(f"{p}\t{g}\n" for p, g in truth)
        with open(f"{tmp}/p/pred.tsv", "w") as f: f.writelines(f"{p}\t{g}\t{s:.4f}\n" for p, g, s in pred)
        res, _ = cafa_eval(obo, f"{tmp}/p", f"{tmp}/gt.tsv", ia=ia, no_orphans=False, norm="cafa", prop="max", th_step=0.5, max_terms=None, n_cpu=4)
    x = res.reset_index()
    return float(x[x.ns == NAMESPACE[aspect]].f_w.max())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions", required=True); ap.add_argument("--data", required=True); ap.add_argument("--test-set", default="proteins")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(f"{a.data}/proteins.jsonl")]
    test = [r for r in rows if r["split"] == "test" and r["test_set"] == a.test_set]
    P = json.load(open(a.predictions)); out = []
    for asp, col in ASPECTS.items():
        use = [r for r in test if asp in r["scored_aspects"] and r["labels"][col]]
        truth = [(r["id"], g) for r in use for g in r["labels"][col]]
        pred = [(r["id"], g, 1.0) for r in use for g in P.get(r["id"], {}).get(col, {})]
        missing = sum(r["id"] not in P for r in use)
        out.append(f1(f"{a.data}/ontology/go-basic.obo", f"{a.data}/ontology/IA.txt", truth, pred, asp))
        print(f"{asp}: weighted F1 {out[-1]:.3f} ({len(use)} proteins, {missing} without a prediction)")
    print(f"mean: {sum(out) / 3:.3f}")


if __name__ == "__main__": main()
