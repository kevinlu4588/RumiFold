"""Prot2Text metrics, following the Prot2Text-V2 benchmark: exact match, BLEU-2/4, ROUGE, BERTScore (RoBERTa-large, BioBERT).

  python eval/prot2text.py --predictions pred.json [--no-bert]      pred.json: {id: {"pred": text, "true": text}}
"""
import argparse
import json
import re


def bertscore(pred, true):
    import evaluate
    from transformers import BertTokenizer, RobertaTokenizer
    bert = evaluate.load("bertscore"); out = {}
    for name, cls, model, extra in (("roberta-large", RobertaTokenizer, "FacebookAI/roberta-large", dict(lang="en")),
                                    ("biobert-large", BertTokenizer, "dmis-lab/biobert-large-cased-v1.1",
                                     dict(model_type="dmis-lab/biobert-large-cased-v1.1", num_layers=24))):
        tok = cls.from_pretrained(model)
        cut = lambda t: tok.batch_decode(tok(t, padding="max_length", truncation=True, max_length=495, return_tensors="pt")["input_ids"], skip_special_tokens=True)
        s = bert.compute(predictions=cut(pred), references=cut(true), **extra)
        out[name] = {k: sum(s[k]) / len(s[k]) for k in ("precision", "recall", "f1")}
    return out


def main():
    import evaluate
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("--predictions", required=True); ap.add_argument("--no-bert", action="store_true")
    a = ap.parse_args(); rows = list(json.load(open(a.predictions)).values())
    pred, true = [r["pred"] for r in rows], [r["true"] for r in rows]
    norm = lambda t: re.sub(r"[^\w]", "", t.lower()); bleu = evaluate.load("bleu")
    out = {"n": len(rows), "exact_match": sum(norm(p) == norm(t) for p, t in zip(pred, true)) / len(rows),
           "bleu2": bleu.compute(predictions=pred, references=true, max_order=2)["bleu"], "bleu4": bleu.compute(predictions=pred, references=true)["bleu"],
           "rouge": evaluate.load("rouge").compute(predictions=pred, references=true)}
    if not a.no_bert: out["bertscore"] = bertscore(pred, true)
    print(json.dumps(out, indent=1))


if __name__ == "__main__": main()
