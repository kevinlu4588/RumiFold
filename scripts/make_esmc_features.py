"""ESM-C 6B features for RumiFold (needs `pip install esm`).

  python scripts/make_esmc_features.py --sequence MKV... --out protein.safetensors [--layer 76]

--layer final (default) is what every RumiFold model reads except L76; --layer 76 is layer 76 after the encoder's final LayerNorm.
"""
import argparse

import torch
from safetensors.torch import save_file


@torch.inference_mode()
def encode(model, tokenizer, sequence, layer="final"):
    ids = tokenizer([sequence], return_tensors="pt")["input_ids"].to(next(model.parameters()).device)
    states = model(input_ids=ids, output_hidden_states=True).hidden_states
    x = states[-1] if layer == "final" else model.transformer.norm(states[int(layer)])
    return x[0, 1:-1].to(torch.bfloat16).cpu().contiguous()


if __name__ == "__main__":
    from esm.models.esmc import EsmcModel, EsmcTokenizer
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sequence", required=True); p.add_argument("--out", required=True)
    p.add_argument("--layer", default="final"); p.add_argument("--model", default="biohub/ESMC-6B")
    a = p.parse_args()
    save_file({"embeddings": encode(EsmcModel.from_pretrained(a.model), EsmcTokenizer(), a.sequence, a.layer)}, a.out)
    print("wrote", a.out)
