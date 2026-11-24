# What each released checkpoint was trained on

The checkpoint table summarizes the stored training configurations. The folding corpus and length schedule below reflect the authors’ release clarification. Runs were stopped, restarted and re-weighted along the way; the configs in this repository are a starting point, not a replay of that history.

| Checkpoint | Step | Comes from | Notes |
|---|---|---|---|
| RumiFold-Struct 100k | 100,000 | its own line, from base Qwen3-8B | folding only, proteins shorter than 512 residues |
| RumiFold-Struct 174k | 173,711 | RumiFold-Struct 100k | folding only, expanded to chains up to 2,048 residues |
| RumiFold 95k | 95,000 | its own line, from base Qwen3-8B | folding, structural QA, functional training and replay |
| RumiFold 156k | 156,000 | RumiFold 95k | structure-weighted continuation |
| RumiFold-GO | 222,800 | RumiFold 95k | GO-GPT v2 training set; exact GO-GPT v2 test sequences removed |
| RumiFold-GO (interpretability) | 187,550 | RumiFold 95k | earlier checkpoint of the RumiFold-GO line |
| RumiFold-Prot2Text | 246,000 | RumiFold 95k | Prot2Text training split |
| RumiFold-Prot2Text (L76) | weight average | RumiFold 95k, through the Prot2Text run | reads ESM-C layer 76 |

## Folding data

Base folding training used the training split of [RumiFold_Data](https://huggingface.co/datasets/RumiFold/RumiFold_Data), which combines PDB assemblies first released before 1 January 2022 with AlphaFold Database models. The folding corpus was screened against CAMEO22 and CASP14.

RumiFold-Struct trained for its first 100,000 steps on proteins shorter than 512 residues. Training then continued on longer proteins under the same CAMEO22 and CASP14 screening criteria.

For RumiFold-GO (step 222,800), the folding loss used AlphaFold models of the Gene Ontology training proteins.
