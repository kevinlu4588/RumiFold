# What each released checkpoint was trained on

Read from the training configuration stored inside each checkpoint, except where noted. Runs were stopped, restarted and re-weighted along the way; the configs in this repository are a starting point, not a replay of that history.

| Checkpoint | Step | Comes from | Notes |
|---|---|---|---|
| RumiFold-Struct 100k | 100,000 | its own line, from base Qwen3-8B | folding only, chains up to 512 residues |
| RumiFold-Struct 174k | 173,711 | RumiFold-Struct 100k | folding only, expanded to chains up to 2,048 residues |
| RumiFold 95k | 95,000 | its own line, from base Qwen3-8B | folding, structural QA, functional training and replay |
| RumiFold 156k | 156,000 | RumiFold 95k | structure-weighted continuation |
| RumiFold-GO | 222,800 | RumiFold 95k | GO-GPT v2 training set; exact GO-GPT v2 test sequences removed |
| RumiFold-GO (interpretability) | 187,550 | RumiFold 95k | earlier checkpoint of the RumiFold-GO line |
| RumiFold-Prot2Text | 246,000 | RumiFold 95k | Prot2Text training split |
| RumiFold-Prot2Text (L76) | weight average | RumiFold 95k, through the Prot2Text run | reads ESM-C layer 76 |

## Folding data

The released checkpoints were trained on a larger set that also includes PDB entries from 2022 onward. `RumiFold_Data` is the pre-2022 cutoff dataset.

Of the 239,217 PDB training rows, 135,995 (57%) were released in 2022 or later. 239 of the 334 FoldBench targets and 110 of the 200 CAMEO 2025 targets share a PDB entry with training. The 34 CASP14 targets with public coordinates were screened by sequence; see below for CAMEO22.

RumiFold-Struct 100k read a 512-residue list, which was screened for both CASP14 and CAMEO22. RumiFold-Struct 174k (which continues from 100k) and RumiFold 156k read the 2,048-residue list, and the stored configuration shared by the Gene Ontology fine-tunes names the same list, although RumiFold-GO (222,800) did not draw folding items from it: its fold loss used AlphaFold models of the Gene Ontology proteins. That list put back 14,146 rows that had been excluded only for similarity to a CAMEO22 target (5,393 PDB rows released before 2022 and 8,753 AlphaFold models), so CAMEO22 is not held out at 30% identity for models trained on it. The CAMEO22 entries themselves stayed excluded. `RumiFold_Data` removes those rows, and everything else matching a CAMEO22 target or CASP14 sequence, but no released checkpoint was trained on it.
