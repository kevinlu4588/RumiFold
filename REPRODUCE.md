# What this code reproduces

| Result | Check | Outcome |
|---|---|---|
| GO, Table 1 (RumiFold) | `eval/go.py` on saved predictions | 0.685 / 0.615 / 0.748, mean 0.683: matches the paper |
| Prot2Text (L76), all 4,203 proteins | `eval/prot2text.py` on saved predictions | BLEU-4 42.5, ROUGE-L 55.8: matches the paper. BERTScore, the novel-protein subset and the GPT-4o judge were not rechecked |
| Folding, FoldBench / CASP16 / CAMEO 2025 (RumiFold, RumiFold-Struct) | `eval/fold.py` metrics on saved structures | TM-score and CA-lDDT match the paper to three decimals. GDT-TS comes out 0.002-0.004 higher than the paper, whose folding table was scored with the benchmark authors' script |
| Inference | this code against the original research code | identical backbones and 12 of 12 replies on 3 proteins (95k); 18 of 18 GO predictions on 12 test proteins (RumiFold-GO); 16 of 16 replies on 16 test proteins (L76) |
| RumiFold-Struct (174k) | this code against the original research code | identical backbones and pLDDT on two proteins and on a two-chain input |
| Loading | every `rumifold-*` checkpoint | loads and folds (95k and L76 were checked as above); on subtilisin (mature enzyme, against its AlphaFold model) the TM-score is 0.970 (Prot2Text), 0.977 (156k), 0.984 (GO, interpretability), 0.986 (GO), 0.980 (Struct 100k), 0.989 (Struct 174k) |

Not tested here: multi-chain inputs for the language models. The `controls/` and `saes/` weights have no loading code.

