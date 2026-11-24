# Training

One trainer, `train/trainer.py`, runs every stage. A config lists tasks with sampling weights; each step accumulates `effective_batch` items. It saves and resumes checkpoints in `output`, and writes `model.safetensors` with the trained parameters.

Configs that continue from a checkpoint expect it in `checkpoints/rumifold-95k` (`hf download RumiFold/RumiFold_Models --include "rumifold-95k/*" --local-dir checkpoints`), and the datasets in folders named after their repositories.

```bash
torchrun --nproc_per_node 8 -m train.trainer configs/1_fold.json
```

| Task | Reads | Checked against the original code |
|---|---|---|
| `fold` | `RumiFold_Data` | loss and gradients identical |
| `replay` | `RumiFold_Functional/replay` | loss; gradients within GPU noise |
| `prot2text` | `RumiFold_Prot2Text` | loss (text answer; fold loss not used) |
| `gene_ontology` | `RumiFold_GO` | loss and 7,208 answers (text) |

Supplementary, with `configs/supplementary_*.json`:

| Task | Reads | Checked |
|---|---|---|
| `gene_ontology` with `head`, `residue_labels` | `RumiFold_GO` | both heads' outputs on the released weights identical; residue targets and loss identical on 892 proteins; GO targets follow `go_vocab.json` and were not compared |
| `residue` | `RumiFold_Functional` | same residue head, targets and loss |
| `structural_qa` | `RumiFold_Data`, `RumiFold_Functional` | wording, tokens and loss identical on 403 published questions |

Not included: the function-question tasks, the observation and residue-address items, and the on-the-fly question families of the paper's recipe. The configs are a starting point with the settings read from the released checkpoints; the released runs were stopped, restarted and reweighted, so these are not a replay of them. Nothing here was trained end to end: each task was run for a few steps on slices of the published data.

[lineage.md](lineage.md) lists what each released checkpoint was trained on.
