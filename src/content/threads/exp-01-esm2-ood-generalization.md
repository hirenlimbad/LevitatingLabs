---
title: "Experiment 01: Out-of-Domain Generalization of ESM2 Embeddings in Protein-Ligand Binding"
pubDatetime: 2026-09-10T09:00:00.000Z
modDatetime: 2026-09-16T14:30:00.000Z
status: "checkpoint-reached"
description: "Tracking daily progress testing ESM2 zero-shot protein embeddings against Graph Neural Networks under scaffold-disjoint splits."
tags: ["ESM2", "Binding-Affinity", "Scaffold-Hopping", "OOD"]
relatedArticle: "hetrinet-esm2-ood"
featured: true
---

## Day 1: Baseline Architecture & Dataset Prep
*10 Sep 2026*

- Initialized scaffold-disjoint split on **BindingDB 2024** benchmark dataset.
- Extracting raw sequences for target proteins and calculating **ESM2 (650M parameter model)** residue-level embeddings.
- **Goal:** Ensure zero overlap between training compound scaffolds and test compound scaffolds.

```python
import torch
from transformers import AutoTokenizer, EsmModel

tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
model = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D").cuda()

# Generate mean-pooled sequence representation
def extract_embedding(seq):
    inputs = tokenizer(seq, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)
```

> **Observation:** Initial preprocessing shows 14.2% sequence similarity drop between train and test sets when strict scaffold grouping is enforced.

---

## Day 3: Initial Benchmarking & Leakage Checks
*12 Sep 2026*

- Benchmarking 1D CNN baseline against 3D-Graph Transformer.
- **Metric tracked:** Concordance Index (CI) and Root Mean Squared Error (RMSE) on test set.

### Preliminary Results

| Model Variant | Train RMSE | Test OOD RMSE | Test CI |
| :--- | :--- | :--- | :--- |
| Baseline MLP | 0.412 | 0.894 | 0.612 |
| HetriNet + ESM2 | 0.385 | **0.541** | **0.784** |

- Math verification for loss penalty weighting:
  $$\mathcal{L} = \mathcal{L}_{\text{MSE}} + \lambda \|\mathbf{W}_h\|_2^2$$

---

## Day 6: Checkpoint Reached! 🎯
*15 Sep 2026*

- Reached target performance threshold: RMSE $< 0.55$ under zero-shot target scaffold splits.
- All evaluation scripts verified with zero data leakage.
- **Checkpoint Action:** Compiled findings, loss curves, and architectural diagrams into a full research article post!
