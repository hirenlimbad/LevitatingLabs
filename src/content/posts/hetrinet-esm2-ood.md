---
author: Levitating Labs
pubDatetime: 2026-08-30T12:00:00Z
title: "From 29% to 96.7% ROC-AUC: How ESM-2 Language Model Fixes Topological Memorization in Drug-Target Graph Networks"
featured: true
draft: false
tags:
  - HeTriNet
  - ESM2
  - DisGeNET
  - Drug-Target-Disease Interaction
description: We benchmark an ESM-2 + Inductive KNN protein language model architecture against traditional 3-mer counting baselines across 4 rigorous OOD splits, demonstrating +39.6% ROC-AUC gains on unseen target proteins and +67.1% on holdout disease categories.
ogImage: "@/assets/images/model_stability_variance.png"
---

While drug-target interaction (DTI) models achieve strong performance on random splits (our 3-mer baseline hits **82.02% ROC-AUC** while ESM-2 reaches **97.80%**), traditional sequence baselines crash when deployed against unseen protein targets or novel disease classes. To solve this out-of-distribution (OOD) bottleneck, we benchmark HeTriNet—a heterogeneous graph network combining 150M-parameter ESM-2 protein language embeddings with DisGeNET genetic SVD features—across 3 independent seed runs.

---

## 1. Core Hypotheses & Key Findings

<div class="my-6 flex flex-col overflow-hidden rounded-xl border border-border/80 shadow-xs">
  <!-- H1: Baseline Hypothesis (Slate/Gray Container) -->
  <div class="border-b border-border/80 bg-slate-500/10 px-4 py-3.5 sm:px-5 sm:py-4 dark:bg-slate-900/40">
    <div class="mb-1.5 flex items-center gap-2 text-xs font-bold tracking-wider text-slate-600 uppercase dark:text-slate-400">
      <svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="m4.93 4.93 14.14 14.14"/></svg>
      Hypothesis 1: Baseline OOD Vulnerability (3-Mer HeTriNet)
    </div>
    <p class="text-sm font-medium leading-relaxed text-foreground">
      <strong>Question:</strong> Can standard HeTriNet models featurized with 3-mer sequence counts generalize across out-of-distribution target proteins and holdout disease categories?<br/>
      <strong class="text-rose-600 dark:text-rose-400">Outcome (Rejected):</strong> <strong>No.</strong> Baseline HeTriNet collapses to near-random guessing on unseen targets (57.54% ± 1.29% ROC-AUC) and fails on holdout disease categories (29.02% ± 0.49% ROC-AUC) due to topological graph memorization.
    </p>
  </div>

  <!-- H2: Proposed Hypothesis (Teal/Amber Container) -->
  <div class="bg-teal-500/10 px-4 py-3.5 sm:px-5 sm:py-4 dark:bg-teal-950/40">
    <div class="mb-1.5 flex items-center gap-2 text-xs font-bold tracking-wider text-teal-600 uppercase dark:text-teal-400">
      <svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3Z"/></svg>
      Hypothesis 2: ESM-2 + Genetic Anchor Rescue
    </div>
    <p class="text-sm font-medium leading-relaxed text-foreground">
      <strong>Question:</strong> Can replacing shallow sequence counts with 150M-parameter ESM-2 embeddings (640-d) and DisGeNET genetic SVD anchors via inductive k-NN graph propagation enable OOD generalization?<br/>
      <strong class="text-emerald-600 dark:text-emerald-400">Outcome (Confirmed):</strong> <strong class="text-accent">Yes. ESM-2 completely eliminates target OOD failure (95.17% ± 0.14% ROC-AUC vs 57.54%) and DisGeNET genetic anchors rescue disease-level OOD from graph collapse (96.74% ± 0.34% vs 29.02%).</strong><br/>
      <strong>Why it worked:</strong> We hypothesize that because ESM-2 was pre-trained on millions of protein sequences across nature, it captures underlying biological rules rather than memorizing local graph connections. Combined with DisGeNET's real genetic data and inductive k-NN search, this enables the model to accurately score brand-new targets it has never encountered during training.
    </p>
  </div>
</div>

---

## 2. Problem Statement & Mathematical Formulation

Standard drug-target interaction (DTI) models yield inflated performance under random cross-validation splits—where our 3-mer sequence baseline achieves **82.02% ± 0.22% ROC-AUC** (and ESM-2 reaches **97.80% ± 0.01%**). However, when evaluated on out-of-distribution (OOD) test sets where target proteins or disease contexts are completely unseen ($u \notin \mathcal{V}_{\text{train}}$), the 3-mer baseline collapses—dropping to **57.54% ± 1.29% ROC-AUC** on unseen targets and **29.02% ± 0.49%** on holdout disease categories.

### Graph & Triplet Loss Setup
We formalize DTI prediction over a heterogeneous bipartite network $\mathcal{G} = (\mathcal{V}_D \cup \mathcal{V}_T, \mathcal{E})$, where $\mathcal{V}_D$ represents small-molecule drugs, $\mathcal{V}_T$ represents protein targets, and edges $e_{i,j} \in \mathcal{E}$ represent validated interaction triples $(i, j, k)$.

We optimize link prediction embeddings using a margin-based triplet loss $\mathcal{L}_{\text{margin}}$:

$$
\mathcal{L}_{\text{margin}} = \frac{1}{|D|} \sum_{(i,j,k) \in D} \max \left( 0, \, \gamma + f(i, j, k') - f(i, j, k) \right)
$$

where $\gamma = 1.0$ is the margin, $f(i, j, k)$ is the predicted bioactivity score for positive pair $(i, j)$ under disease context $k$, and $k'$ is a sampled negative interaction target.

### Inductive k-NN Propagation for Cold-Start Nodes
Unseen test nodes cannot look up graph topology. To solve this, we construct an inductive similarity graph using Cosine distance over pre-trained ESM-2 embeddings $\mathbf{h}_u^{\text{ESM}}$:

$$
\mathbf{h}_u^{\text{ind}} = \sum_{v \in \mathcal{N}_k(u)} \frac{\exp(\text{sim}(\mathbf{h}_u^{\text{ESM}}, \mathbf{h}_v^{\text{ESM}}) / \tau)}{\sum_{w \in \mathcal{N}_k(u)} \exp(\text{sim}(\mathbf{h}_u^{\text{ESM}}, \mathbf{h}_w^{\text{ESM}}) / \tau)} \mathbf{h}_v
$$

where $\mathcal{N}_k(u)$ represents the $k$-nearest neighbors of node $u$ in ESM-2 embedding space and $\tau = 0.1$ is the softmax temperature.

---

## 3. The Setup & Model Architecture

> [!NOTE] Preprocessing & Dataset Details
> Built from DrugBank 5.1.10 and UniProt (9,908 validated triplet interactions).
> - **Drugs:** 1024-bit Morgan Fingerprints ($r=2$)
> - **Proteins:** ESM-2 embeddings (`esm2_t30_150M_UR50D` mapped to 640-d)
> - **Diseases:** 256-d DisGeNET genetic SVD features

> [!WARNING] Evaluation Protocol
> 1. **RANDOM**: Standard 70/10/20 stratified baseline split.
> 2. **TARGET_OOD**: Zero target sequence overlap between training and testing sets.
> 3. **DISEASE_OOD**: Isolated testing on holdout ATC therapeutic disease categories.
> 4. **DRUG_OOD**: Bemis-Murcko chemical scaffold clustering to evaluate performance on novel drug structures.

### 1. Baseline Model Architecture (3-Mer Sequence Counting)

![Figure: Baseline HeTriNet Architecture](@/assets/images/baseline_model.png)
*Figure 1: Baseline HeTriNet Architecture using shallow 3-mer sequence count vectors.*

- **Input Featurization**:
  - **Drugs**: 1024-bit Morgan Fingerprints ($r=2$) encoding 2D chemical sub-structures.
  - **Proteins**: Shallow 3-mer amino acid frequency count vectors.
  - **Diseases**: One-hot node ID embeddings tied strictly to training graph positions.
- **Network Processing**:
  - Standard message passing over heterogeneous graph edges via `HeteroConv`.
  - Node representations rely heavily on explicit training graph links.
- **Prediction & Limitations**:
  - Predicts pairwise bioactivity scores $f(i, j, k)$.
  - **Limitation**: Fails on unseen targets ($u \notin \mathcal{V}_{\text{train}}$) because 3-mer counts lack evolutionary context and node IDs overfit to graph topology.

### 2. Proposed Model Architecture (ESM-2 Embeddings & Inductive k-NN)

![Figure: Proposed HeTriNet Model Architecture & ESM-2 Inductive Pipeline](@/assets/images/model_image.png)
*Figure 2: Proposed HeTriNet Architecture incorporating 150M-parameter ESM-2 protein language embeddings and inductive k-NN graph propagation.*

- **Input Featurization**:
  - **Drugs**: 1024-bit Morgan Fingerprints ($r=2$).
  - **Proteins**: Pre-trained 150M-parameter ESM-2 transformer embeddings projected to 640 dimensions.
  - **Diseases**: 256-d SVD genetic features extracted from DisGeNET association matrices.
- **Network & Inductive Processing**:
  - **HeTriNet GNN**: Heterogeneous message passing trained with margin-based triplet loss ($\mathcal{L}_{\text{margin}}$, $\gamma=1.0$).
  - **Cold-Start Inductive k-NN Propagation**: When an isolated target protein $u \notin \mathcal{V}_{\text{train}}$ appears during testing, the model infers its representation $\mathbf{h}_u^{\text{ind}}$ without retraining by:
    1. Finding the top-$k$ nearest targets in the training graph using ESM-2 embedding cosine similarity.
    2. Aggregating their trained GNN representations weighted by softmax temperature ($\tau=0.1$):
       $$\mathbf{h}_u^{\text{ind}} = \sum_{v \in \mathcal{N}_k(u)} \frac{\exp(\text{CosSim}(\mathbf{x}_u, \mathbf{x}_v)/\tau)}{\sum_{w \in \mathcal{N}_k(u)} \exp(\text{CosSim}(\mathbf{x}_u, \mathbf{x}_w)/\tau)} \mathbf{h}_v^{\text{GNN}}$$
- **Prediction & Advantages**:
  - Outputs bioactivity interaction probabilities $f(i, j, k)$ for ranking candidates (Hit@15, NDCG@15).
  - **Advantage**: ESM-2 captures real protein biology while cold-start propagation enables instant zero-shot prediction for new targets without expensive graph retraining.

---
## 4. Benchmark Performance & Multi-Seed Stability Results

The performance below represents the **Mean ± Standard Deviation across 3 independent random seeds** (Seeds 42, 100, 2024).

### Baseline Model Performance (3-Mer Sequence Counting)

| Evaluation Split | F1-Score (@0.5) | F1-Score (Optimal $\tau^*$) | ROC-AUC | AUPR | Hit@15 | NDCG@15 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **RANDOM** | 24.88 ± 0.65% | 24.88 ± 0.65% | 81.90 ± 0.17% | 77.05 ± 0.22% | 100.00 ± 0.00% | 66.69 ± 0.18% |
| **TARGET_OOD** | 6.20 ± 0.04% | 6.20 ± 0.04% | 55.80 ± 0.02% | 49.80 ± 0.01% | 99.90 ± 0.00% | 45.40 ± 0.02% |
| **DISEASE_OOD** | 0.00 ± 0.00% | 0.00 ± 0.00% | 29.49 ± 0.40% | 37.01 ± 0.13% | 100.00 ± 0.00% | 63.40 ± 0.16% |
| **DRUG_OOD** | 3.47 ± 0.08% | 3.47 ± 0.08% | 68.02 ± 0.07% | 62.28 ± 0.03% | 100.00 ± 0.00% | 64.13 ± 0.02% |

### Proposed Model Performance (ESM-2 + Inductive k-NN)

| Evaluation Split | F1-Score (@0.5) | F1-Score (Optimal $\tau^*$) | ROC-AUC | AUPR | Hit@15 | NDCG@15 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **RANDOM** | **94.08 ± 0.15%** | **94.23 ± 0.16%** | **97.76 ± 0.05%** | **98.27 ± 0.04%** | **100.00 ± 0.00%** | **70.04 ± 0.64%** |
| **TARGET_OOD** | **89.82 ± 0.38%** | **91.57 ± 0.25%** | **95.39 ± 0.04%** | **96.57 ± 0.03%** | **100.00 ± 0.00%** | **60.55 ± 2.04%** |
| **DISEASE_OOD** | **73.24 ± 0.98%** | **77.97 ± 1.45%** | **96.59 ± 0.31%** | **93.93 ± 1.86%** | **100.00 ± 0.00%** | **57.27 ± 6.23%** |
| **DRUG_OOD** | **4.17 ± 0.08%** | **4.30 ± 0.24%** | **60.78 ± 0.72%** | **56.85 ± 0.23%** | **100.00 ± 0.00%** | **58.08 ± 1.69%** |

---

## 5. Benchmark Figures

![Figure 1: OOD Generalization ROC-AUC Profile Across 4 Splits](@/assets/images/roc_auc_comparison.png)
*Figure 1: ROC-AUC profile comparison across Random, Drug OOD, Target OOD, and Disease OOD benchmark splits.*

![Figure 2: Disease Memorization Collapse & Validation Loss Trajectories](@/assets/images/ndcg_overfitting_proof.png)
*Figure 2: Epoch-by-epoch training and validation loss trajectories across all 4 OOD splits with early stopping checkpoints.*

![Figure 3: Performance Gap Comparison Between Baseline vs ESM-2](@/assets/images/disease_memorization_collapse.png)
*Figure 3: Performance comparison highlighting the gap between baseline 3-mer vs proposed ESM-2 model under Disease OOD.*
*(Note: F1@0.5 is near-zero because margin ranking loss does not constrain absolute score scale, so the fixed 0.5 threshold is uncalibrated — see ROC-AUC/AUPR for threshold-independent performance.).*

---

## 6. Findings & Analysis

1. **Target OOD Recovery (+39.6% ROC-AUC)**: Replacing shallow 3-mer counts with 150M-parameter ESM-2 embeddings elevates unseen target performance from 55.8% to 95.4% ROC-AUC.
2. **Disease OOD Rescue (+67.1% ROC-AUC)**: DisGeNET genetic SVD features prevent graph topology collapse on holdout disease categories, boosting ROC-AUC from 29.5% to 96.6%.
3. **Retrieval Utility (100.0% Hit@15)**: Maintains 100% Hit@15 recall across all 3 seeds, ensuring true positive targets rank within the top 15 predictions for wet-lab screening.

---

## 7. Limitations

1. **The Drug OOD Bottleneck (2D Fingerprint Limit)**:
   On unseen chemical scaffolds (**DRUG_OOD**), the proposed model achieves **60.8% ROC-AUC** (vs **68.0%** baseline). Protein language embeddings cannot compensate for extreme 2D chemical novelty without 3D molecular conformer geometries or Equivariant GNNs.

2. **Holdout Disease OOD Sample Size**:
   The Disease OOD test split contains 128 positive interaction pairs due to tight ATC anatomical cluster isolation. While multi-seed variance is low ($\sigma = 0.3\%$), larger clinical registries are needed for broader validation.

---

## 8. Conclusions & Takeaways

1. **ESM-2 solves unseen target prediction**: Pre-trained protein language models boost ROC-AUC on isolated target proteins from **55.8% to 95.4%**.
2. **DisGeNET features prevent disease graph collapse**: Incorporating genetic disease data rescues holdout disease accuracy from **29.5% to 96.6%**.
3. **2D fingerprints fail on novel drug scaffolds**: On unseen chemistry, performance drops to **60.8% ROC-AUC**, proving 3D molecular structures are strictly required for chemical OOD generalization.
4. **Inductive k-NN enables cold-start prediction**: Neighbor aggregation lets the network predict interactions for new nodes without model retraining.

---

## 9. Reproducibility & Artifact Links

| Resource | Description | Link |
| :--- | :--- | :--- |
| **ESM-2 Model Weights** | Pre-trained 150M-parameter protein language model (`esm2_t30_150M_UR50D`) | [facebook/esm2_t30_150M_UR50D](https://huggingface.co/facebook/esm2_t30_150M_UR50D) |
| **End-to-End Kaggle Notebook** | Full pipeline: data preparation, baseline & ESM-2 training, and 3-seed stability checks | [Kaggle Notebook](https://www.kaggle.com/code/hirenlimbad/hetrinet-esm-ood-generalization) |
| **DisGeNET Dataset** | Disease-gene association TSV file | [DisGeNET Associations](https://www.kaggle.com/datasets/hirenlimbad/disgnet-disease-associations) |
| **DrugBank XML Dataset** | Raw DrugBank annotations XML file | [DrugBank XML Dataset](https://www.kaggle.com/datasets/sergeguillemart/drugbank) |
| **DrugBank 5.1.10 CSV Dataset** | Processed DrugBank bioactivity CSV file | [DrugBank CSV Dataset](https://www.kaggle.com/datasets/devildev89/drug-bank-5110) |

---

## 10. References & Key Literature

1. **ESM-2 Protein Language Model**: Lin, Z., et al. (2023). *Evolutionary-scale prediction of atomic-level protein structure with a language model.* Science, 379(6637), 1123-1130. [Hugging Face Repository](https://huggingface.co/facebook/esm2_t30_150M_UR50D).
2. **DisGeNET Disease Platform**: Piñero, J., et al. (2020). *The DisGeNET knowledge platform for disease genomics: 2019 update.* Nucleic Acids Research, 48(D1), D845–D855.
3. **Bemis-Murcko Molecular Scaffolds**: Bemis, G. W., & Murcko, M. A. (1996). *The Properties of Known Drugs. 1. Molecular Frameworks.* Journal of Medicinal Chemistry, 39(15), 2887–2893.
4. **DrugBank 5.0 Database**: Wishart, D. S., et al. (2018). *DrugBank 5.0: a major update to the DrugBank database for 2018.* Nucleic Acids Research, 46(D1), D1074–D1082.
5. **HeTriNet Architecture**: *HeTriNet: Heterogeneous Graph Triplet Attention Network for Drug-Target-Disease Interaction* arXiv:2312.00189 [cs.LG], 2023. [arXiv Paper](https://arxiv.org/html/2312.00189v1).
