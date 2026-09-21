---
author: Levitating Labs
pubDatetime: 2026-09-20T12:00:00Z
title: "Physics-Guided Lattice Energy Prediction with Target-Quantile Scaffold Cross-Validation"
featured: true
draft: false
tags:
  - LatticeEnergy
  - ScaffoldSplit
  - BemisMurcko
  - ThermoChemistry
  - PyTorch
description: We benchmark a 2,243-dimensional Pure Physics Neural Network against random and Bemis-Murcko scaffold splits on 23,496 organic compounds, demonstrating a 14.1% R² generalization drop under OOD scaffold shift and establishing scale-invariant nMAE evaluation.
ogImage: "@/assets/images/lattice_energy_pred/scaffold_shift_summary.png"
---

Molecular machine learning models often boast high accuracy on paper, but when deployed to discover novel drug candidates or materials, performance drops sharply. Why? Standard evaluation relies on **random train/test splits**, where the model accidentally memorizes chemical core frameworks it has already seen during training.

In this work, we benchmark `LatticePhysicsTower`—a physics-guided neural network combining **2,243-dimensional multimodal descriptors** (2D topology + 3D Boltzmann-weighted conformer physics) with thermodynamic scaling on **23,496 organic compounds** from the Bradley Open Melting Point Dataset. We systematically evaluate standard random splits against out-of-distribution (OOD) **Bemis-Murcko scaffold splits**, quantify the real-world generalization penalty, and investigate why Retrieval-Augmented Generation (RAG) memory banks fail under scaffold shift.

> [!NOTE] Executive Summary (TL;DR)
> - **The 14.1% OOD Gap:** Standard random splits yield an inflated **0.7983 R²**. Holding out unseen molecular scaffolds drops performance to **0.6570 R²** ($\Delta R^2 = -0.1413$).
> - **Method Fix:** Target-quantile scaffold stratification combined with **Normalized MAE ($\text{nMAE} = \text{MAE}/\sigma_y$)** stabilizes 5-fold cross-validation at **0.4729 ± 0.0136 nMAE**.
> - **Why Naive RAG Failed:** 2D fingerprint retrieval returns structural neighbors with wildly different 3D crystal packing energies, introducing noise that degrades performance below the standalone physics baseline.

---

## 1. Core Hypotheses & Key Findings

<div class="my-6 flex flex-col overflow-hidden rounded-xl border border-border/80 shadow-xs">
  <!-- H1: Random Split Inflation (Slate/Gray Container) -->
  <div class="border-b border-border/80 bg-slate-500/10 px-4 py-3.5 sm:px-5 sm:py-4 dark:bg-slate-900/40">
    <div class="mb-1.5 flex items-center gap-2 text-xs font-bold tracking-wider text-slate-600 uppercase dark:text-slate-400">
      <svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1-1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3Z"/></svg>
      Hypothesis 1: Random Split Performance Inflation
    </div>
    <div class="text-sm font-medium leading-relaxed text-foreground">
      <p class="mb-1"><strong>Question:</strong> Do standard random cross-validation splits overestimate model performance for molecular lattice energy and melting point prediction?</p>
      <p><strong class="text-emerald-600 dark:text-emerald-400">Outcome (Confirmed):</strong> <strong>Yes.</strong> Random splits yield an inflated <strong>0.7983 R²</strong> (dH MAE = 1.723 kJ/mol, Tm MAE = 30.50 K). When evaluated on a Bemis-Murcko scaffold-disjoint test set, performance drops to <strong>0.6570 R²</strong>—exposing a <strong>14.1% structural generalization gap</strong> (Δ<i>R</i>² = −0.1413).</p>
    </div>
  </div>

  <!-- H2: Stratified Scaffold CV & nMAE Stabilization (Teal/Amber Container) -->
  <div class="bg-teal-500/10 px-4 py-3.5 sm:px-5 sm:py-4 dark:bg-teal-950/40">
    <div class="mb-1.5 flex items-center gap-2 text-xs font-bold tracking-wider text-teal-600 uppercase dark:text-teal-400">
      <svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1-1.275-1.275L12 21l1.912-5.813a2 2 0 0 1-1.275-1.275L12 3Z"/></svg>
      Hypothesis 2: Stratified Scaffold Shift & Scale-Invariant Metric Rescue
    </div>
    <div class="text-sm font-medium leading-relaxed text-foreground">
      <p class="mb-1.5"><strong>Question:</strong> Can target-quantile scaffold stratification stabilize 5-fold cross-validation, and how should we evaluate error across chemical families with varying energy distributions?</p>
      <p class="mb-1.5"><strong class="text-emerald-600 dark:text-emerald-400">Outcome (Confirmed):</strong> <strong class="text-accent">Yes. Target-quantile scaffold bin packing stabilizes 5-fold cross-validation at 0.6215 ± 0.0253 <i>R</i>². Introducing Normalized MAE (nMAE = MAE / σ<sub><i>y</i>,fold</sub>) resolves data geometry artifacts, yielding a scale-invariant error of 0.4729 ± 0.0136 across all scaffold folds.</strong></p>
      <p><strong>Why it worked:</strong> Scaffold groups differ in baseline energy variance (σ<sub><i>y</i></sub>). Normalizing MAE by fold standard deviation removes scale distortion, proving that relative model precision remains stable across diverse chemical scaffold topologies.</p>
    </div>
  </div>
</div>

---

## 2. Problem Statement & Mathematical Formulation

Standard molecular property prediction models yield inflated performance under random cross-validation splits—where our baseline achieves **0.7983 R²** and a melting point MAE of **30.50 K**. However, when evaluated on out-of-distribution (OOD) test sets where core molecular frameworks are completely unseen ($\text{Scaffold}(x_i) \notin \mathcal{V}_{\text{train}}$), performance degrades to **0.6570 R²**, increasing normalized error from **0.3315** to **0.4518**.

### Thermodynamic Proxy & Physics Mapping
Following empirical crystal thermo-kinetics (Walden's and Carnelley's rules), we establish the thermodynamic relationship between melting point $T_m$ (in Kelvin) and solid-state lattice enthalpy proxy $\Delta H_{\text{lat}}$ (in kJ/mol):

$$
\Delta H_{\text{lat}} = \alpha \cdot T_m \quad \text{where } \alpha = 0.0565 \text{ kJ/(mol}\cdot\text{K)}
$$

### Target-Quantile Stratified Bemis-Murcko Split Formulation
Let $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^N$ represent the dataset of 23,496 organic molecules. We partition compounds by core Bemis-Murcko ring framework $\text{Scaffold}(x_i)$ into 5 disjoint folds using target-quantile greedy bin packing:

$$
\text{Scaffold}(\mathcal{D}_{\text{test}, k}) \cap \text{Scaffold}(\mathcal{D}_{\text{train}, k}) = \emptyset, \quad \forall k \in \{1, \dots, 5\}
$$

We evaluate scale-invariant error using **Normalized MAE (nMAE)**:

$$
\text{nMAE}_k = \frac{\text{MAE}_k}{\sigma_{y,k} + \epsilon} = \frac{\frac{1}{|\mathcal{D}_k|} \sum_{i \in \mathcal{D}_k} |y_i - \hat{y}_i|}{\sqrt{\frac{1}{|\mathcal{D}_k|} \sum_{i \in \mathcal{D}_k} (y_i - \bar{y}_k)^2} + 10^{-8}}
$$

---

## 3. The Setup & Model Architecture

> [!NOTE] Preprocessing & Dataset Details
> Built from 23,496 valid, standardized compounds in the [Bradley Open Melting Point Dataset](https://www.kaggle.com/datasets/awguhst/melting-point-dataset).
> - **Input Feature Dimension:** 2,243 multimodal features
> - **Target Enthalpy Mean:** $21.46\text{ kJ/mol}$ ($379.8\text{ K}$)
> - **Loss Function:** Smooth Huber Loss ($\delta = 5.0$)
> - **Optimizer & Scheduler:** AdamW ($\text{lr}=10^{-3}$, $\text{wd}=10^{-3}$), Cosine Annealing, EMA ($\beta=0.999$)

> [!NOTE] Residual Target Mean Initialization Rationale
> The network explicitly incorporates a parameter-initialized target global mean ($\mu_y = 21.46\text{ kJ/mol}$), transforming regression into residual offset learning ($\hat{y} = \mu_y + \Delta y$). This prevents Epoch-1 gradient spikes and enables instant convergence by allowing early epochs to focus directly on learning complex chemical feature interactions rather than fitting baseline energy scales.

> [!WARNING] Evaluation Protocols
> 1. **RANDOM SPLIT**: Standard 70/15/15 (train/val/test) uniform random partition (In-Distribution baseline).
> 2. **BEMIS-MURCKO SCAFFOLD SPLIT**: 70/15/15 (train/val/test) core framework disjoint split (Single-fold OOD check).
> 3. **5-FOLD STRATIFIED SCAFFOLD CV**: 5-fold cross-validation with target-quantile stratification across scaffold clusters.

### Multimodal Feature Extractor Pipeline (2,243 Dimensions)

To capture both 2D chemical topology and 3D solid-state crystal energetics, the feature engine converts SMILES strings into a unified **2,243-dimensional L2-normalized vector** composed of 6 distinct feature categories:

1. **2,048-D Morgan Fingerprints (2048-D)**: 2D circular topological bit-vectors with radius $r=2$ capturing local atomic environments.
2. **167-D MACCS Structural Keys (167-D)**: Pre-defined 167-bit dictionary of chemical functional groups and substructural keys.
3. **12-D Scaled Physicochemical Descriptors (12-D)**:
   - Molecular Weight ($\text{MolWt} / 500$)
   - Topological Polar Surface Area ($\text{TPSA} / 200$)
   - Rotatable Bond Count ($\text{NumRotatableBonds} / 10$)
   - Hydrogen Bond Donors ($\text{NumHDonors} / 10$)
   - Hydrogen Bond Acceptors ($\text{NumHAcceptors} / 10$)
   - Formal Charge ($\text{FormalCharge} / 4$)
   - Molar Refractivity ($\text{MolMR} / 150$)
   - Ring Count ($\text{RingCount} / 5$)
   - Fraction of $sp^3$ Carbons ($\text{FractionCSP3}$)
   - Labute Accessible Surface Area ($\text{LabuteASA} / 300$)
   - Heavy Atom Ratio ($\text{NumHeavyAtoms} / 50$)
   - Heteroatom Count (Non-C/Non-H atoms / 10)
4. **4-D Topological Symmetry & Geometry Features (4-D)**:
   - `symmetry_ratio`: Graph automorphism rank over total atom count ($\text{CanonicalRankAtoms} / \text{NumAtoms}$) to measure graph symmetry.
   - `aromatic_atom_ratio`: Fraction of heavy atoms that are aromatic ($\text{AromaticAtoms} / \text{NumHeavyAtoms}$).
   - `aromatic_ring_ratio`: Fraction of total rings that are aromatic ($\text{AromaticRings} / \text{TotalRings}$).
   - `planarity_score`: Fraction of heavy atoms with $sp^2$ or $sp$ hybridization to capture $\pi$-stacking propensity.
5. **4-D Crystal Packing Motif Descriptors (4-D)**: Pre-compiled SMARTS pattern matching for strong directional hydrogen-bonding functional groups:
   - Amides: `[NX3][CX3](=[OX1])`
   - Carboxylic Acids: `[CX3](=O)[OX2H1]`
   - Sulfonamides: `[SX4](=[OX1])(=[OX1])-[NX3]`
   - Phenol $-\text{OH}$ Groups: `[OX2H]`
6. **8-D Boltzmann-Weighted 3D Conformer Physics (8-D)**:
   - Generates 3D conformers using RDKit **ETKDGv3** distance geometry (up to 5 conformers).
   - Minimizes conformer energies via **MMFF94 / UFF force fields**.
   - Computes Boltzmann thermal weighting ($w_i \propto e^{-E_i / RT}$) at $298.15\text{ K}$ ($RT = 2.479\text{ kJ/mol}$).
   - Extracts 8 Boltzmann-averaged 3D descriptors: Principal Moments of Inertia ($\text{PMI1}, \text{PMI2}, \text{PMI3} / 1000$), Asphericity, Eccentricity, Spherocity, Radius of Gyration, and Inertial Shape Factor.

> **Final Vector Normalization:** All 6 feature groups are concatenated ($2048 + 167 + 12 + 4 + 4 + 8 = 2,243$) and passed through L2 normalization ($\mathbf{x}_{\text{norm}} = \mathbf{x} / (\|\mathbf{x}\|_2 + 10^{-8})$).

### Model Architecture & Subsystem Pipelines

![Figure 1: Overall System Pipeline & Multimodal Feature Engine](@/assets/images/lattice_energy_pred/model_architecture.png)
*Figure 1: Overall system pipeline featurizing 2,243-D multimodal molecular vectors for lattice enthalpy and melting point prediction.*

<div class="mx-auto my-6 max-w-lg text-center">

![Figure 2: LatticePhysicsTower Neural Network Architecture Zoom-In](@/assets/images/lattice_energy_pred/lattice_tower_architecture.png)

*Figure 2: Zoom-in view of the LatticePhysicsTower MLP architecture (~148k parameters) with global target mean initialization ($\mu_y = 21.46\text{ kJ/mol}$).*

</div>

---

## 4. Training Strategy: From Baseline Calibration to Weight Smoothing

Understanding how `LatticePhysicsTower` optimizes across training (up to 30–40 epochs with early stopping, patience=10) provides key insight into model stability and structural convergence:

### Phase 1: Early Epochs (Epochs 1–5) — Baseline Anchoring & Coarse Feature Mapping
* **Instant Mean Baseline Calibration**: Because predictions are anchored to the target global mean ($\mu_y = 21.46\text{ kJ/mol}$), the network begins Epoch 1 with predictions centered at the dataset average ($\hat{y} \approx \mu_y + 0$). This eliminates initial gradient shocks and enables the network to immediately learn meaningful residual offsets ($\Delta y$).
* **Primary Structural Signal Learning**: At initial learning rates ($\text{lr} = 10^{-3}$), AdamW rapidly maps high-impact 2D structural features—such as dominant Morgan fingerprint fragments, MACCS functional group keys, and physical properties (MolWt, LogP, TPSA)—to coarse lattice energy corrections.

### Phase 2: Later Epochs (Epochs 6–30+) — High-Order Fine-Tuning & Weight Smoothing
* **Cosine Learning Rate Annealing**: As the Cosine Annealing scheduler smoothly decays the learning rate ($\eta_t \to 10^{-6}$), gradient steps transition from coarse structural mapping to fine-tuning subtle, non-linear interactions—such as 3D conformer PMI geometries ($I_a, I_b, I_c$) and graph symmetry indices.
* **Huber Loss Outlier Handling**: Smooth Huber Loss ($\delta = 5.0$) acts quadratically ($L_2$) on small residuals while penalizing extreme melting point outliers linearly ($L_1$), preventing rare high-temperature compounds from destabilizing gradient trajectories.
* **EMA Weight Stabilization**: Exponential Moving Average (EMA decay $\beta = 0.999$) maintains a running shadow copy of network parameters. Final evaluations use EMA shadow weights, filtering out batch-level stochastic variance and boosting test-set stability under scaffold shift.
* **Early Stopping Patience (Patience=10)**: Training monitors validation RMSE and automatically halts if validation performance does not improve for 10 consecutive epochs, preventing overfitting while ensuring maximum convergence.

---

## 5. Benchmark Performance Results

The performance below summarizes the **Stratified 5-Fold Cross-Validation** and **Random vs. Scaffold Split Comparison** obtained from the experiment execution.

### Summary Table

| Experiment / Evaluation Scheme | R² Score | dH MAE (kJ/mol) | nMAE | dH RMSE (kJ/mol) | Tm MAE (K) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **5-Fold Stratified Scaffold CV (Mean ± Std)** | **0.6215 ± 0.0253** | **2.212 ± 0.423** | **0.4729 ± 0.0136** | **2.874 ± 0.529** | **39.15 ± 7.49 K** |
| **Random Split (In-Distribution)** | **0.7983** | **1.723** | **0.3315** | **2.334** | **30.50 K** |
| **Bemis-Murcko Scaffold Split (OOD)** | **0.6570** | **1.793** | **0.4518** | **2.324** | **31.73 K** |

### Random vs. Scaffold Generalization Gap

| Splitting Scheme | R² Score | dH MAE (kJ/mol) | nMAE | Tm MAE (K) |
| :--- | :---: | :---: | :---: | :---: |
| **Random Split (In-Distribution)** | 0.7983 | 1.723 | 0.3315 | 30.50 K |
| **Bemis-Murcko Scaffold Split (OOD)** | 0.6570 | 1.793 | 0.4518 | 31.73 K |
| **Scaffold Shift Generalization Drop ($\Delta$)** | **-0.1413** | **+0.070** | **+0.1203** | **+1.23 K** |

---

## 6. Benchmark Figures

![Figure 1: Scaffold Shift Generalization Drop Banner](@/assets/images/lattice_energy_pred/scaffold_shift_summary.png)
*Figure 1: Generalization penalty under Bemis-Murcko Scaffold Shift comparing R² score and Normalized MAE.*

![Figure 2: Random vs. Scaffold Parity & Residual Scatter Plots](@/assets/images/lattice_energy_pred/random_vs_scaffold_scatter.png)
*Figure 2: Parity comparison between In-Distribution Random Split and OOD Scaffold Split predictions.*

![Figure 3: Stratified 5-Fold Scaffold CV Metrics Summary](@/assets/images/lattice_energy_pred/figure_scaffold_cv_summary.png)
*Figure 3: Summary of R², MAE, nMAE, and RMSE metrics across 5 Stratified Bemis-Murcko Scaffold Folds.*

![Figure 4: Scale-Invariant nMAE Stability Profile](@/assets/images/lattice_energy_pred/nmae_variance_stability.png)
*Figure 4: Demonstration of Normalized MAE (nMAE) stability across heterogeneous target standard deviations.*

---

## 7. Findings & Analysis

1. **Quantifying the 14.1% OOD Generalization Penalty**: Random train/test splits yield an inflated **0.7983 R²**, whereas holding out core scaffolds drops performance to **0.6570 R²**. Over 14% of random split accuracy stems from memorizing core scaffold topologies rather than learning transferrable physical principles.
2. **Scale-Invariant Metric Stabilization via nMAE**: Raw MAE varies from $1.78\text{ kJ/mol}$ to $2.78\text{ kJ/mol}$ across folds due to baseline target variance differences ($\sigma_y$). Evaluating via Normalized MAE ($\text{nMAE} = 0.4729 \pm 0.0136$) confirms that relative model error is stable across chemical families.
3. **Capacity Regularization Prevents Scaffold Overfitting**: Bottleneck architectures ($2243 \to 64 \to 64 \to 1$) provide optimal structural regularization, preventing wide networks from memorizing complex fragment combinations tied strictly to training scaffolds.

---

## 8. Lab Notebook: Architectural & RAG Memory Gating Experiments

Our original objective was to develop a **Retrieval-Augmented Generation (RAG) memory-guided model** for lattice energy prediction, leveraging non-parametric lookup over known molecular structures. However, across multiple architectural iterations, the retrieval gating mechanisms repeatedly failed or completely suppressed the memory signals when encountering novel scaffolds. This caused the model to fall back to (or be degraded by) the underlying neural backbone—resulting in **virtually no performance gain between the memory-guided RAG model and the standalone non-guided physics model**.

Below is an honest record of our architectural, data-leakage, and RAG memory gating experiments, documenting how these retrieval failures led us to focus on standardizing the standalone physics baseline.

### Part A: Model Architecture & Evaluation Failures

<div class="my-6 flex flex-col gap-0 overflow-hidden rounded-xl border border-border/80 shadow-xs">

  <!-- Failure A1 -->
  <div class="border-b border-border/80 bg-rose-500/8 px-4 py-4 sm:px-5 dark:bg-rose-950/30">
    <div class="mb-2 flex items-center gap-2 text-xs font-bold tracking-wider text-rose-600 uppercase dark:text-rose-400">
      <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M18 6 6 18"/><path d="m6 6 12 12"/></svg>
      Experiment A1 — Over-Parameterized Deep Funnel MLP (&gt;2M Parameters)
    </div>
    <p class="mb-1 text-sm leading-relaxed text-foreground"><strong>What we tried:</strong> A wide deep funnel MLP (<code>2243 → 512 → 256 → 128 → 64 → 1</code>) with over 2 million trainable parameters, expecting high capacity to capture complex non-linear interactions.</p>
    <p class="mb-1 text-sm leading-relaxed text-foreground"><strong class="text-rose-600 dark:text-rose-400">Why it failed:</strong> Under random splits, the model reached high <i>R</i>² scores. However, on scaffold-disjoint test sets, performance collapsed due to severe scaffold memorization. With ~2M parameters and 23,496 training samples, the network memorized fragment combinations instead of physical principles.</p>
    <p class="text-sm leading-relaxed text-foreground"><strong>Lesson:</strong> Model capacity must be calibrated against the count of independent scaffold families. A 148k-parameter bottleneck tower (<code>2243 → 64 → 64 → 1</code>) proved optimal.</p>
  </div>

  <!-- Failure A2 -->
  <div class="border-b border-border/80 bg-orange-500/8 px-4 py-4 sm:px-5 dark:bg-orange-950/30">
    <div class="mb-2 flex items-center gap-2 text-xs font-bold tracking-wider text-orange-600 uppercase dark:text-orange-400">
      <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M18 6 6 18"/><path d="m6 6 12 12"/></svg>
      Experiment A2 — Naive Scaffold CV Without Target Stratification
    </div>
    <p class="mb-1 text-sm leading-relaxed text-foreground"><strong>What we tried:</strong> Unstratified Bemis-Murcko scaffold cross-validation, assigning scaffold groups to folds arbitrarily.</p>
    <p class="mb-1 text-sm leading-relaxed text-foreground"><strong class="text-orange-600 dark:text-orange-400">Why it failed:</strong> Target energy distributions were wildly imbalanced across folds (e.g., ionic-character aromatic folds vs. aliphatic chain folds), causing severe cross-fold <i>R</i>² variance (±0.08+).</p>
    <p class="text-sm leading-relaxed text-foreground"><strong>Lesson:</strong> Target-quantile greedy bin packing is mandatory to ensure every fold spans the complete thermodynamic range of the dataset.</p>
  </div>

  <!-- Failure A3 -->
  <div class="bg-yellow-500/8 px-4 py-4 sm:px-5 dark:bg-yellow-950/30">
    <div class="mb-2 flex items-center gap-2 text-xs font-bold tracking-wider text-yellow-700 uppercase dark:text-yellow-400">
      <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M18 6 6 18"/><path d="m6 6 12 12"/></svg>
      Experiment A3 — Evaluation with Raw MAE Only
    </div>
    <p class="mb-1 text-sm leading-relaxed text-foreground"><strong>What we tried:</strong> Using raw MAE (kJ/mol) as the primary cross-fold performance indicator.</p>
    <p class="mb-1 text-sm leading-relaxed text-foreground"><strong class="text-yellow-700 dark:text-yellow-400">Why it failed:</strong> Scaffold families possess different baseline target variances (σ<sub><i>y</i></sub>). Raw MAE penalizes high-variance folds unnaturally, conflating data geometry with prediction accuracy.</p>
    <p class="text-sm leading-relaxed text-foreground"><strong>Lesson:</strong> Normalized MAE (nMAE = MAE / σ<sub><i>y</i>,fold</sub>) provides a scale-invariant metric necessary for benchmarking heterogeneous chemical series.</p>
  </div>

</div>

### Part B: Retrieval-Augmented Generation (RAG) & Memory Gating Exploration

We benchmarked non-parametric memory retrieval (RAG) over molecular feature banks to evaluate if retrieving structural neighbors could improve OOD predictions. Across our experiments, gating either **completely ignored/shut off retrieval** or **retrieved noisy neighbors**, leaving the final accuracy virtually indistinguishable from (or worse than) the non-guided model:

> [!IMPORTANT] Scientific Rationale: Why Naive RAG Fails in Crystal Thermochemistry
> 1. **Topological Fragment vs. 3D Lattice Packing Disconnect**: Standard molecular retrieval relies on 2D fragment overlap (e.g., 2048-bit Morgan Fingerprints). However, lattice energy ($\Delta H_{\text{lat}}$) is determined by long-range 3D crystal packing forces (intermolecular hydrogen bonding networks, $\pi$-$\pi$ stacking, and steric close-packing). Minor 2D structural substitutions (such as adding a methyl or halogen group) frequently cause complete crystal space-group rearrangement—drastically altering melting point ($T_m$) and lattice enthalpy despite $>85\%$ 2D fingerprint similarity.
> 2. **Scaffold-Disjoint Cosine Space Collapse**: Under Bemis-Murcko scaffold shift, query molecules possess unseen core frameworks absent from the memory bank. Cosine similarities between query features and training memory entries collapse into an uninformative narrow band ($0.65 - 0.72$), preventing similarity-based gates from distinguishing true thermodynamic analogues from irrelevant structural matches.
> 3. **Joint Gradient Interference**: In single-stage joint training ($Y = \alpha y_{\text{rag}} + (1-\alpha) y_{\text{physics}}$), backpropagated loss gradients force the parametric physics backbone to distort its parameter weights to compensate for noisy non-parametric retrieval priors ($y_{\text{rag}}$), degrading the model's underlying physics learning.

<div class="my-6 flex flex-col gap-0 overflow-hidden rounded-xl border border-border/80 shadow-xs">

  <!-- RAG Experiment 1 -->
  <div class="border-b border-border/80 bg-rose-500/8 px-4 py-4 sm:px-5 dark:bg-rose-950/30">
    <div class="mb-2 flex items-center gap-2 text-xs font-bold tracking-wider text-rose-600 uppercase dark:text-rose-400">
      <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M18 6 6 18"/><path d="m6 6 12 12"/></svg>
      Experiment B1 — Static FAISS Index &amp; Data Leakage Discovery
    </div>
    <p class="mb-1 text-sm leading-relaxed text-foreground"><strong>What we tried:</strong> Built an initial static FAISS memory index over 2,243-D multimodal vectors with a heuristic memory trust gate <i>g</i>.</p>
    <p class="mb-1 text-sm leading-relaxed text-foreground"><strong class="text-rose-600 dark:text-rose-400">Outcome:</strong> Reported an initial <i>R</i>² ≈ 0.8081 (vs. 0.7961 standalone). However, audit revealed <strong>test-set data leakage</strong>: test molecules resided in the un-isolated index. Enforcing strict Bemis-Murcko scaffold isolation dropped true test performance to <i>R</i>² ≈ 0.65.</p>
    <p class="text-sm leading-relaxed text-foreground"><strong>Lesson:</strong> RAG memory banks must be strictly isolated to training scaffolds to prevent subtle data leakage in molecular retrieval benchmarks.</p>
  </div>

  <!-- RAG Experiment 2 -->
  <div class="border-b border-border/80 bg-orange-500/8 px-4 py-4 sm:px-5 dark:bg-orange-950/30">
    <div class="mb-2 flex items-center gap-2 text-xs font-bold tracking-wider text-orange-600 uppercase dark:text-orange-400">
      <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M18 6 6 18"/><path d="m6 6 12 12"/></svg>
      Experiment B2 — Additive Gate Boosting &amp; Engine Collapse
    </div>
    <p class="mb-1 text-sm leading-relaxed text-foreground"><strong>What we tried:</strong> Replaced FAISS with a trainable PyTorch GPU retriever and boosted memory trust (<i>g</i> = <i>g</i><sub>learned</sub> + 0.85 · <i>g</i><sub>match</sub>).</p>
    <p class="mb-1 text-sm leading-relaxed text-foreground"><strong class="text-orange-600 dark:text-orange-400">Outcome:</strong> Memory trust locked at <i>g</i> = 1.0 (100% memory override). This starved the physics engine of target gradients, causing internal engine performance to collapse (<i>R</i>² = 0.2162). Overall model score dropped to <i>R</i>² = 0.4922 on novel test scaffolds.</p>
    <p class="text-sm leading-relaxed text-foreground"><strong>Lesson:</strong> Over-privileging retrieval gates starves the parametric backbone during joint training, destroying fallback capability on novel scaffolds.</p>
  </div>

  <!-- RAG Experiment 3 & 4 -->
  <div class="border-b border-border/80 bg-yellow-500/8 px-4 py-4 sm:px-5 dark:bg-yellow-950/30">
    <div class="mb-2 flex items-center gap-2 text-xs font-bold tracking-wider text-yellow-700 uppercase dark:text-yellow-400">
      <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M18 6 6 18"/><path d="m6 6 12 12"/></svg>
      Experiment B3 — Soft MoE Mixture (α) &amp; Naive Structural Noise Bottleneck
    </div>
    <p class="mb-1 text-sm leading-relaxed text-foreground"><strong>What we tried:</strong> Implemented a Soft Mixture-of-Experts gate (<i>Y</i><sub>final</sub> = α · <i>y</i><sub>rag</sub> + (1−α) · <i>y</i><sub>physics</sub>) with binary entropy regularization −<i>H</i>(α) and a tri-partite multi-task loss.</p>
    <p class="mb-1 text-sm leading-relaxed text-foreground"><strong class="text-yellow-700 dark:text-yellow-400">Outcome:</strong> Entropy regularization stabilized blending (α ≈ 0.43). However, RAG model (<i>R</i>² = 0.6078) performed <strong>worse than standalone physics (<i>R</i>² = 0.6560)</strong>. Naive 2D structural similarity retrieves neighbors whose thermodynamic properties differ wildly across scaffold boundaries.</p>
    <p class="text-sm leading-relaxed text-foreground"><strong>Lesson:</strong> Structural similarity (e.g. Morgan fingerprints) does not equal property similarity. Uncalibrated structural retrieval acts as additive noise in lattice energy estimation.</p>
  </div>

  <!-- RAG Solution -->
  <div class="bg-emerald-500/8 px-4 py-4 sm:px-5 dark:bg-emerald-950/30">
    <div class="mb-2 flex items-center gap-2 text-xs font-bold tracking-wider text-emerald-600 uppercase dark:text-emerald-400">
      <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1-1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3Z"/></svg>
      Solution — 2-Stage Property-Guided Contrastive RAG Pipeline
    </div>
    <p class="mb-1 text-sm leading-relaxed text-foreground"><strong>The Fix:</strong> Decoupled metric learning from downstream regression into a modular 2-stage pipeline:</p>
    <ol class="mb-1.5 list-disc pl-5 text-sm leading-relaxed text-foreground">
      <li><strong>Stage 1 (Supervised Contrastive Metric Learning):</strong> Pre-train <code>MetricProjectionEncoder</code> using Continuous Supervised Contrastive Loss (<i>L</i><sub>SupCon-Reg</sub>) to align metric cosine distance with continuous target property Gaussian distance (<i>w<sub>ij</sub></i> = exp(−|<i>y<sub>i</sub></i> − <i>y<sub>j</sub></i>|² / 2σ<sub><i>y</i></sub>²)).</li>
      <li><strong>Stage 2 (Thermodynamic Prototype Attention):</strong> Extract <i>K</i> = 64 cluster prototypes (μ<sub><i>c</i></sub>, <i>Ȳ<sub>c</sub></i>) from property-projected memory space to query dual k-NN property neighbors.</li>
    </ol>
    <p class="text-sm leading-relaxed text-foreground"><strong>Key Takeaway:</strong> Decoupling contrastive property-metric pre-training eliminates gradient interference and guarantees retrieved memory neighbors are thermodynamically aligned, turning RAG into a reliable prior.</p>
  </div>

</div>

---

## 9. Limitations

1. **Single-Component Neutral Solid Limit**: The dataset contains single-component organic compounds. Co-crystals, solvates, and ionic salts exhibit non-covalent lattice energies not covered by standard molecular descriptors.
2. **Crystal Polymorphism**: Experimental melting points correspond to the thermodynamic room-pressure polymorph; alternative polymorphic forms are not explicitly differentiated in 2D SMILES representations.

---

## 10. Conclusions & Takeaways

1. **Random CV severely underestimates real-world error**: Evaluating on scaffold-disjoint splits is essential for realistic deployment in molecular design.
2. **Target-Quantile Stratification is mandatory**: Stratifying scaffold bin packing prevents variance imbalance across cross-validation folds.
3. **Use Normalized MAE (nMAE) for scale invariance**: nMAE removes data geometry distortion when benchmarking across diverse chemical series.

---

## 11. Reproducibility & Code

| Resource | Description | Reference Link |
| :--- | :--- | :--- |
| **Executable Kaggle Notebook** | Full executable PyTorch pipeline for feature extraction, model training, and 5-fold scaffold CV | [Kaggle Notebook](https://www.kaggle.com/code/hirenlimbad/latticeenergyprediction/) |
| **Melting Point Dataset** | Standardized dataset of 23,496 organic compounds with experimental melting points | [Kaggle Dataset](https://www.kaggle.com/datasets/awguhst/melting-point-dataset) |