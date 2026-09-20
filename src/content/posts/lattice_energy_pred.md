---
author: Levitating Labs
pubDatetime: 2026-08-30T12:00:00Z
title: "From 0.80 to 0.66 R²: How Scaffold Shifts Impact Physics-Guided Lattice Energy Models"
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

While molecular machine learning models achieve strong performance on random train/test splits (our pure physics baseline reaches **0.7983 R²** with a $dH\text{ MAE}$ of **1.723 kJ/mol**), traditional evaluation schemes mask severe degradation when deployed against novel chemical core topologies. To quantify this out-of-distribution (OOD) generalization gap, we benchmark `LatticePhysicsTower`—a physics-guided neural network combining 2,243-dimensional multimodal descriptors with thermodynamic mapping—across target-quantile <mark>stratified Bemis-Murcko scaffold partitions</mark> on 23,496 organic compounds from the Bradley Open Melting Point Dataset.

---

## 1. Core Hypotheses & Key Findings

<div class="my-6 flex flex-col overflow-hidden rounded-xl border border-border/80 shadow-xs">
  <!-- H1: Random Split Inflation (Slate/Gray Container) -->
  <div class="border-b border-border/80 bg-slate-500/10 px-4 py-3.5 sm:px-5 sm:py-4 dark:bg-slate-900/40">
    <div class="mb-1.5 flex items-center gap-2 text-xs font-bold tracking-wider text-slate-600 uppercase dark:text-slate-400">
      <svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="m4.93 4.93 14.14 14.14"/></svg>
      Hypothesis 1: Random Split Performance Inflation
    </div>
    <p class="text-sm font-medium leading-relaxed text-foreground">
      <strong>Question:</strong> Do standard random cross-validation splits overestimate model performance for molecular lattice energy and melting point prediction?<br/>
      <strong class="text-rose-600 dark:text-rose-400">Outcome (Rejected):</strong> <strong>Yes.</strong> Random splits yield an inflated <strong>0.7983 R²</strong> (dH MAE = 1.723 kJ/mol, Tm MAE = 30.50 K). When evaluated on a Bemis-Murcko scaffold-disjoint test set, performance drops to <strong>0.6570 R²</strong>—exposing a <strong>14.1% structural generalization gap</strong> ($\Delta R^2 = -0.1413$).
    </p>
  </div>

  <!-- H2: Stratified Scaffold CV & nMAE Stabilization (Teal/Amber Container) -->
  <div class="bg-teal-500/10 px-4 py-3.5 sm:px-5 sm:py-4 dark:bg-teal-950/40">
    <div class="mb-1.5 flex items-center gap-2 text-xs font-bold tracking-wider text-teal-600 uppercase dark:text-teal-400">
      <svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3Z"/></svg>
      Hypothesis 2: Stratified Scaffold Shift & Scale-Invariant Metric Rescue
    </div>
    <p class="text-sm font-medium leading-relaxed text-foreground">
      <strong>Question:</strong> Can target-quantile scaffold stratification stabilize 5-fold cross-validation, and how should we evaluate error across chemical families with varying energy distributions?<br/>
      <strong class="text-emerald-600 dark:text-emerald-400">Outcome (Confirmed):</strong> <strong class="text-accent">Yes. Target-quantile scaffold bin packing stabilizes 5-fold cross-validation at 0.6215 ± 0.0253 R². Introducing Normalized MAE ($\text{nMAE} = \text{MAE}/\sigma_{y,\text{fold}}$) resolves data geometry artifacts, yielding a scale-invariant error of 0.4729 ± 0.0136 across all scaffold folds.</strong><br/>
      <strong>Why it worked:</strong> Scaffold groups differ in baseline energy variance ($\sigma_y$). Normalizing MAE by fold standard deviation removes scale distortion, proving that relative model precision remains stable across diverse chemical scaffold topologies.
    </p>
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
> Built from 23,496 valid, standardized compounds in the Bradley Open Melting Point Dataset.
> - **Input Feature Dimension:** 2,243 multimodal features
> - **Target Enthalpy Mean:** $21.46\text{ kJ/mol}$ ($379.8\text{ K}$)
> - **Loss Function:** Smooth Huber Loss ($\delta = 5.0$)
> - **Optimizer & Scheduler:** AdamW ($\text{lr}=10^{-3}$, $\text{wd}=10^{-3}$), Cosine Annealing, EMA ($\beta=0.999$)

> [!WARNING] Evaluation Protocols
> 1. **RANDOM SPLIT**: Standard 80/20 uniform random partition (In-Distribution baseline).
> 2. **BEMIS-MURCKO SCAFFOLD SPLIT**: 80/20 core framework disjoint split (Single-fold OOD check).
> 3. **5-FOLD STRATIFIED SCAFFOLD CV**: 5-fold cross-validation with target-quantile stratification across scaffold clusters.

### Model Architecture (`LatticePhysicsTower`)

![Figure: Model Architecture & Multimodal Feature Pipeline](@/assets/images/lattice_energy_pred/model_architecture.png)
*Figure 1: Overview of the LatticePhysicsTower architecture featurizing 2,243-D multimodal molecular vectors.*

- **Input Featurization (2,243 Dimensions)**:
  - **2,048-bit Morgan Circular Fingerprints ($r=2$)**: Structural fragment presence.
  - **167-bit MACCS Structural Keys**: Functional group subgraph patterns.
  - **12 Physical Descriptors**: MolWt, LogP, TPSA, HBD, HBA, Rotatable Bonds, Aromatic Rings, Net Charge.
  - **4 Topological Symmetry Indices**: Graph automorphism orbits and symmetry metrics.
  - **4 Crystal Packing Motifs**: H-bond donor/acceptor stoichiometric ratios and rigid ring fractions.
  - **8 3D Conformer Descriptors**: Principal Moments of Inertia ($I_a, I_b, I_c$), Asymmetry Parameters, and Radius of Gyration computed via ETKDGv3 3D conformers.
- **Network Processing (`LatticePhysicsTower`)**:
  - `Linear(2243 -> 64) -> SiLU() -> Dropout(0.2)`
  - `Linear(64 -> 64) -> SiLU()`
  - `Linear(64 -> 1) + Global Mean Parameter Initializer (21.46 kJ/mol)`
- **Optimization & Regularization**:
  - Trained using Smooth Huber Loss ($\delta = 5.0$) to handle extreme thermal outliers.
  - Exponential Moving Average (EMA decay $\beta = 0.999$) maintained for evaluation stability.

---

## 4. Benchmark Performance Results

The performance below summarizes the **Stratified 5-Fold Cross-Validation** and **Random vs. Scaffold Split Comparison** obtained from the experiment execution.

### Consolidated Publication Experimental Summary Table

| Experiment / Evaluation Scheme | R² Score | dH MAE (kJ/mol) | nMAE | dH RMSE (kJ/mol) | Tm MAE (K) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **5-Fold Stratified Scaffold CV (Mean ± Std)** | **0.6215 ± 0.0253** | **2.212 ± 0.423** | **0.4729 ± 0.0136** | **2.874 ± 0.529** | **39.15 ± 7.49 K** |
| **Random Split (In-Distribution)** | **0.7983** | **1.723** | **0.3315** | **2.334** | **30.50 K** |
| **Bemis-Murcko Scaffold Split (OOD)** | **0.6570** | **1.793** | **0.4518** | **2.324** | **31.73 K** |

### Detailed Random vs. Scaffold Generalization Gap

| Splitting Scheme | R² Score | dH MAE (kJ/mol) | nMAE | Tm MAE (K) |
| :--- | :---: | :---: | :---: | :---: |
| **Random Split (In-Distribution)** | 0.7983 | 1.723 | 0.3315 | 30.50 K |
| **Bemis-Murcko Scaffold Split (OOD)** | 0.6570 | 1.793 | 0.4518 | 31.73 K |
| **Scaffold Shift Generalization Drop ($\Delta$)** | **-0.1413** | **+0.070** | **+0.1203** | **+1.23 K** |

---

## 5. Benchmark Figures

![Figure 1: Scaffold CV Summary Metrics Across Folds](@/assets/images/lattice_energy_pred/figure_scaffold_cv_summary.png)
*Figure 1: Summary of R², MAE, nMAE, and RMSE metrics across 5 Stratified Bemis-Murcko Scaffold Folds.*

![Figure 2: Random vs. Scaffold Parity & Residual Scatter Plots](@/assets/images/lattice_energy_pred/random_vs_scaffold_scatter.png)
*Figure 2: Parity comparison between In-Distribution Random Split and OOD Scaffold Split predictions.*

![Figure 3: Scale-Invariant nMAE Stability Profile](@/assets/images/lattice_energy_pred/nmae_variance_stability.png)
*Figure 3: Demonstration of Normalized MAE (nMAE) stability across heterogeneous target standard deviations.*

---

## 6. Findings & Analysis

1. **Quantifying the 14.1% OOD Generalization Penalty**: Random train/test splits yield an inflated **0.7983 R²**, whereas holding out core scaffolds drops performance to **0.6570 R²**. Over 14% of random split accuracy stems from memorizing core scaffold topologies rather than learning transferrable physical principles.
2. **Scale-Invariant Metric Stabilization via nMAE**: Raw MAE varies from $1.78\text{ kJ/mol}$ to $2.78\text{ kJ/mol}$ across folds due to baseline target variance differences ($\sigma_y$). Evaluating via Normalized MAE ($\text{nMAE} = 0.4729 \pm 0.0136$) confirms that relative model error is stable across chemical families.
3. **Capacity Regularization Prevents Scaffold Overfitting**: Bottleneck architectures ($2243 \to 64 \to 64 \to 1$) provide optimal structural regularization, preventing wide networks from memorizing complex fragment combinations tied strictly to training scaffolds.

---

## 7. Limitations

1. **Single-Component Neutral Solid Limit**: The dataset contains single-component organic compounds. Co-crystals, solvates, and ionic salts exhibit non-covalent lattice energies not covered by standard molecular descriptors.
2. **Crystal Polymorphism**: Experimental melting points correspond to the thermodynamic room-pressure polymorph; alternative polymorphic forms are not explicitly differentiated in 2D SMILES representations.

---

## 8. Conclusions & Takeaways

1. **Random CV severely underestimates real-world error**: Evaluating on scaffold-disjoint splits is essential for realistic deployment in molecular design.
2. **Target-Quantile Stratification is mandatory**: Stratifying scaffold bin packing prevents variance imbalance across cross-validation folds.
3. **Use Normalized MAE (nMAE) for scale invariance**: nMAE removes data geometry distortion when benchmarking across diverse chemical series.

---

## 9. Reproducibility & Artifact Links

| Resource | Description | Path / Reference |
| :--- | :--- | :--- |
| **Primary Pipeline Notebook** | Executable PyTorch code for feature extraction, training, and CV | [`latticeenergyprediction.ipynb`](file:///d-drive/research/work/lattice_energy/latticeenergyprediction.ipynb) |
| **5-Fold CV Metrics CSV** | Exported metric metrics across all 5 scaffold folds | `artifacts/metrics/scaffold_5fold_cv_metrics.csv` |
| **Random vs Scaffold CSV** | Comparative metric evaluation export | `artifacts/metrics/random_vs_scaffold_metrics.csv` |
| **Publication Summary Table** | Consolidated summary markdown table | `artifacts/metrics/publication_summary_table.md` |
| **Summary Plot Figure** | Master multi-panel publication summary plot | `artifacts/plots/figure_scaffold_cv_summary.png` |