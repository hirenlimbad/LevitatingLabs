---
title: "Experiment 02: Equivariant GNNs for Crystal Lattice Energy Prediction"
pubDatetime: 2026-09-18T10:30:00.000Z
modDatetime: 2026-09-21T16:00:00.000Z
status: "in-progress"
description: "Daily experiment log evaluating E(3)-equivariant Graph Neural Networks on periodic crystal structure unit cells."
tags: ["GNN", "Crystal-Structure", "Equivariance", "Materials-AI"]
relatedArticle: null
featured: true
---

## Day 1: Periodic Boundary Condition Setup
*18 Sep 2026*

- Configured periodic graph construction with $k$-nearest neighbor graphs based on fractional coordinates.
- Implemented $SO(3)$ spherical harmonic representations for directional edge features.

$$\vec{r}_{ij} = (\mathbf{x}_j + \mathbf{R} \cdot \mathbf{n}) - \mathbf{x}_i$$

```python
# Periodic edge vector calculation
def periodic_vector(pos1, pos2, lattice, cell_offsets):
    shift = torch.matmul(cell_offsets.float(), lattice)
    return (pos2 + shift) - pos1
```

---

## Day 3: Convergence Monitoring
*20 Sep 2026*

- Ran 50 epochs of training on **Materials Project 2024 subset**.
- Loss curve exhibits smooth decay, but validation error fluctuates around epoch 35 due to high force gradient magnitudes.

> **Next Steps:** Testing gradient clipping at $\|g\| \le 1.0$ and adding radial bessel basis functions to smooth atomic pair interactions at cutoff radius $r_c = 6.0\text{ \AA}$.

---

## Day 4: Current Status 🧪
*21 Sep 2026*

- Gradient clipping stabilized validation loss curve.
- Current validation MAE: **18.4 meV/atom**.
- Target checkpoint threshold for full article publication is $< 15.0\text{ meV/atom}$.
- Training continues on 4x A100 GPU cluster. Next update expected in 2 days when epoch 150 completes.
