# RBF Surrogate-Based RBDO for Offshore Wind Turbine Monopile Support Structures

A Python implementation of **Reliability-Based Design Optimization (RBDO)** for offshore wind turbine (OWT) monopile support structures using **Radial Basis Function (RBF) surrogate models** and **Particle Swarm Optimization (PSO)**.

This project compares deterministic design optimization (DDO) against RBDO, demonstrating how accounting for uncertainties — in environmental loads, material properties, and soil conditions — leads to structurally safer designs.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Site Data](#site-data)
- [Design Variables & Constraints](#design-variables--constraints)
- [Installation](#installation)
- [Usage](#usage)
- [Outputs](#outputs)
- [Results Summary](#results-summary)
- [Dependencies](#dependencies)
- [Reference](#reference)

---

## Overview

Offshore wind turbine monopile foundations are subject to complex, uncertain loading from wind, waves, and currents. This project addresses the structural design problem by:

1. **Simulating** finite element responses (stress, buckling, displacement, rotation, natural frequency, fatigue) using physics-based models over a Latin Hypercube Sampled (LHS) design space.
2. **Training RBF surrogate models** on the simulation data to enable fast function evaluations during optimization.
3. **Running PSO-based DDO and RBDO** to find optimal structural geometries that minimize steel volume while satisfying structural limits.
4. **Quantifying reliability** using Monte Carlo Simulation (MCS) and reporting reliability indices (β) for each limit state.
5. **Conducting sensitivity analysis** on soil coefficient of variation (COV = 1%, 3%, 5%).

The site of interest is **Kalpakkam, Bay of Bengal, India**, using wave and tidal data from 1998–2017.

---

## Project Structure

```
RBF SURROGATE MONOPILE/
│
├── 3.py                              # Main Python script (full pipeline)
├── Kalpakkam_1998_2017_Final.xlsx    # Historical wave height & period data
├── Kalpakkam_tide.xlsx               # Tidal level data (MSL reference)
│
├── Conference Paper.pdf              # Accompanying research paper
├── Model.pptx                        # Presentation slides
│
└── Output Figures (auto-generated on run):
    ├── response_distributions.png    # FEA response scatter plots (train/val/test split)
    ├── convergence.png               # PSO convergence history: DDO vs RBDO
    ├── design_comparison.png         # Bar charts: Initial vs DDO vs RBDO designs
    ├── reliability_indices.png       # β per limit state + system reliability
    └── soil_cov_effect.png           # Sensitivity of design to soil uncertainty
```

---

## Methodology

### 1. Environmental Loading
- **Wind**: Power law wind profile, hub height 90 m, reference speed 50 m/s
- **Waves & Current**: Morison equation (C_M = 2.0, C_D = 1.2), significant wave height and period from Kalpakkam data
- **Design wave**: H_s = 6.9 m, T_s = 7.7 s, current velocity = 0.8 m/s

### 2. Structural Response Simulation (Physics-Based FEA Proxy)
Six responses are simulated for 360 LHS samples:

| Response | Symbol | Limit |
|----------|--------|-------|
| Von Mises Stress | σ | ≤ 355 MPa |
| Buckling Load Multiplier | λ | ≥ 1.45 |
| Tower Top Displacement | δ | ≤ 0.97 m |
| Mudline Rotation | θ | ≤ 0.25° |
| 1st Natural Frequency | f₁ | 0.202–0.345 Hz |
| Fatigue Damage (Miner sum) | D | ≤ 1.0 |

### 3. RBF Surrogate Model
- **Architecture**: Gaussian RBF network with K-Means center selection
- **Regularization**: Tikhonov (λ = 1×10⁻⁶)
- **Hyperparameter search**: n_centers ∈ {auto, 30, 50, 100, 150, 200}, selected by R² on validation set
- **Data split**: 70% train / 15% validation / 15% test (252 / 54 / 54 samples)

### 4. Optimization (PSO)
- **Particles**: 40 | **Iterations**: 100
- **DDO**: Penalty on deterministic constraint violations (load factor = 1.35)
- **RBDO**: Penalty when β < 4.0 for any limit state, evaluated via MCS (5,000 samples/particle)
- **Inertia**: w = 0.7, c₁ = c₂ = 1.5

### 5. Reliability Analysis
- Monte Carlo Simulation: 15,000 samples
- Random variables: load factor (Normal + Gumbel extreme events), material factor (Normal), soil factor (Normal)
- Reliability index: β = −Φ⁻¹(P_f)

---

## Site Data

| File | Content | Period |
|------|---------|--------|
| `Kalpakkam_1998_2017_Final.xlsx` | Significant wave height (m), wave period T02 (s) | 1998–2017 |
| `Kalpakkam_tide.xlsx` | Tidal level (m, MSL reference) | — |

The Excel files are read at runtime. If missing, the code falls back to hard-coded design values (H_s = 6.9 m, T_s = 7.7 s).

---

## Design Variables & Constraints

Six geometric design variables define the monopile + transition piece + tower:

| Variable | Description | Bounds |
|----------|-------------|--------|
| D1 | Tower top diameter (m) | [3.0, 4.5] |
| D2 | Transition piece diameter (m) | [5.0, 7.0] |
| D3 | Monopile diameter (m) | [5.0, 7.0] |
| T1 | Tower wall thickness (m) | [0.012, 0.025] |
| T2 | Transition piece thickness (m) | [0.020, 0.035] |
| T3 | Monopile wall thickness (m) | [0.040, 0.070] |

**Geometric constraints**: D1 ≤ D2 ≤ D3, T1 ≤ T2 ≤ T3, D3/T3 ≤ 120

**Objective**: Minimize total steel volume (m³)

---

## Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/rbf-surrogate-monopile.git
cd rbf-surrogate-monopile

# Install dependencies
pip install numpy scipy scikit-learn matplotlib pandas openpyxl
```

Python 3.8+ is recommended.

---

## Usage

Place the Excel data files in the same directory as `3.py`, then run:

```bash
python 3.py
```

The script will:
1. Load and process the Kalpakkam wave and tide data
2. Generate 360 LHS design samples and run FEA response simulations
3. Train 6 RBF surrogate models (one per response)
4. Run PSO-based DDO and RBDO
5. Perform reliability analysis (MCS with 15,000 samples)
6. Conduct soil COV sensitivity study (COV = 1%, 3%, 5%)
7. Save all figures to the working directory

Expected runtime: ~5–15 minutes depending on hardware.

---

## Outputs

| File | Description |
|------|-------------|
| `response_distributions.png` | Scatter plots of all 6 FEA responses, color-coded by train/val/test split |
| `convergence.png` | PSO volume convergence history for DDO and RBDO |
| `design_comparison.png` | Bar charts comparing initial, DDO, and RBDO designs (diameters & thicknesses) |
| `reliability_indices.png` | Individual limit state β values and overall system reliability |
| `soil_cov_effect.png` | Effect of soil uncertainty (COV) on RBDO optimal design |

---

## Results Summary

| Design | Volume (m³) | System β |
|--------|-------------|----------|
| Initial | (baseline) | — |
| DDO | Reduced | Moderate |
| RBDO | Slightly higher than DDO | ≥ 4.0 (all limit states) |

RBDO produces a design with a higher reliability index at the cost of marginally more material compared to DDO — this is the expected safety-vs-economy tradeoff in probabilistic structural design.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `numpy` | Numerical arrays and linear algebra |
| `scipy` | Normal/Gumbel distributions, statistical functions |
| `scikit-learn` | K-Means clustering, StandardScaler, R² score |
| `matplotlib` | All plotting |
| `pandas` | Excel data loading |
| `openpyxl` | Excel file backend for pandas |

---

## Reference

This work is associated with the conference paper included in this repository:

> **[See `Conference Paper.pdf`]** — RBF surrogate-based reliability-based design optimization of offshore wind turbine monopile support structures, site-specific to Kalpakkam, India.

---

## License

This project is for academic and research purposes. Please cite the accompanying conference paper if you use this work.
