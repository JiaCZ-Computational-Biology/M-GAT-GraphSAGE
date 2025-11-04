# 🧬 M-GAT-GraphSAGE

A hybrid molecular graph learning framework integrating **Graph Attention Networks (GAT)** and **GraphSAGE** for molecular property prediction and interpretability.

---

## 📁 Directory Structure

| Folder/File | Description |
|--------------|-------------|
| **`data/`** | Contains all datasets used in the experiments (full dataset, training set, validation set, and independent test set). |
| **`train.py`** | Final model training script. |
| **`test.py`** | Independent testing script. |
| **`ablation/`** | Includes all ablation experiment code and related materials. |
| **`gnn/`** | Contains implementations of the investigated graph neural networks. |
| **`fingerprint/`** | Includes all molecular fingerprints used for experimental comparisons. |

---

## ⚙️ Python Environment

| Library | Version |
|----------|----------|
| `python` | 3.8.18 |
| `numpy` | 1.24.4 |
| `pandas` | 2.0.3 |
| `pillow` | 10.4.0 |
| `torch` | 2.4.1 |
| `catboost` | 1.2.8 |
| `function` | 1.2.0 |
| `pycaret` | 3.2.0 |
| `xgboost` | 2.1.4 |

---

## 🧪 Experimental Procedures

### Step 0 — Inputs and Target

For each molecule *i* (from the test CSV), precompute:

| Symbol | Description |
|---------|-------------|
| `oindex_i` | Row index in the original CSV |
| `oprediction_i` | Graph-level model output |
| `oavg_importance_i` | Mean node importance (average L2 norm of gradients across atoms) |
| `onum_atoms_i` | Number of atoms in the molecule |

Aggregate all results into a DataFrame **U** with columns:  
`[index, prediction, avg_importance, num_atoms]`

Set target sample size:  
`n = 200` (for detailed interpretability analysis)

---

### Step 1 — Quota Allocation Across Dimensions

| Dimension | Formula | Samples | Purpose |
|------------|----------|----------|----------|
| **Prediction** | `floor(0.40 × n)` | 80 | Cover full prediction range |
| **Importance** | `floor(0.30 × n)` | 60 | Emphasize interpretability signal |
| **Molecule Size** | `floor(0.20 × n)` | 40 | Ensure structural diversity |
| **Residual** | Remaining | ~20 | Random completion for diversity |

*Rationale:* Prioritize model coverage, interpretability signals, and molecular scale diversity.

---

### Step 2 — Stratified Sampling by Prediction

- **Initial pool:** `R = U`  
- Apply **5-quantile binning** (`pandas.qcut`) on `R['prediction']`: `Q1 ... Q5`  
- Per bin allocation: `m_pred = floor(80 / 5) = 16`  
- For each bin:
  - If `|Ck| ≥ 16`: uniformly sample 16 entries (`random_state=42`)
  - If `|Ck| < 16`: take all entries (no replacement)

**Outcome:** ~80 samples covering the full prediction spectrum.

**Corner cases:**
- Fewer than 5 bins → evenly redistribute quota.
- Fallback: random sampling if binning fails.

---

### Step 3 — Stratified Sampling by Average Importance

- Remaining pool: `R = U \ S`
- Apply 5-quantile binning on `R['avg_importance']`
- Per-bin allocation: `m_imp = floor(60 / 5) = 12`
- Apply same rule as Step 2

**Outcome:** ≈60 additional samples, improving interpretability coverage.

---

### Step 4 — Stratified Sampling by Molecular Size

- Remaining pool: `R = U \ S`
- Apply 5-quantile binning on `R['num_atoms']`
- Per-bin allocation: `m_size = floor(40 / 5) = 8`
- Apply same rule as Step 2

**Outcome:** ≈40 samples spanning all molecular sizes.

---

### Step 5 — Random Completion to Target Size

If `|S| < n`:  
- From remaining pool `R = U \ S`, randomly sample `n − |S|` entries (`random_state=42`)  
- Add to **S** to ensure `|S| = n`

**Motivation:** Includes rare or underrepresented molecules, enhancing chemical diversity.

---

### Step 6 — Reporting Coverage

Report quantitative coverage ranges across all dimensions:

| Metric | Range |
|---------|-------|
| **Prediction** | 0.160 – 0.201 |
| **Importance** | 0.001 – 0.009 |
| **Molecule Size** | 11 – 94 atoms |

This confirms the final sample set evenly covers prediction confidence, interpretability signal, and molecular scale.

---

## 📊 Summary

| Category | Description |
|-----------|-------------|
| **Sampling Method** | Hierarchical stratified sampling |
| **Focus** | Model behavior, interpretability, structural variety |
| **Sample Size** | 200 representative molecules |
| **Verification** | Quantitative range analysis across three dimensions |

---

## 📚 Citation

If you use this repository or its methods, please cite appropriately (citation to be added).

---
