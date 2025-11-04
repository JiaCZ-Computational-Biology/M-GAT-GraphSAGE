# M-GAT-GraphSAGE

---

## 🧪 Description

- **data**: Contains all datasets used in the experiments (full dataset, training set, validation set, independent test set).  
- **train.py**: Final model training script.  
- **test.py**: Independent testing script.  
- **ablation**: Includes all ablation experiment code and related materials.  
- **gnn**: Contains implementations of the investigated graph neural networks.  
- **fingerprint**: Includes all molecular fingerprints used for experimental comparisons.  

---

## ⚙️ Python Environment


python      == 3.8.18
numpy       == 1.24.4
pandas      == 2.0.3
pillow      == 10.4.0
torch       == 2.4.1
catboost    == 1.2.8
function    == 1.2.0
pycaret     == 3.2.0
xgboost     == 2.1.4

---

## ⚙️ Experimental Procedures

Step-by-step Procedure for Stratified Sampling of Representative Molecules
Step 0. Inputs and Target
For each molecule i (from the test CSV), we pre-compute via a lightweight gradient-based pass: 
oindex_i: row index in the original CSV
oprediction_i: graph-level model output
oavg_importance_i: mean node importance (average L2 norm of gradients across atoms)
onum_atoms_i: number of atoms in the molecule
Aggregate the results into a DataFrame U with columns [index, prediction, avg_importance, num_atoms].
Set the target sample size n = 200 for subsequent “detailed interpretability analysis.”
Step 1. Quota Allocation across Three Dimensions
Prediction quota: T_pred = floor(0.40 × n) = 80
Importance quota: T_imp = floor(0.30 × n) = 60
Size quota: T_size = floor(0.20 × n) = 40
Residual slots: n − (T_pred + T_imp + T_size) ≈ 20, reserved for final random completion.
Rationale: priorities reflect our goals—first cover model behavior, then interpretability strength, and finally structural scale.
Step 2. Stratify and Sample by Prediction (First Dimension)
Current candidate pool R = U (no samples selected yet).
Apply 5-quantile binning (q = 5) to R['prediction']: Q1,…,Q5 (pandas.qcut with labels=False, duplicates='drop').
Per-bin allocation: m_pred = floor(T_pred / 5) = 16.
For each bin Qk: 
oLet Ck denote all samples in bin Qk.
oIf |Ck| ≥ 16: uniformly sample 16 entries (random_state=42 for reproducibility).
oIf |Ck| < 16: take all entries in Ck (no replacement).
Add selected indices to the set S.
Outcome: approximately 80 samples spanning the full prediction range (low–mid–high).
Corner Cases for Step 2
If qcut produces fewer than 5 bins due to many ties: still distribute T_pred as evenly as possible across the available bins; if a bin is undersized, apply the “take all” rule.
If qcut fails entirely (rare): fallback to simple random sampling of T_pred entries from R (fixed seed).
Step 3. Stratify and Sample by Average Importance (Second Dimension)
Update the remaining pool: R = U \ S (exclude those already selected by prediction).
Apply 5-quantile binning to R['avg_importance']: P1,…,P5.
Per-bin allocation: m_imp = floor(T_imp / 5) = 12.
Apply the same rule as in Step 2 (“uniformly sample up to quota; if insufficient, take all”), and add samples to S.
Outcome: approximately 60 additional samples (actual number may be lower depending on residual pool size and bin populations), covering the full spectrum of interpretability signal strength.
Note: Sampling only from the remaining pool reduces overlap with the first dimension and improves joint coverage.
Step 4. Stratify and Sample by Molecular Size (Third Dimension)
Update the remaining pool: R = U \ S.
Apply 5-quantile binning to R['num_atoms']: A1,…,A5.
Per-bin allocation: m_size = floor(T_size / 5) = 8.
Apply the same rule and add selected samples to S.
Outcome: approximately 40 additional samples spanning molecular sizes from small to large.
Note: As with Step 3, operating on the remaining pool further minimizes overlap with prior dimensions.
Step 5. Random Completion to Target Size
If |S| < n (common when bins are undersized and taken in full): 
oFrom the remaining pool R = U \ S, uniformly sample n − |S| entries (random_state=42).
oAdd them to S to reach |S| = n.
Motivation: ensures inclusion of rare or previously overlooked samples, enhancing overall chemical diversity.
Step 6. Reporting Coverage
Return S (a list of 200 indices).
Compute and report coverage ranges along the three axes as quantitative evidence: 
oPrediction range: [min_S(prediction), max_S(prediction)]
oImportance range: [min_S(avg_importance), max_S(avg_importance)]
oSize range: [min_S(num_atoms), max_S(num_atoms)]
The results of this experiment：
Prediction distribution:
 Range: 0.160 - 0.201
Importance distribution:
  Range: 0.001 - 0.009
Molecule size distribution:
  Range: 11 - 94 atoms
