# DBMS-OB-Project: Orthogonal Bias (OB) + Sparse OB (SOB) for Fair ML

> A complete fairness-focused ML pipeline built on three datasets:
> **Adult Income**, **COMPAS Recidivism**, and **Loan Approval**.
>
> This project demonstrates how to:
> 1) **Remove sensitive information** from features using **Orthogonal Bias (OB)** projection
> 2) Build **Sparse SOB** feature sets via **L1-regularized selection**
> 3) Evaluate with **Accuracy + AUC** and multiple **fairness metrics**, including a **CF (Counterfactual) metric**
> 4) Explore an **Extension**: *Adaptive Fairness State (AFS) → AFS-SOB*

---

## Repository Contents

- **Datasets**
  - `adult.csv` (Adult Income)
  - `compas-scores-two-years.csv.zip` + extracted folder `compas-scores-two-years.csv/`
  - `loan_approval_dataset.csv` (Loan Approval)

- **Notebooks (end-to-end experiments)**
  - `Project.ipynb` – Adult Income: OB → SOB → baselines → CF/EO/AA metrics + extension (AFS)
  - `ProjectExtension.ipynb` – additional / extended fairness experiments (AFS-SOB style)
  - `LoanAppExtensio.ipynb` – Loan Approval: OB → SOB → fairness metrics + extension
  - `compas-scores-two-years.csv/Compas-prediction.ipynb` – COMPAS: OB → SOB → baselines → fairness metrics

- **Already provided**
  - `README.md` (this file)

---

## Key Idea: Orthogonal Bias (OB)

Many models learn spurious correlations between non-sensitive features and **sensitive attributes**.

This project uses a linear projection to construct “fairer” features:

\[
X_{fair} = X - S(S^TS)^{-1}S^T X
\]

Where:
- `X` = original features
- `S` = sensitive attributes (e.g., **sex & race**; or **income_annum/education/self_employed** depending on dataset setup)
- `X_fair` = residual component orthogonal to the sensitive subspace

---

## Sparse OB (SOB): Selecting Useful Fair Features

OB can keep many features. To get a compact model, the project applies an **SOB approximation**:

1. Standardize features (`StandardScaler`) to get `X_std`, `S_std`
2. Compute the OB projection in standardized space to obtain `X_ob`
3. Fit an **L1-regularized Logistic Regression** selector on `X_ob`
4. Keep only features with non-zero coefficients:

- Selected mask:
  - `important = abs(coef_) > 1e-6`
- Final sparse fair representation:
  - `X_sob = X_ob[:, important]`

---

## Models Used

Across notebooks, the core learning steps typically include:
- **Baselines**
  - Logistic Regression (sometimes trained on `X+S` vs `X`-only FTU)
- **Final predictors**
  - RandomForestClassifier
  - LogisticRegression (for selector)
  - XGBoostClassifier (in COMPAS experiment)

---

## Fairness Evaluation Metrics

The experiments report both **utility** and **fairness**.

### 1) Utility
- **ACC**: Accuracy
- **AUC**: ROC-AUC (when probabilities are available)

### 2) CF (Counterfactual) Metric
The notebooks implement a CF-style signal:

\[
CF = \frac{1}{n}\sum_{i=1}^n \left|\hat{y}_i - \hat{y}_i^{cf}\right|
\]

Implementation detail used in the notebooks:
- Build a simple counterfactual by **flipping one sensitive feature** (e.g., gender/sex)
- Recompute predicted probabilities
- Take the mean absolute difference

Lower CF → closer predictions under counterfactual change.

### 3) EO Fairness (Equal Opportunity gap)
- Uses **TPR difference** between sensitive groups.

### 4) AA Fairness (Average Odds gap proxy)
- Uses difference in **positive prediction rates** across groups.

All notebooks aggregate these into a results table like:
- `Method, ACC, AUC, CF-Metric, EO Fairness, AA Fairness`

---

## Extension: Adaptive Fairness State (AFS) → AFS-SOB

Some notebooks add an iterative extension that maintains a **fairness state** updated using observed group gaps.

High-level behavior:
1. Initialize `fairness_state` from standardized sensitive subspace `S_std`
2. Repeat for `T` iterations:
   - Project features using an OB-like residual with current fairness state
   - Train a model, predict labels
   - Measure fairness gaps (e.g., gender/race gaps)
   - Update state using `gamma` smoothing
3. Apply L1 feature selection again to produce **AFS-SOB**
4. Evaluate final ACC/AUC/CF/EO/AA metrics

---

## How to Run (No execution performed here)

Open the notebook(s) in Jupyter / VSCode and run cells in order:

### Adult Income
- `Project.ipynb`

### COMPAS
- `compas-scores-two-years.csv/Compas-prediction.ipynb`

### Loan Approval
- `LoanAppExtensio.ipynb`

> If your environment is missing `xgboost`, install it before running the COMPAS notebook.

---

## Project Notes / Practical Tips

- **Label encoding** is performed for object/string columns via `LabelEncoder`.
- The OB projection uses `np.linalg.pinv` (pseudoinverse) for stability.
- SOB feature selection is based on **L1 coefficients** from Logistic Regression.
- Fairness metrics assume sensitive attributes can be grouped into binary (or at least two-group) comparisons.

---

## Outcomes (What you should observe)

Across datasets and methods, you should expect:
- **FTU (feature-only) vs OB vs SOB** trade-offs between:
  - predictive performance (ACC/AUC)
  - fairness gaps (CF/EO/AA)
- SOB typically yields a **more compact** fair feature set than dense OB features.

---

## Files Quick Map

- **OB → SOB pipeline**
  - Implemented repeatedly in:
    - `Project.ipynb`
    - `LoanAppExtensio.ipynb`
    - `compas-scores-two-years.csv/Compas-prediction.ipynb`

- **CF/EO/AA metrics**
  - Implemented in:
    - `Project.ipynb`
    - `LoanAppExtensio.ipynb`
    - `compas-scores-two-years.csv/Compas-prediction.ipynb`

- **AFS extension**
  - Implemented in:
    - `Project.ipynb` and `ProjectExtension.ipynb` (observed from content)
    - `LoanAppExtensio.ipynb`

---

## License

Add your license here (e.g., MIT/Apache-2.0) if you plan to publish the project.

