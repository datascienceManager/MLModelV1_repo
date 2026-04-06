Your README - Across the 12 sections:

**Sections 1–3** cover package setup (`randomForest`, `ranger`, `caret`, `ROSE`, `pROC`), data loading with exploratory checks, and full feature engineering including the stratified train/test split.

**Section 4** builds the model with both `randomForest` and `ranger` (the fast C++ alternative), with all key arguments explained side by side.

**Section 5** covers evaluation — confusion matrix, AUC-ROC, Precision-Recall curve, and OOB error plot — with a clear note that accuracy is misleading for imbalanced churn data.

**Section 6** gives five imbalance strategies: `classwt`, ROSE, SMOTE, manual oversampling, and threshold tuning with an F1-optimisation loop to find the best cutoff automatically.

**Section 7** explains regularisation via `nodesize`, `maxnodes`, `mtry`, `sampsize`, plus pre-processing steps to remove near-zero variance and highly correlated features, and RFE.

**Section 8** goes under the hood — `getTree()` for individual tree structure, depth/node stats across all trees, feature importance (MDA and MDGini), proximity matrix with MDS plot, and partial dependence plots.

**Section 9** covers three tuning approaches: manual grid search using OOB error, `caret` cross-validation, and the built-in `tuneRF` automated search, plus a parallel `ranger` grid search.

**Sections 10–12** handle saving (with a metadata bundle), scoring new data with proper factor-level alignment, left-joining predictions back as a `"Churn"`/`"Active"` column, and a full quick-reference cheatsheet of function arguments and metrics.
