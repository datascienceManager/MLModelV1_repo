Your README - Across the 12 sections:

**Sections 1–3** cover package setup (`randomForest`, `ranger`, `caret`, `ROSE`, `pROC`), data loading with exploratory checks, and full feature engineering including the stratified train/test split.

**Section 4** builds the model with both `randomForest` and `ranger` (the fast C++ alternative), with all key arguments explained side by side.

**Section 5** covers evaluation — confusion matrix, AUC-ROC, Precision-Recall curve, and OOB error plot — with a clear note that accuracy is misleading for imbalanced churn data.

**Section 6** gives five imbalance strategies: `classwt`, ROSE, SMOTE, manual oversampling, and threshold tuning with an F1-optimisation loop to find the best cutoff automatically.

**Section 7** explains regularisation via `nodesize`, `maxnodes`, `mtry`, `sampsize`, plus pre-processing steps to remove near-zero variance and highly correlated features, and RFE.

**Section 8** goes under the hood — `getTree()` for individual tree structure, depth/node stats across all trees, feature importance (MDA and MDGini), proximity matrix with MDS plot, and partial dependence plots.

**Section 9** covers three tuning approaches: manual grid search using OOB error, `caret` cross-validation, and the built-in `tuneRF` automated search, plus a parallel `ranger` grid search.

**Sections 10–12** handle saving (with a metadata bundle), scoring new data with proper factor-level alignment, left-joining predictions back as a `"Churn"`/`"Active"` column, and a full quick-reference cheatsheet of function arguments and metrics.



#========= Ensemble Model ========

**Section 1** opens with a taxonomy diagram explaining the three families — Bagging, Boosting, and Stacking — and a table for when to use each method.

**Section 2–3** cover package setup (`randomForest`, `gbm`, `xgboost`, `lightgbm`, `glmnet`, `caretEnsemble`) and a three-way 70/15/15 train/validation/test split, which is critical for stacking to avoid leakage.

**Section 4** trains five individual base learners — Random Forest, GBM, XGBoost, LightGBM, and LASSO logistic regression — each with validation AUC printed for comparison.

**Section 5** is the heart of the guide — five ensemble strategies: Hard voting, Soft voting, Weighted average (with `optim()` to find optimal weights), full Stacking with out-of-fold predictions and a LASSO meta-learner, Blending (the faster variant), and `caretEnsemble` for automated stacking.

**Section 6** covers regularisation for every model type including L1/L2 in XGBoost and LightGBM, structural constraints in RF and GBM, and using `lambda.1se` instead of `lambda.min` for the meta-learner.

**Section 7** handles class imbalance across all models — `classwt`, `scale_pos_weight`, `is_unbalance`, ROSE, SMOTE, and per-model threshold optimisation.

**Section 8** digs under the hood — feature importance comparison across all five models, SHAP values via XGBoost, GBM partial dependence plots, and a calibration plot for the ensemble.

**Sections 9–11** cover parallel hyperparameter tuning (including `doParallel`), saving the entire ensemble as a single RDS bundle with metadata, and a complete scoring function that aligns factor levels, rebuilds the matrix, scores all five base models, applies all three ensemble strategies, and joins back `"Churn"`/`"Active"` predictions with `prob_stack` to the original dataset.

**Section 12** produces a side-by-side AUC comparison table and overlaid ROC curves for all models. **Section 13** is a full cheatsheet with a decision flowchart for choosing which ensemble strategy to use.
