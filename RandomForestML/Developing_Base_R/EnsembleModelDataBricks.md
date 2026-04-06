# Ensemble Base Models on Databricks — Complete Guide
### Churn Prediction · Binary Output: `"Churn"` / `"Active"`
### Platform: Databricks · Language: R (SparkR / sparklyr) · MLflow Tracking

> **Stack:** `sparklyr` · `SparkR` · `randomForest` · `xgboost` · `lightgbm` · `glmnet` · `MLflow` · `Delta Lake` · Databricks AutoML

---

## Table of Contents

1. [Databricks Architecture Overview](#1-databricks-architecture-overview)
2. [Cluster Setup & Library Installation](#2-cluster-setup--library-installation)
3. [Connect to Spark & Load Data from Delta Lake](#3-connect-to-spark--load-data-from-delta-lake)
4. [Feature Engineering & Data Prep](#4-feature-engineering--data-prep)
5. [Base Model 1 — Random Forest (sparklyr)](#5-base-model-1--random-forest-sparklyr)
6. [Base Model 2 — GBM / Gradient Boosting (sparklyr)](#6-base-model-2--gbm--gradient-boosting-sparklyr)
7. [Base Model 3 — XGBoost (distributed via SparkR)](#7-base-model-3--xgboost-distributed-via-sparkr)
8. [Base Model 4 — LightGBM (sparklyr + MMLSpark)](#8-base-model-4--lightgbm-sparklyr--mmlspark)
9. [Base Model 5 — Logistic Regression / LASSO (sparklyr)](#9-base-model-5--logistic-regression--lasso-sparklyr)
10. [Handling Class Imbalance on Databricks](#10-handling-class-imbalance-on-databricks)
11. [Regularisation](#11-regularisation)
12. [Model Inspection — Under the Hood](#12-model-inspection--under-the-hood)
13. [Hyperparameter Tuning](#13-hyperparameter-tuning)
14. [MLflow — Track Every Experiment](#14-mlflow--track-every-experiment)
15. [Save & Register Models in MLflow Model Registry](#15-save--register-models-in-mlflow-model-registry)
16. [Predict on New Data & Join Back](#16-predict-on-new-data--join-back)
17. [Write Predictions to Delta Lake](#17-write-predictions-to-delta-lake)
18. [Quick-Reference Cheatsheet](#18-quick-reference-cheatsheet)

---

## 1. Databricks Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                      DATABRICKS WORKSPACE                           │
│                                                                     │
│  ┌─────────────┐   ┌──────────────┐   ┌──────────────────────────┐ │
│  │  Notebook   │   │   Jobs /     │   │   MLflow Tracking        │ │
│  │  (R kernel) │   │  Workflows   │   │   - Experiments          │ │
│  └──────┬──────┘   └──────┬───────┘   │   - Runs / Metrics       │ │
│         │                 │           │   - Artifacts / Models   │ │
│         └────────┬────────┘           └──────────────────────────┘ │
│                  │                                                  │
│         ┌────────▼────────┐                                         │
│         │  Spark Cluster  │  Driver + N Worker nodes                │
│         │  (sparklyr /    │  Scala/Java executor JVMs               │
│         │   SparkR)       │                                         │
│         └────────┬────────┘                                         │
│                  │                                                  │
│         ┌────────▼────────────────────────────────────────┐        │
│         │              DELTA LAKE (DBFS / S3 / ADLS)      │        │
│         │  - churn_data (raw)                              │        │
│         │  - churn_features (engineered)                  │        │
│         │  - churn_predictions (output)                   │        │
│         └──────────────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────────────────┘

DATA FLOW:
Delta Lake → sparklyr (Spark DataFrame) → ML Pipeline → MLflow → Delta Lake
                                ↓
                    Collect sample to R driver
                    for randomForest / xgboost / glmnet
                    (base R packages run on driver node only)
```

### Databricks vs plain R: key differences

| Concern | Plain R | Databricks R |
|---|---|---|
| Data storage | Local CSV / RDS | Delta Lake (DBFS / cloud storage) |
| Data size | RAM-limited | Petabyte-scale via Spark |
| ML training | Single node | Distributed (sparklyr ML) or driver-only (base R) |
| Model tracking | Manual saveRDS | MLflow (auto-logged) |
| Model serving | Manual scripts | MLflow Model Registry + REST endpoint |
| Scheduling | Cron / local | Databricks Jobs / Workflows |
| Package install | `install.packages()` | Cluster Libraries UI or `%pip install` |

---

## 2. Cluster Setup & Library Installation

### Step 1 — Create a cluster (UI or API)

```
Databricks UI → Compute → Create Cluster

Recommended settings:
  Databricks Runtime:  13.3 LTS ML  (includes MLlib, MLflow, XGBoost)
  Node type:           Standard_DS4_v2  (28 GB RAM, 8 cores)
  Min workers:         2
  Max workers:         8  (autoscaling)
  Auto-terminate:      60 minutes
```

> **Use an ML Runtime** (e.g. `13.3 LTS ML`) — it pre-installs MLflow, XGBoost, LightGBM, and Spark ML, saving you manual installs.

### Step 2 — Install R libraries on the cluster

```r
# In a Databricks R notebook cell:

# sparklyr and ML packages
install.packages(c(
  "sparklyr",       # Spark + R bridge
  "dplyr",          # data manipulation
  "randomForest",   # Random Forest (driver node)
  "glmnet",         # LASSO / Ridge
  "pROC",           # AUC-ROC
  "PRROC",          # AUC-PR
  "ROSE",           # imbalance handling
  "ggplot2",        # visualisation
  "carrier"         # package R functions for MLflow pyfunc
), repos = "https://cran.rstudio.com/")

# MLflow R client
install.packages("mlflow", repos = "https://cran.rstudio.com/")

# XGBoost (included in ML Runtime — verify version)
packageVersion("xgboost")

# LightGBM (install if not in runtime)
install.packages("lightgbm", repos = "https://cran.rstudio.com/")
```

> **Tip — Cluster Libraries (persistent):** Go to *Cluster → Libraries → Install New → CRAN* to install packages that survive cluster restarts. Notebook-level installs are lost when the cluster restarts.

### Step 3 — Verify Spark connection

```r
library(sparklyr)
library(dplyr)

sc <- spark_connect(method = "databricks")   # no host/token needed inside Databricks

# Confirm
spark_version(sc)    # e.g. "3.4.1"
cat("Connected to Spark:", spark_version(sc), "\n")
```

---

## 3. Connect to Spark & Load Data from Delta Lake

```r
library(sparklyr)
library(dplyr)

# ── Connect (inside Databricks notebook) ────────────────────────────
sc <- spark_connect(method = "databricks")

# ── Read from Delta Lake table ───────────────────────────────────────
df_spark <- spark_read_table(sc, name = "analytics.churn_data")

# ── Or read directly from DBFS / cloud path ──────────────────────────
df_spark <- spark_read_delta(sc, path = "/mnt/datalake/churn/raw/")

# ── Or read Parquet / CSV ────────────────────────────────────────────
df_spark <- spark_read_parquet(sc, name = "churn", path = "/mnt/datalake/churn/")
df_spark <- spark_read_csv(sc, name = "churn",
                           path   = "/mnt/datalake/churn/churn.csv",
                           header = TRUE, infer_schema = TRUE)

# ── Schema and class balance ─────────────────────────────────────────
sdf_schema(df_spark)
glimpse(df_spark)

df_spark %>%
  group_by(Churn) %>%
  summarise(n = n()) %>%
  mutate(pct = round(n / sum(n) * 100, 2)) %>%
  collect()

# ── Null check ───────────────────────────────────────────────────────
df_spark %>%
  summarise(across(everything(), ~ sum(as.integer(is.na(.))))) %>%
  collect()
```

---

## 4. Feature Engineering & Data Prep

```r
# ── 1. Cast label ────────────────────────────────────────────────────
df_spark <- df_spark %>%
  mutate(label = ifelse(Churn == "Yes", 1L, 0L))

# ── 2. Feature engineering (runs distributed on Spark cluster) ───────
df_spark <- df_spark %>%
  mutate(
    tenure_sq       = tenure * tenure,
    avg_charges     = TotalCharges / ifelse(tenure > 0, tenure, 1),
    high_value_flag = ifelse(MonthlyCharges > 70, 1L, 0L),
    charge_ratio    = MonthlyCharges / ifelse(TotalCharges > 0, TotalCharges, 1)
  )

# ── 3. Impute nulls (Spark Imputer) ──────────────────────────────────
imputer <- ft_imputer(
  sc,
  input_cols  = c("tenure", "MonthlyCharges", "TotalCharges",
                  "tenure_sq", "avg_charges"),
  output_cols = c("tenure_imp", "MonthlyCharges_imp", "TotalCharges_imp",
                  "tenure_sq_imp", "avg_charges_imp"),
  strategy    = "median"
)

imputer_model <- ml_fit(imputer, df_spark)
df_spark      <- ml_transform(imputer_model, df_spark)

# ── 4. String indexing for categoricals (Spark native) ───────────────
indexers <- list(
  ft_string_indexer(sc, "Contract",       "Contract_idx"),
  ft_string_indexer(sc, "InternetService","InternetService_idx"),
  ft_string_indexer(sc, "PaymentMethod",  "PaymentMethod_idx")
)

for (idx in indexers) {
  fit_idx  <- ml_fit(idx, df_spark)
  df_spark <- ml_transform(fit_idx, df_spark)
}

# ── 5. One-hot encode indexed columns ────────────────────────────────
encoder <- ft_one_hot_encoder(
  sc,
  input_cols  = c("Contract_idx", "InternetService_idx", "PaymentMethod_idx"),
  output_cols = c("Contract_ohe", "InternetService_ohe", "PaymentMethod_ohe")
)
encoder_fit <- ml_fit(encoder, df_spark)
df_spark    <- ml_transform(encoder_fit, df_spark)

# ── 6. Assemble feature vector ────────────────────────────────────────
feature_cols <- c(
  "tenure_imp", "tenure_sq_imp", "avg_charges_imp",
  "MonthlyCharges_imp", "TotalCharges_imp",
  "high_value_flag", "charge_ratio",
  "Contract_ohe", "InternetService_ohe", "PaymentMethod_ohe"
)

assembler <- ft_vector_assembler(
  sc,
  input_cols  = feature_cols,
  output_col  = "features"
)
df_spark <- ml_transform(assembler, df_spark)

# ── 7. Stratified train / val / test split ───────────────────────────
splits <- df_spark %>%
  sdf_random_split(train = 0.70, val = 0.15, test = 0.15, seed = 42)

train <- splits$train
val   <- splits$val
test  <- splits$test

cat("Train:", sdf_nrow(train),
    "| Val:", sdf_nrow(val),
    "| Test:", sdf_nrow(test), "\n")

# ── 8. Cache hot DataFrames in Spark memory ──────────────────────────
train <- sdf_register(train, "train_cached")
tbl_cache(sc, "train_cached")
val   <- sdf_register(val, "val_cached")
tbl_cache(sc, "val_cached")
```

> **Driver vs Cluster:** Steps 1–8 run **distributed** across the Spark cluster. When you use `collect()`, data moves to the R driver node (single machine). For large data always keep it in Spark as long as possible.

---

## 5. Base Model 1 — Random Forest (sparklyr)

```r
library(mlflow)

# ── Start MLflow run ─────────────────────────────────────────────────
mlflow_set_experiment("/Users/yourname@company.com/churn_base_models")

with(mlflow_start_run(run_name = "random_forest_v1"), {

  # ── Model parameters ───────────────────────────────────────────────
  params <- list(
    num_trees               = 300,
    max_depth               = 8,
    min_instances_per_node  = 5,
    feature_subset_strategy = "sqrt",
    subsampling_rate        = 0.8
  )

  # ── Log parameters to MLflow ────────────────────────────────────────
  mlflow_log_param("num_trees",               params$num_trees)
  mlflow_log_param("max_depth",               params$max_depth)
  mlflow_log_param("min_instances_per_node",  params$min_instances_per_node)
  mlflow_log_param("feature_subset_strategy", params$feature_subset_strategy)
  mlflow_log_param("subsampling_rate",        params$subsampling_rate)

  # ── Train on Spark cluster ───────────────────────────────────────────
  rf_model <- ml_random_forest_classifier(
    x                       = train,
    formula                 = label ~ .,
    features_col            = "features",
    label_col               = "label",
    num_trees               = params$num_trees,
    max_depth               = params$max_depth,
    min_instances_per_node  = params$min_instances_per_node,
    feature_subset_strategy = params$feature_subset_strategy,
    subsampling_rate        = params$subsampling_rate,
    seed                    = 42
  )

  # ── Evaluate on validation set ──────────────────────────────────────
  val_preds <- ml_predict(rf_model, val)

  auc_roc <- ml_binary_classification_evaluator(
    val_preds,
    label_col          = "label",
    raw_prediction_col = "rawPrediction",
    metric_name        = "areaUnderROC"
  )
  auc_pr <- ml_binary_classification_evaluator(
    val_preds,
    label_col          = "label",
    raw_prediction_col = "rawPrediction",
    metric_name        = "areaUnderPR"
  )
  f1 <- ml_multiclass_classification_evaluator(
    val_preds,
    label_col      = "label",
    prediction_col = "prediction",
    metric_name    = "f1"
  )

  cat("RF  AUC-ROC:", round(auc_roc, 4),
      "| AUC-PR:", round(auc_pr, 4),
      "| F1:", round(f1, 4), "\n")

  # ── Log metrics to MLflow ───────────────────────────────────────────
  mlflow_log_metric("val_auc_roc", auc_roc)
  mlflow_log_metric("val_auc_pr",  auc_pr)
  mlflow_log_metric("val_f1",      f1)

  # ── Feature importance ──────────────────────────────────────────────
  imp <- ml_feature_importances(rf_model, train)
  imp_path <- "/tmp/rf_feature_importance.csv"
  write.csv(imp, imp_path, row.names = FALSE)
  mlflow_log_artifact(imp_path, "feature_importance")

  # ── Save model to MLflow ────────────────────────────────────────────
  ml_save(rf_model, path = "/tmp/rf_model", overwrite = TRUE)
  mlflow_log_artifact("/tmp/rf_model", "spark_model")
})
```

---

## 6. Base Model 2 — GBM / Gradient Boosting (sparklyr)

```r
# Spark ML has Gradient Boosted Trees (GBT) — different from gbm package
# GBT in Spark is a sequential boosting algorithm on Spark DataFrames

with(mlflow_start_run(run_name = "gbt_v1"), {

  mlflow_log_param("max_iter",              100)
  mlflow_log_param("max_depth",             5)
  mlflow_log_param("step_size",             0.1)    # learning rate
  mlflow_log_param("subsampling_rate",      0.8)
  mlflow_log_param("feature_subset_strategy","sqrt")

  gbt_model <- ml_gbt_classifier(
    x                       = train,
    formula                 = label ~ .,
    features_col            = "features",
    label_col               = "label",
    max_iter                = 100,          # number of boosting rounds
    max_depth               = 5,
    step_size               = 0.1,          # learning rate (shrinkage)
    subsampling_rate        = 0.8,
    feature_subset_strategy = "sqrt",
    min_instances_per_node  = 10,
    seed                    = 42
  )

  val_preds_gbt <- ml_predict(gbt_model, val)

  auc_roc_gbt <- ml_binary_classification_evaluator(
    val_preds_gbt, label_col = "label",
    raw_prediction_col = "rawPrediction", metric_name = "areaUnderROC")
  f1_gbt <- ml_multiclass_classification_evaluator(
    val_preds_gbt, label_col = "label",
    prediction_col = "prediction", metric_name = "f1")

  cat("GBT AUC-ROC:", round(auc_roc_gbt, 4), "| F1:", round(f1_gbt, 4), "\n")

  mlflow_log_metric("val_auc_roc", auc_roc_gbt)
  mlflow_log_metric("val_f1",      f1_gbt)

  ml_save(gbt_model, path = "/tmp/gbt_model", overwrite = TRUE)
  mlflow_log_artifact("/tmp/gbt_model", "spark_model")
})
```

> **Note:** For the original `gbm` package (runs on driver only), collect a sample:

```r
# Collect a manageable sample to R driver for gbm package
train_r <- collect(sdf_sample(train, fraction = 0.3, seed = 42))
train_r$Churn_bin <- as.integer(train_r$label)

library(gbm)
set.seed(42)
gbm_driver <- gbm(
  Churn_bin ~ tenure + MonthlyCharges + TotalCharges + tenure_sq + avg_charges,
  data              = train_r,
  distribution      = "bernoulli",
  n.trees           = 500,
  interaction.depth = 4,
  shrinkage         = 0.05,
  n.minobsinnode    = 10,
  bag.fraction      = 0.7,
  cv.folds          = 5,
  verbose           = FALSE
)
best_n <- gbm.perf(gbm_driver, method = "cv", plot.it = FALSE)
```

---

## 7. Base Model 3 — XGBoost (distributed via SparkR)

```r
# Option A: XGBoost via sparklyr (SparkXGBoost — distributed)
# Requires Databricks ML Runtime which includes spark-xgboost jar

with(mlflow_start_run(run_name = "xgboost_spark_v1"), {

  mlflow_log_param("num_round",     200)
  mlflow_log_param("max_depth",     5)
  mlflow_log_param("eta",           0.05)
  mlflow_log_param("subsample",     0.8)
  mlflow_log_param("lambda",        1.0)
  mlflow_log_param("alpha",         0.0)

  xgb_model <- ml_xgboost_classifier(
    x              = train,
    formula        = label ~ .,
    features_col   = "features",
    label_col      = "label",
    num_round      = 200,
    max_depth      = 5,
    eta            = 0.05,           # learning rate
    subsample      = 0.8,
    colsample_bytree = 0.8,
    min_child_weight = 5,
    gamma          = 0.1,
    lambda         = 1.0,            # L2 regularisation
    alpha          = 0.0,            # L1 regularisation
    objective      = "binary:logistic",
    eval_metric    = "auc",
    seed           = 42
  )

  val_preds_xgb <- ml_predict(xgb_model, val)

  auc_roc_xgb <- ml_binary_classification_evaluator(
    val_preds_xgb, label_col = "label",
    raw_prediction_col = "rawPrediction", metric_name = "areaUnderROC")

  cat("XGB AUC-ROC:", round(auc_roc_xgb, 4), "\n")
  mlflow_log_metric("val_auc_roc", auc_roc_xgb)

  ml_save(xgb_model, "/tmp/xgb_model", overwrite = TRUE)
  mlflow_log_artifact("/tmp/xgb_model", "spark_model")
})

# ── Option B: XGBoost base R (driver node, on collected sample) ───────
library(xgboost)

train_r     <- collect(sdf_sample(train, fraction = 0.5, seed = 42))
val_r       <- collect(val)

X_train_r   <- model.matrix(label ~ . - 1,
                 data = train_r[, c("label", "tenure", "MonthlyCharges",
                                    "TotalCharges", "tenure_sq", "avg_charges")])
X_val_r     <- model.matrix(label ~ . - 1,
                 data = val_r[, c("label", "tenure", "MonthlyCharges",
                                  "TotalCharges", "tenure_sq", "avg_charges")])
y_train_r   <- train_r$label
y_val_r     <- val_r$label

dtrain <- xgb.DMatrix(X_train_r, label = y_train_r)
dval   <- xgb.DMatrix(X_val_r,   label = y_val_r)

set.seed(42)
xgb_cv <- xgb.cv(
  params  = list(objective = "binary:logistic", eval_metric = "auc",
                 eta = 0.05, max_depth = 5, subsample = 0.8,
                 colsample_bytree = 0.8, min_child_weight = 5,
                 lambda = 1.0, alpha = 0.0),
  data     = dtrain,
  nrounds  = 500,
  nfold    = 5,
  early_stopping_rounds = 30,
  verbose  = 0
)

xgb_driver <- xgb.train(
  params    = list(objective = "binary:logistic", eval_metric = "auc",
                   eta = 0.05, max_depth = 5, subsample = 0.8,
                   colsample_bytree = 0.8, lambda = 1.0),
  data      = dtrain,
  nrounds   = xgb_cv$best_iteration,
  watchlist = list(val = dval),
  verbose   = 0
)
```

---

## 8. Base Model 4 — LightGBM (sparklyr + MMLSpark)

```r
# Option A: LightGBM via SynapseML (MMLSpark) — distributed on cluster
# SynapseML is pre-installed on Databricks ML Runtime

with(mlflow_start_run(run_name = "lightgbm_v1"), {

  mlflow_log_param("num_iterations",   300)
  mlflow_log_param("learning_rate",    0.05)
  mlflow_log_param("num_leaves",       31)
  mlflow_log_param("min_data_in_leaf", 20)
  mlflow_log_param("lambda_l2",        1.0)

  # LightGBM via SynapseML (Python-based, call via reticulate or Scala)
  # Most practical approach in R: collect sample, use lightgbm R package

  train_r <- collect(sdf_sample(train, fraction = 0.5, seed = 42))
  val_r   <- collect(val)

  # Build matrices
  feature_names <- c("tenure", "MonthlyCharges", "TotalCharges",
                     "tenure_sq", "avg_charges", "high_value_flag")

  X_tr <- as.matrix(train_r[, feature_names])
  X_va <- as.matrix(val_r[,   feature_names])
  y_tr <- train_r$label
  y_va <- val_r$label

  dtrain_lgb <- lgb.Dataset(X_tr, label = y_tr)
  dval_lgb   <- lgb.Dataset(X_va, label = y_va, reference = dtrain_lgb)

  lgb_params <- list(
    objective         = "binary",
    metric            = "auc",
    learning_rate     = 0.05,
    num_leaves        = 31,
    max_depth         = -1,
    min_data_in_leaf  = 20,
    feature_fraction  = 0.8,
    bagging_fraction  = 0.8,
    bagging_freq      = 5,
    lambda_l1         = 0.0,
    lambda_l2         = 1.0,
    verbose           = -1
  )

  lgb_model <- lgb.train(
    params    = lgb_params,
    data      = dtrain_lgb,
    nrounds   = 500,
    valids    = list(val = dval_lgb),
    early_stopping_rounds = 30,
    verbose   = -1
  )

  lgb_val_prob <- predict(lgb_model, X_va)
  library(pROC)
  auc_lgb <- as.numeric(auc(roc(y_va, lgb_val_prob, quiet = TRUE)))

  cat("LGB AUC-ROC:", round(auc_lgb, 4), "\n")
  mlflow_log_metric("val_auc_roc",  auc_lgb)
  mlflow_log_param("best_iter",     lgb_model$best_iter)

  # Save LightGBM model to DBFS
  lgb.save(lgb_model, "/dbfs/tmp/lgb_churn_v1.txt")
  mlflow_log_artifact("/dbfs/tmp/lgb_churn_v1.txt", "lgb_model")
})
```

---

## 9. Base Model 5 — Logistic Regression / LASSO (sparklyr)

```r
# ── Option A: Spark Logistic Regression (distributed) ────────────────
with(mlflow_start_run(run_name = "logistic_regression_spark_v1"), {

  mlflow_log_param("elastic_net_param", 1.0)   # 1.0 = LASSO, 0.0 = Ridge
  mlflow_log_param("reg_param",         0.01)
  mlflow_log_param("max_iter",          100)

  lr_model <- ml_logistic_regression(
    x                = train,
    formula          = label ~ .,
    features_col     = "features",
    label_col        = "label",
    elastic_net_param = 1.0,     # alpha: 1=LASSO, 0=Ridge, 0.5=ElasticNet
    reg_param        = 0.01,     # overall regularisation strength (lambda)
    max_iter         = 100,
    standardization  = TRUE      # always standardise for LR
  )

  val_preds_lr <- ml_predict(lr_model, val)

  auc_lr <- ml_binary_classification_evaluator(
    val_preds_lr, label_col = "label",
    raw_prediction_col = "rawPrediction", metric_name = "areaUnderROC")
  f1_lr <- ml_multiclass_classification_evaluator(
    val_preds_lr, label_col = "label",
    prediction_col = "prediction", metric_name = "f1")

  cat("LR AUC-ROC:", round(auc_lr, 4), "| F1:", round(f1_lr, 4), "\n")

  # Model coefficients
  cat("Coefficients:\n")
  print(lr_model$coefficients)

  mlflow_log_metric("val_auc_roc", auc_lr)
  mlflow_log_metric("val_f1",      f1_lr)

  ml_save(lr_model, "/tmp/lr_model", overwrite = TRUE)
  mlflow_log_artifact("/tmp/lr_model", "spark_model")
})

# ── Option B: glmnet LASSO (driver node) ─────────────────────────────
library(glmnet)

train_r  <- collect(sdf_sample(train, fraction = 0.5, seed = 42))
val_r    <- collect(val)
X_tr     <- model.matrix(label ~ . - 1,
              data = train_r[, c("label", "tenure", "MonthlyCharges",
                                 "TotalCharges", "tenure_sq", "avg_charges")])
X_va     <- model.matrix(label ~ . - 1,
              data = val_r[, c("label", "tenure", "MonthlyCharges",
                               "TotalCharges", "tenure_sq", "avg_charges")])

set.seed(42)
lasso_cv  <- cv.glmnet(X_tr, train_r$label,
                        family = "binomial", alpha = 1, nfolds = 5,
                        type.measure = "auc")
lasso_fit <- glmnet(X_tr, train_r$label,
                    family = "binomial", alpha = 1,
                    lambda = lasso_cv$lambda.min)

lasso_prob <- as.numeric(predict(lasso_fit, newx = X_va, type = "response"))
cat("LASSO AUC:", round(auc(roc(val_r$label, lasso_prob, quiet = TRUE)), 4), "\n")
```

---

## 10. Handling Class Imbalance on Databricks

```r
# ── Check imbalance ───────────────────────────────────────────────────
train %>%
  group_by(label) %>%
  summarise(n = n()) %>%
  mutate(pct = round(n / sum(n) * 100, 2)) %>%
  collect()

# ── Strategy 1: weightCol — native in Spark ML ───────────────────────
n_total  <- sdf_nrow(train)
n_churn  <- train %>% filter(label == 1) %>% sdf_nrow()
n_active <- n_total - n_churn

w_churn  <- n_total / (2 * n_churn)
w_active <- n_total / (2 * n_active)

train_w <- train %>%
  mutate(class_weight = ifelse(label == 1, w_churn, w_active))

# Apply to any Spark ML classifier
rf_weighted <- ml_random_forest_classifier(
  x          = train_w,
  formula    = label ~ .,
  num_trees  = 300,
  weight_col = "class_weight",   # ← works for RF, GBT, LR in Spark ML
  seed       = 42
)

# ── Strategy 2: Spark-side oversampling (minority class) ─────────────
majority <- train %>% filter(label == 0)
minority  <- train %>% filter(label == 1)

ratio        <- ceiling(sdf_nrow(majority) / sdf_nrow(minority))
minority_over <- sdf_sample(minority, fraction = ratio,
                             replacement = TRUE, seed = 42)

train_balanced <- sdf_bind_rows(majority, minority_over) %>%
  sdf_sample(fraction = 1.0, replacement = FALSE, seed = 99)  # shuffle

cat("Balanced train:", sdf_nrow(train_balanced), "\n")
train_balanced %>% group_by(label) %>% summarise(n = n()) %>% collect()

# ── Strategy 3: ROSE on driver (for smaller datasets / samples) ───────
library(ROSE)
train_sample <- collect(sdf_sample(train, fraction = 0.3, seed = 42))
# ROSE needs Churn as factor
train_sample$Churn_f <- factor(train_sample$label,
                                levels = c(0, 1),
                                labels = c("Active", "Churn"))
train_rose <- ROSE(Churn_f ~ tenure + MonthlyCharges + TotalCharges +
                     tenure_sq + avg_charges,
                   data = train_sample, seed = 42)$data
table(train_rose$Churn_f)

# ── Strategy 4: Threshold tuning after scoring ────────────────────────
# Extract probability column from Spark predictions
val_preds_prob <- ml_predict(rf_model, val) %>%
  mutate(prob_churn = vector_to_array(probability)[2]) %>%
  select(label, prob_churn) %>%
  collect()

# Find best F1 threshold
thresholds <- seq(0.2, 0.8, by = 0.05)
f1_scores  <- sapply(thresholds, function(t) {
  pred <- factor(ifelse(val_preds_prob$prob_churn >= t, "Churn", "Active"),
                 levels = c("Active", "Churn"))
  act  <- factor(ifelse(val_preds_prob$label == 1, "Churn", "Active"),
                 levels = c("Active", "Churn"))
  cm   <- caret::confusionMatrix(pred, act, positive = "Churn")
  cm$byClass["F1"]
})

best_thr <- thresholds[which.max(f1_scores)]
cat("Best threshold:", best_thr, "| F1:", round(max(f1_scores, na.rm = TRUE), 4), "\n")
```

---

## 11. Regularisation

```r
# ── Spark Random Forest — structural regularisation ──────────────────
rf_reg <- ml_random_forest_classifier(
  x = train, formula = label ~ .,
  num_trees               = 500,
  max_depth               = 5,       # lower → simpler trees
  min_instances_per_node  = 15,      # large leaf → less overfit
  min_info_gain           = 0.01,    # only meaningful splits
  subsampling_rate        = 0.65,    # row sampling per tree
  feature_subset_strategy = "sqrt",  # column sampling
  max_bins                = 32,
  seed                    = 42
)

# ── Spark GBT — regularisation ────────────────────────────────────────
gbt_reg <- ml_gbt_classifier(
  x = train, formula = label ~ .,
  max_iter               = 200,
  step_size              = 0.01,    # very small learning rate
  max_depth              = 3,       # shallow weak learners
  min_instances_per_node = 20,
  subsampling_rate       = 0.6,
  feature_subset_strategy = "sqrt",
  seed                   = 42
)

# ── Spark Logistic Regression — L1/L2 ────────────────────────────────
lr_reg <- ml_logistic_regression(
  x = train, formula = label ~ .,
  elastic_net_param = 0.5,   # ElasticNet: mix of L1 and L2
  reg_param         = 0.1,   # stronger regularisation
  max_iter          = 200,
  standardization   = TRUE
)

# ── XGBoost (driver) — full regularisation suite ─────────────────────
xgb_params_reg <- list(
  objective        = "binary:logistic",
  eval_metric      = "auc",
  eta              = 0.01,       # small learning rate
  max_depth        = 4,
  subsample        = 0.7,
  colsample_bytree = 0.7,
  min_child_weight = 10,
  gamma            = 0.5,        # split only if gain exceeds this
  lambda           = 2.0,        # L2 regularisation on leaf weights
  alpha            = 0.5         # L1 regularisation on leaf weights
)

# ── LightGBM (driver) — regularisation ───────────────────────────────
lgb_params_reg <- list(
  objective         = "binary",
  metric            = "auc",
  learning_rate     = 0.01,
  num_leaves        = 15,        # lower = less complex
  min_data_in_leaf  = 30,
  feature_fraction  = 0.7,
  bagging_fraction  = 0.7,
  bagging_freq      = 5,
  lambda_l1         = 0.5,
  lambda_l2         = 2.0,
  min_gain_to_split = 0.1,
  verbose           = -1
)
```

### Regularisation parameters at a glance

| Model | Parameter | Direction | Effect |
|---|---|---|---|
| RF / GBT | `max_depth` | ↓ lower | Simpler trees |
| RF / GBT | `min_instances_per_node` | ↑ higher | Larger leaves |
| RF / GBT | `subsampling_rate` | ↓ lower | Diverse trees |
| GBT / XGB / LGB | learning rate | ↓ lower | Slower, safer learning |
| XGBoost | `lambda` | ↑ higher | L2 weight penalty |
| XGBoost | `alpha` | ↑ higher | L1 weight penalty |
| XGBoost | `gamma` | ↑ higher | Min gain to split |
| LightGBM | `lambda_l2` | ↑ higher | L2 leaf regularisation |
| LightGBM | `num_leaves` | ↓ lower | Less model complexity |
| Logistic Reg | `reg_param` | ↑ higher | Stronger L1/L2 penalty |

---

## 12. Model Inspection — Under the Hood

```r
# ── Random Forest: feature importance ────────────────────────────────
rf_imp <- ml_feature_importances(rf_model, train)
print(rf_imp %>% arrange(desc(importance)))

# ── Random Forest: tree debug string (first tree) ────────────────────
tree_str <- ml_stage(rf_model) %>%
  invoke("trees") %>%
  .[[1]] %>%
  invoke("toDebugString")
cat(tree_str)

# ── RF: per-tree depth and node count ────────────────────────────────
trees_java  <- ml_stage(rf_model) %>% invoke("trees")
tree_depths <- sapply(trees_java, function(t) invoke(t, "depth"))
tree_nodes  <- sapply(trees_java, function(t) invoke(t, "numNodes"))
summary(data.frame(depth = tree_depths, nodes = tree_nodes))

# ── GBT: number of trees used ────────────────────────────────────────
gbt_model$num_trees
ml_stage(gbt_model) %>% invoke("getMaxIter")

# ── Logistic Regression: coefficients ────────────────────────────────
lr_model$coefficients
lr_model$intercept

# ── XGBoost (driver): SHAP values ────────────────────────────────────
shap_vals <- predict(xgb_driver,
                     xgb.DMatrix(X_val_r),
                     predcontrib = TRUE)
# Column per feature + BIAS column at end
shap_df <- data.frame(
  feature = c(colnames(X_val_r), "BIAS"),
  mean_abs_shap = colMeans(abs(shap_vals))
) %>% arrange(desc(mean_abs_shap))
print(head(shap_df, 15))

# ── XGBoost: tree dump ───────────────────────────────────────────────
xgb.dump(xgb_driver, with_stats = TRUE)[1:30]

# ── XGBoost: importance plot ─────────────────────────────────────────
xgb_imp <- xgb.importance(model = xgb_driver)
xgb.plot.importance(xgb_imp, top_n = 15,
                    main = "XGBoost Feature Importance (Gain)")

# ── LightGBM: importance ─────────────────────────────────────────────
lgb_imp <- lgb.importance(lgb_model, percentage = TRUE)
lgb.plot.importance(lgb_imp, top_n = 15,
                    main = "LightGBM Feature Importance")

# ── Spark ML: raw prediction = vote counts for RF ────────────────────
# rawPrediction[2] = number of trees voting "Churn"
ml_predict(rf_model, val) %>%
  mutate(
    prob_churn = vector_to_array(probability)[2],
    vote_count = vector_to_array(rawPrediction)[2]
  ) %>%
  select(label, prob_churn, vote_count, prediction) %>%
  head(10) %>%
  collect()
```

---

## 13. Hyperparameter Tuning

### Cross-validation via Spark ML Pipeline

```r
# ── Build pipeline ────────────────────────────────────────────────────
pipeline <- ml_pipeline(sc) %>%
  ft_r_formula(label ~ .) %>%
  ml_random_forest_classifier(
    label_col    = "label",
    features_col = "features"
  )

# ── Define parameter grid ─────────────────────────────────────────────
param_grid <- list(
  random_forest_classifier = list(
    num_trees               = c(100L, 300L, 500L),
    max_depth               = c(4L, 6L, 8L),
    min_instances_per_node  = c(5L, 10L),
    subsampling_rate        = c(0.7, 0.9)
  )
)

# ── AUC-ROC evaluator ────────────────────────────────────────────────
evaluator <- ml_binary_classification_evaluator(
  sc,
  label_col   = "label",
  metric_name = "areaUnderROC"
)

# ── 3-fold cross-validation ───────────────────────────────────────────
cv <- ml_cross_validator(
  sc,
  estimator             = pipeline,
  estimator_param_maps  = param_grid,
  evaluator             = evaluator,
  num_folds             = 3,
  seed                  = 42
)

cv_model   <- ml_fit(cv, train)
best_model <- cv_model$best_model

# ── Review all CV results ─────────────────────────────────────────────
cv_results <- ml_validation_metrics(cv_model)
print(cv_results %>% arrange(desc(areaUnderROC)))

# Best parameters
best_rf_stage <- ml_stage(best_model, "random_forest_classifier")
cat("Best num_trees:", invoke(best_rf_stage, "getNumTrees"), "\n")
cat("Best max_depth:", invoke(best_rf_stage, "getMaxDepth"), "\n")
```

### Log CV results to MLflow

```r
with(mlflow_start_run(run_name = "rf_cv_tuning"), {

  cv_model <- ml_fit(cv, train)
  best_rf  <- cv_model$best_model

  best_stage <- ml_stage(best_rf, "random_forest_classifier")

  mlflow_log_param("best_num_trees", invoke(best_stage, "getNumTrees"))
  mlflow_log_param("best_max_depth", invoke(best_stage, "getMaxDepth"))

  val_preds <- ml_predict(best_rf, val)
  best_auc  <- ml_binary_classification_evaluator(
    val_preds, label_col = "label",
    raw_prediction_col = "rawPrediction", metric_name = "areaUnderROC")

  mlflow_log_metric("best_val_auc_roc", best_auc)

  # Save CV results as artifact
  cv_df <- ml_validation_metrics(cv_model)
  write.csv(cv_df, "/tmp/cv_results.csv", row.names = FALSE)
  mlflow_log_artifact("/tmp/cv_results.csv", "cv_results")
})
```

### Databricks AutoML (no-code baseline)

```
Databricks UI → Experiments → Create AutoML Experiment

Settings:
  Problem type:    Classification
  Input table:     analytics.churn_features
  Prediction target: label
  Metric:          F1
  Timeout:         60 minutes

AutoML will:
  - Train RF, GBT, XGBoost, LightGBM, LR variants
  - Log all runs to MLflow automatically
  - Produce a best-model notebook you can inspect and modify
  - Register the best model in the Model Registry
```

```r
# Call AutoML from R notebook via reticulate
library(reticulate)
automl <- import("databricks.automl")

summary <- automl$classify(
  dataset     = spark_dataframe(train),
  target_col  = "label",
  primary_metric = "f1",
  timeout_minutes = 60L
)

cat("Best run ID:", summary$best_trial$mlflow_run_id, "\n")
```

---

## 14. MLflow — Track Every Experiment

```r
library(mlflow)

# ── Set experiment (creates it if it doesn't exist) ──────────────────
mlflow_set_experiment("/Users/yourname@company.com/churn_base_models")

# ── Log a complete run ───────────────────────────────────────────────
with(mlflow_start_run(run_name = "rf_experiment_1"), {

  # Parameters
  mlflow_log_param("model_type",    "random_forest")
  mlflow_log_param("num_trees",     300)
  mlflow_log_param("max_depth",     8)
  mlflow_log_param("imbalance",     "class_weight")
  mlflow_log_param("train_rows",    sdf_nrow(train))
  mlflow_log_param("val_rows",      sdf_nrow(val))

  # Train
  rf <- ml_random_forest_classifier(
    x = train_w, formula = label ~ .,
    num_trees = 300, max_depth = 8,
    weight_col = "class_weight", seed = 42
  )

  # Evaluate
  preds <- ml_predict(rf, val)
  auc   <- ml_binary_classification_evaluator(preds,
             label_col = "label", metric_name = "areaUnderROC")
  f1    <- ml_multiclass_classification_evaluator(preds,
             label_col = "label", metric_name = "f1")

  # Metrics
  mlflow_log_metric("val_auc_roc", auc)
  mlflow_log_metric("val_f1",      f1)

  # Artefacts — feature importance plot
  imp      <- ml_feature_importances(rf, train)
  imp_plot <- ggplot(head(imp, 15),
                     aes(x = reorder(feature, importance), y = importance)) +
              geom_col(fill = "#1D9E75") + coord_flip() +
              labs(title = "Feature Importance", x = NULL)
  ggsave("/tmp/feature_importance.png", imp_plot, width = 8, height = 5)
  mlflow_log_artifact("/tmp/feature_importance.png", "plots")

  # Save Spark ML model
  ml_save(rf, "/tmp/rf_model_run", overwrite = TRUE)
  mlflow_log_artifact("/tmp/rf_model_run", "spark_model")

  # Tag the run
  mlflow_set_tag("team",          "data_science")
  mlflow_set_tag("data_version",  "v2.1")
  mlflow_set_tag("feature_set",   "v3")

  run_id <- mlflow_active_run()$run_uuid
  cat("Run ID:", run_id, "\n")
})

# ── Search runs programmatically ─────────────────────────────────────
runs <- mlflow_search_runs(
  filter        = "metrics.val_auc_roc > 0.85",
  order_by      = "metrics.val_auc_roc DESC",
  max_results   = 10
)
print(runs[, c("run_id", "metrics.val_auc_roc", "metrics.val_f1",
               "params.num_trees", "params.max_depth")])
```

---

## 15. Save & Register Models in MLflow Model Registry

```r
# ── Step 1: Log and register a Spark ML model ────────────────────────
with(mlflow_start_run(run_name = "rf_final_register"), {

  rf_final <- ml_random_forest_classifier(
    x = train_w, formula = label ~ .,
    num_trees = 300, max_depth = 8,
    weight_col = "class_weight", seed = 42
  )

  preds   <- ml_predict(rf_final, test)
  auc_test <- ml_binary_classification_evaluator(preds,
               label_col = "label", metric_name = "areaUnderROC")

  mlflow_log_metric("test_auc_roc", auc_test)

  # Save Spark model to DBFS (MLflow artefact store)
  model_path <- "/dbfs/models/rf_churn_final"
  ml_save(rf_final, path = model_path, overwrite = TRUE)
  mlflow_log_artifact(model_path, "spark_ml_model")

  # Register in Model Registry
  run_id <- mlflow_active_run()$run_uuid
})

# ── Step 2: Register model in Model Registry UI or API ───────────────
# UI: Experiments → Run → "Register Model" button
# Or via API:
model_uri <- paste0("runs:/", run_id, "/spark_ml_model")
mlflow_register_model(model_uri, name = "churn_rf_classifier")

# ── Step 3: Transition to Staging / Production ────────────────────────
mlflow_transition_model_version_stage(
  name    = "churn_rf_classifier",
  version = 1,
  stage   = "Staging"    # "Staging", "Production", "Archived"
)

# ── Step 4: Load a registered model for inference ────────────────────
# Load the "Production" version
prod_model_uri <- "models:/churn_rf_classifier/Production"

# In a Databricks Job / serving notebook:
rf_production <- ml_load(sc, path = "/dbfs/models/rf_churn_final")

# ── Save non-Spark models (XGBoost, LightGBM, glmnet) ────────────────
# XGBoost
xgb.save(xgb_driver,  "/dbfs/models/xgb_churn_v1.model")
mlflow_log_artifact("/dbfs/models/xgb_churn_v1.model", "xgb_model")

# LightGBM
lgb.save(lgb_model, "/dbfs/models/lgb_churn_v1.txt")
mlflow_log_artifact("/dbfs/models/lgb_churn_v1.txt", "lgb_model")

# glmnet / randomForest (base R)
saveRDS(list(
  lasso      = lasso_fit,
  metadata   = list(version = "v1", trained_on = Sys.Date(),
                    auc_roc  = auc_test, threshold = best_thr,
                    features = feature_names)
), "/dbfs/models/lasso_bundle_v1.rds")
mlflow_log_artifact("/dbfs/models/lasso_bundle_v1.rds", "lasso_model")
```

---

## 16. Predict on New Data & Join Back

```r
# ── Load production models ───────────────────────────────────────────
rf_prod    <- ml_load(sc, path = "/dbfs/models/rf_churn_final")
xgb_prod   <- xgb.load("/dbfs/models/xgb_churn_v1.model")
lgb_prod   <- lgb.load("/dbfs/models/lgb_churn_v1.txt")
lasso_prod <- readRDS("/dbfs/models/lasso_bundle_v1.rds")

# ── Load new data from Delta Lake ────────────────────────────────────
new_spark <- spark_read_table(sc, "analytics.new_customers")

# ── Apply IDENTICAL feature engineering ─────────────────────────────
new_spark <- new_spark %>%
  mutate(
    tenure_sq       = tenure * tenure,
    avg_charges     = TotalCharges / ifelse(tenure > 0, tenure, 1),
    high_value_flag = ifelse(MonthlyCharges > 70, 1L, 0L),
    charge_ratio    = MonthlyCharges / ifelse(TotalCharges > 0, TotalCharges, 1)
  )

# ── Score with Spark RF (stays distributed) ──────────────────────────
scored_spark <- ml_predict(rf_prod, new_spark) %>%
  mutate(
    prob_churn = vector_to_array(probability)[2],
    Prediction = ifelse(prediction == 1, "Churn", "Active")
  )

# ── Join predictions back to original dataset ─────────────────────────
final_spark <- new_spark %>%
  left_join(
    scored_spark %>% select(CustomerID, prob_churn, Prediction),
    by = "CustomerID"
  )

# Preview
final_spark %>%
  select(CustomerID, tenure, MonthlyCharges, prob_churn, Prediction) %>%
  head(10) %>%
  collect()
# CustomerID  tenure  MonthlyCharges  prob_churn  Prediction
# C0001           3         82.45       0.9124       Churn
# C0002          42         55.30       0.0891       Active
# C0003           8         91.10       0.8867       Churn
# C0004          60         34.20       0.0312       Active

# ── Apply optimal threshold ───────────────────────────────────────────
final_spark <- final_spark %>%
  mutate(
    Prediction = ifelse(prob_churn >= best_thr, "Churn", "Active")
  )

# ── Score with driver-based models (XGBoost / LightGBM) ──────────────
# Collect new data to R driver for non-Spark models
new_r <- collect(new_spark)
X_new <- as.matrix(new_r[, feature_names])

# XGBoost
prob_xgb_new  <- predict(xgb_prod, xgb.DMatrix(X_new))

# LightGBM
prob_lgb_new  <- predict(lgb_prod, X_new)

# LASSO
X_new_mat     <- model.matrix(~ . - 1, data = new_r[, feature_names])
prob_lasso_new <- as.numeric(predict(lasso_prod$lasso,
                                      newx = X_new_mat, type = "response"))

cat("Scored", nrow(new_r), "customers\n")
cat("Predicted Churn:", sum(prob_xgb_new >= best_thr), "\n")
```

---

## 17. Write Predictions to Delta Lake

```r
# ── Write Spark predictions directly to Delta table ──────────────────
spark_write_delta(
  final_spark,
  path = "/mnt/datalake/churn/predictions/",
  mode = "overwrite"   # or "append" for incremental scoring
)

# ── Or write to a managed Hive / Unity Catalog table ─────────────────
spark_write_table(
  final_spark,
  name = "analytics.churn_predictions",
  mode = "overwrite"
)

# ── Add partition for date (incremental scoring pipelines) ───────────
final_spark_dated <- final_spark %>%
  mutate(score_date = as.character(Sys.Date()))

spark_write_delta(
  final_spark_dated,
  path           = "/mnt/datalake/churn/predictions_partitioned/",
  mode           = "append",
  partition_by   = "score_date"
)

# ── Verify the write ─────────────────────────────────────────────────
spark_read_table(sc, "analytics.churn_predictions") %>%
  group_by(Prediction) %>%
  summarise(n = n()) %>%
  collect()

# ── Log predictions summary to MLflow ────────────────────────────────
with(mlflow_start_run(run_name = "scoring_run"), {
  total    <- sdf_nrow(final_spark)
  n_churn  <- final_spark %>% filter(Prediction == "Churn") %>% sdf_nrow()
  mlflow_log_metric("scored_total",   total)
  mlflow_log_metric("predicted_churn", n_churn)
  mlflow_log_metric("churn_rate_pct",  round(n_churn / total * 100, 2))
  mlflow_log_param("score_date",      as.character(Sys.Date()))
  mlflow_set_tag("job_type",          "batch_scoring")
})

# ── Disconnect ───────────────────────────────────────────────────────
spark_disconnect(sc)
```

---

## 18. Quick-Reference Cheatsheet

### Databricks-specific functions

| Task | Function | Notes |
|---|---|---|
| Connect to Spark | `spark_connect(method = "databricks")` | Inside notebook only |
| Read Delta table | `spark_read_table(sc, "db.table")` | Preferred for managed tables |
| Read Delta path | `spark_read_delta(sc, path)` | For unmanaged paths |
| Write Delta | `spark_write_delta(df, path, mode)` | mode: overwrite / append |
| Write table | `spark_write_table(df, name, mode)` | Registers in Hive metastore |
| Cache in Spark | `tbl_cache(sc, "table_name")` | After `sdf_register()` |
| Count Spark rows | `sdf_nrow(df)` | Returns exact count |
| Collect to R | `collect(df)` | Moves data to driver node |
| Read DBFS file | `/dbfs/path/to/file` | Local file path in R |
| MLflow experiment | `mlflow_set_experiment("/path")` | Absolute path in workspace |
| Log parameter | `mlflow_log_param(key, value)` | Inside `with(mlflow_start_run...)` |
| Log metric | `mlflow_log_metric(key, value)` | Numeric only |
| Log file | `mlflow_log_artifact(path)` | Any file type |
| Save Spark model | `ml_save(model, path)` | Saves as Parquet-based format |
| Load Spark model | `ml_load(sc, path)` | Needs active Spark session |

### Spark ML classifiers in sparklyr

| Model | Function | Key params |
|---|---|---|
| Random Forest | `ml_random_forest_classifier()` | `num_trees`, `max_depth`, `subsampling_rate` |
| Gradient Boosted Trees | `ml_gbt_classifier()` | `max_iter`, `step_size`, `max_depth` |
| XGBoost (Spark) | `ml_xgboost_classifier()` | `num_round`, `eta`, `lambda`, `alpha` |
| Logistic Regression | `ml_logistic_regression()` | `reg_param`, `elastic_net_param` |
| Linear SVM | `ml_linear_svc()` | `reg_param`, `max_iter` |
| Naive Bayes | `ml_naive_bayes()` | `smoothing` |

### Driver vs Cluster — when to use each

| Approach | When to use | Size limit |
|---|---|---|
| Spark ML (sparklyr) | Full dataset, production pipelines | Unlimited |
| Collect + base R | Prototyping, packages not in Spark | ~5–10M rows |
| Collect sample | XGBoost / LightGBM / glmnet | 100k–500k rows |
| `sdf_sample()` first | Large data, driver-only models | Control with `fraction` |

### Recommended project structure on Databricks

```
Workspace/
├── Repos/
│   └── churn_project/
│       ├── notebooks/
│       │   ├── 01_data_prep.R
│       │   ├── 02_feature_engineering.R
│       │   ├── 03_base_models.R
│       │   ├── 04_tuning.R
│       │   └── 05_scoring.R
│       └── README.md
├── Experiments/
│   └── churn_base_models/     ← MLflow runs
DBFS (Delta Lake):
├── /mnt/datalake/
│   ├── churn/raw/             ← raw input
│   ├── churn/features/        ← engineered features
│   └── churn/predictions/     ← scored output
└── /dbfs/models/
    ├── rf_churn_final/        ← Spark ML model
    ├── xgb_churn_v1.model     ← XGBoost binary
    ├── lgb_churn_v1.txt       ← LightGBM text
    └── lasso_bundle_v1.rds    ← glmnet RDS bundle
```

### Key gotchas on Databricks

```
1. spark_connect(method = "databricks")  — no host/token needed inside a notebook.
   Use method = "local" only for unit tests in CI/CD outside the cluster.

2. DBFS paths: /dbfs/path  in R file functions (readRDS, xgb.save)
               /mnt/path   in Spark functions (spark_read_delta)
               dbfs:/path  in Databricks CLI / DBFS API

3. Package installs:  install.packages() in notebook = session only (lost on restart).
   Use Cluster Libraries UI for persistent installs.

4. Collect carefully: collect() on a 100M-row DataFrame will crash the driver.
   Always sdf_sample() or filter() first.

5. MLflow auto-logging: Databricks ML Runtime auto-logs spark.ml models.
   You still need manual mlflow_log_metric() for custom metrics.

6. Delta writes: always specify mode = "overwrite" or "append" explicitly.
   Default behaviour differs from CSV writes.

7. model.matrix() on collected data: column order must exactly match training.
   Store colnames(X_train) in your model bundle and reorder at score time.
```

---

*Databricks R · sparklyr · Spark ML · MLflow · Delta Lake · Churn Prediction Base Models Reference*
