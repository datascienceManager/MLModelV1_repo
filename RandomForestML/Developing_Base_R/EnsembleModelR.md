# Ensemble Models in Base R — Complete Guide
### Churn Prediction · Binary Output: `"Churn"` / `"Active"`

> **Stack:** `randomForest` · `gbm` · `xgboost` · `lightgbm` · `glmnet` · `caret` · `caretEnsemble` · `stacking` (manual) · `ROSE` · `pROC`

---

## Table of Contents

1. [What Are Ensemble Models?](#1-what-are-ensemble-models)
2. [Setup & Packages](#2-setup--packages)
3. [Load, Prepare & Split Data](#3-load-prepare--split-data)
4. [Base Models — Individual Learners](#4-base-models--individual-learners)
   - 4a. [Bagging — Random Forest](#4a-bagging--random-forest)
   - 4b. [Boosting — GBM (Gradient Boosting Machine)](#4b-boosting--gbm-gradient-boosting-machine)
   - 4c. [Boosting — XGBoost](#4c-boosting--xgboost)
   - 4d. [Boosting — LightGBM](#4d-boosting--lightgbm)
   - 4e. [Linear Base Learner — Logistic Regression / LASSO](#4e-linear-base-learner--logistic-regression--lasso)
5. [Ensemble Methods](#5-ensemble-methods)
   - 5a. [Simple Voting (Hard & Soft)](#5a-simple-voting-hard--soft)
   - 5b. [Weighted Average Ensemble](#5b-weighted-average-ensemble)
   - 5c. [Stacking (Meta-Learning)](#5c-stacking-meta-learning)
   - 5d. [Blending](#5d-blending)
   - 5e. [caretEnsemble (automated stacking)](#5e-caretensemble-automated-stacking)
6. [Regularisation Across Ensemble Members](#6-regularisation-across-ensemble-members)
7. [Handling Class Imbalance](#7-handling-class-imbalance)
8. [Model Inspection — Under the Hood](#8-model-inspection--under-the-hood)
9. [Hyperparameter Tuning](#9-hyperparameter-tuning)
10. [Save & Load Ensemble Models](#10-save--load-ensemble-models)
11. [Predict on New Data & Join Back](#11-predict-on-new-data--join-back)
12. [Compare All Models](#12-compare-all-models)
13. [Quick-Reference Cheatsheet](#13-quick-reference-cheatsheet)

---

## 1. What Are Ensemble Models?

Ensemble models combine multiple individual learners to produce a stronger, more robust prediction than any single model alone. The key principle: **diverse models that fail differently, combined, cancel out each other's errors**.

```
                     ┌─────────────────────────────────────────┐
                     │           ENSEMBLE TAXONOMY              │
                     └─────────────────────────────────────────┘

  PARALLEL (independent models)        SEQUENTIAL (models correct errors)
  ──────────────────────────────        ──────────────────────────────────
  Bagging          Stacking             Boosting
  ───────          ────────             ────────
  Bootstrap        Train base           Each new model focuses
  samples of       models on folds,     on the mistakes of
  the data,        train meta-model     the previous one.
  aggregate        on their OOF
  predictions.     predictions.         GBM / XGBoost / LightGBM

  Random Forest    Blending             AdaBoost
  Extra Trees      caretEnsemble

  ─────────────────────────────────────────────────────────────
  COMBINATION STRATEGIES (post-training)
  ─────────────────────────────────────────────────────────────
  Hard Voting     Majority class wins
  Soft Voting     Average predicted probabilities
  Weighted Avg    Weighted by validation AUC / F1
  Stacking        A meta-model learns the weights
```

### When to use each method

| Method | Best when | Watch out for |
|---|---|---|
| Bagging (RF) | High variance base models | Not great for low-variance models |
| Boosting (XGB/GBM) | High bias base models | Prone to overfit — regularise carefully |
| Stacking | You have diverse model types | Data leakage risk — must use OOF preds |
| Soft voting | Base models are well-calibrated | Poorly calibrated probs hurt the average |
| Weighted avg | Models differ in quality | Weight optimisation can overfit val set |

---

## 2. Setup & Packages

```r
# ── Core ensemble packages ───────────────────────────────────────────
install.packages("randomForest")    # bagging
install.packages("gbm")             # gradient boosting
install.packages("xgboost")         # extreme gradient boosting
install.packages("lightgbm")        # Microsoft LightGBM
install.packages("glmnet")          # LASSO / Ridge (meta-learner)
install.packages("caret")           # unified train/evaluate wrapper
install.packages("caretEnsemble")   # automated ensemble via caret

# ── Imbalance & evaluation ───────────────────────────────────────────
install.packages("ROSE")
install.packages("DMwR2")
install.packages("pROC")
install.packages("PRROC")
install.packages("ggplot2")

# ── Load all ─────────────────────────────────────────────────────────
library(randomForest)
library(gbm)
library(xgboost)
library(lightgbm)
library(glmnet)
library(caret)
library(caretEnsemble)
library(ROSE)
library(pROC)
library(ggplot2)
```

---

## 3. Load, Prepare & Split Data

```r
# ── Load ─────────────────────────────────────────────────────────────
df <- read.csv("churn.csv", stringsAsFactors = TRUE)

# ── Target as factor ─────────────────────────────────────────────────
df$Churn <- factor(df$Churn,
                   levels = c(0, 1),
                   labels = c("Active", "Churn"))

# ── Feature engineering ───────────────────────────────────────────────
df$tenure_sq   <- df$tenure^2
df$avg_charges <- df$TotalCharges / pmax(df$tenure, 1)
df$high_value  <- ifelse(df$MonthlyCharges > 70, 1L, 0L)

# ── Impute missing numeric values ────────────────────────────────────
num_cols <- names(df)[sapply(df, is.numeric)]
for (col in num_cols) {
  df[[col]][is.na(df[[col]])] <- median(df[[col]], na.rm = TRUE)
}

# ── Encode categoricals ──────────────────────────────────────────────
cat_cols <- c("Contract", "InternetService", "PaymentMethod")
df[cat_cols] <- lapply(df[cat_cols], factor)

# ── Remove ID ────────────────────────────────────────────────────────
df$CustomerID <- NULL

# ── Numeric matrix version (needed by XGBoost / glmnet) ──────────────
df_matrix <- model.matrix(Churn ~ . - 1, data = df)
label_vec  <- as.integer(df$Churn == "Churn")  # 0 = Active, 1 = Churn

# ── Stratified 70 / 15 / 15 split (train / validation / test) ────────
# Three-way split is important for stacking — validation for meta-model
set.seed(42)
train_idx <- createDataPartition(df$Churn, p = 0.70, list = FALSE)
temp      <- df[-train_idx, ]
val_idx   <- createDataPartition(temp$Churn, p = 0.50, list = FALSE)

train <- df[ train_idx, ]
val   <- temp[ val_idx, ]
test  <- temp[-val_idx, ]

# Matrix versions for XGBoost / glmnet
X_train <- df_matrix[ train_idx, ]
X_val   <- df_matrix[-train_idx, ][ val_idx, ]
X_test  <- df_matrix[-train_idx, ][-val_idx, ]

y_train <- label_vec[ train_idx]
y_val   <- label_vec[-train_idx][ val_idx]
y_test  <- label_vec[-train_idx][-val_idx]

cat("Train:", nrow(train), "| Val:", nrow(val), "| Test:", nrow(test), "\n")
```

---

## 4. Base Models — Individual Learners

### 4a. Bagging — Random Forest

```r
set.seed(42)
rf_base <- randomForest(
  Churn ~ .,
  data       = train,
  ntree      = 500,
  mtry       = floor(sqrt(ncol(train) - 1)),
  nodesize   = 5,
  importance = TRUE
)

# Validation probabilities
rf_val_prob <- predict(rf_base, newdata = val, type = "prob")[, "Churn"]
cat("RF  AUC-ROC (val):", round(auc(roc(val$Churn, rf_val_prob)), 4), "\n")
```

### 4b. Boosting — GBM (Gradient Boosting Machine)

```r
# GBM requires numeric 0/1 target
train_gbm        <- train
train_gbm$Churn  <- as.integer(train$Churn == "Churn")
val_gbm          <- val
val_gbm$Churn    <- as.integer(val$Churn == "Churn")

set.seed(42)
gbm_base <- gbm(
  formula          = Churn ~ .,
  data             = train_gbm,
  distribution     = "bernoulli",   # binary classification
  n.trees          = 500,
  interaction.depth = 4,            # tree depth (complexity)
  shrinkage        = 0.05,          # learning rate
  n.minobsinnode   = 10,            # min leaf size (regularisation)
  bag.fraction     = 0.7,           # row subsampling
  train.fraction   = 1.0,
  cv.folds         = 5,             # built-in CV for optimal trees
  verbose          = FALSE
)

# Optimal number of trees (by CV)
best_n_gbm <- gbm.perf(gbm_base, method = "cv", plot.it = FALSE)
cat("GBM optimal trees:", best_n_gbm, "\n")

# Validation probabilities
gbm_val_prob <- predict(gbm_base, newdata = val_gbm,
                        n.trees = best_n_gbm, type = "response")
cat("GBM AUC-ROC (val):", round(auc(roc(val$Churn, gbm_val_prob)), 4), "\n")
```

### 4c. Boosting — XGBoost

```r
dtrain <- xgb.DMatrix(data = X_train, label = y_train)
dval   <- xgb.DMatrix(data = X_val,   label = y_val)

xgb_params <- list(
  objective         = "binary:logistic",
  eval_metric       = "auc",
  eta               = 0.05,          # learning rate
  max_depth         = 5,
  subsample         = 0.8,           # row subsampling
  colsample_bytree  = 0.8,           # column subsampling
  min_child_weight  = 5,             # min leaf weight (regularisation)
  gamma             = 0.1,           # min loss reduction to split
  lambda            = 1.0,           # L2 regularisation
  alpha             = 0.0            # L1 regularisation
)

set.seed(42)
xgb_cv <- xgb.cv(
  params   = xgb_params,
  data     = dtrain,
  nrounds  = 500,
  nfold    = 5,
  early_stopping_rounds = 30,
  verbose  = 0
)

best_n_xgb <- xgb_cv$best_iteration
cat("XGB optimal rounds:", best_n_xgb, "\n")

xgb_base <- xgb.train(
  params     = xgb_params,
  data       = dtrain,
  nrounds    = best_n_xgb,
  watchlist  = list(val = dval),
  verbose    = 0
)

# Validation probabilities
xgb_val_prob <- predict(xgb_base, dval)
cat("XGB AUC-ROC (val):", round(auc(roc(val$Churn, xgb_val_prob)), 4), "\n")
```

### 4d. Boosting — LightGBM

```r
dtrain_lgb <- lgb.Dataset(data = X_train, label = y_train)
dval_lgb   <- lgb.Dataset(data = X_val,   label = y_val, reference = dtrain_lgb)

lgb_params <- list(
  objective         = "binary",
  metric            = "auc",
  learning_rate     = 0.05,
  num_leaves        = 31,           # controls tree complexity
  max_depth         = -1,           # -1 = no limit (controlled by num_leaves)
  min_data_in_leaf  = 20,           # regularisation
  feature_fraction  = 0.8,          # column subsampling
  bagging_fraction  = 0.8,          # row subsampling
  bagging_freq      = 5,
  lambda_l1         = 0.0,          # L1 regularisation
  lambda_l2         = 1.0,          # L2 regularisation
  verbose           = -1
)

set.seed(42)
lgb_base <- lgb.train(
  params    = lgb_params,
  data      = dtrain_lgb,
  nrounds   = 500,
  valids    = list(val = dval_lgb),
  early_stopping_rounds = 30,
  verbose   = -1
)

# Validation probabilities
lgb_val_prob <- predict(lgb_base, X_val)
cat("LGB AUC-ROC (val):", round(auc(roc(val$Churn, lgb_val_prob)), 4), "\n")
```

### 4e. Linear Base Learner — Logistic Regression / LASSO

```r
# LASSO logistic regression via glmnet
set.seed(42)
lasso_cv <- cv.glmnet(
  x        = X_train,
  y        = y_train,
  family   = "binomial",
  alpha    = 1,           # alpha=1 → LASSO; alpha=0 → Ridge; 0<a<1 → ElasticNet
  nfolds   = 5,
  type.measure = "auc"
)

best_lambda <- lasso_cv$lambda.min   # or lambda.1se for more regularisation
cat("Best lambda:", round(best_lambda, 6), "\n")

lasso_base <- glmnet(
  x      = X_train,
  y      = y_train,
  family = "binomial",
  alpha  = 1,
  lambda = best_lambda
)

# Validation probabilities
lasso_val_prob <- as.numeric(predict(lasso_base, newx = X_val, type = "response"))
cat("LAS AUC-ROC (val):", round(auc(roc(val$Churn, lasso_val_prob)), 4), "\n")
```

---

## 5. Ensemble Methods

### 5a. Simple Voting (Hard & Soft)

```r
# ── Hard voting: majority class wins ─────────────────────────────────
threshold <- 0.5

rf_vote    <- as.integer(rf_val_prob  >= threshold)
gbm_vote   <- as.integer(gbm_val_prob >= threshold)
xgb_vote   <- as.integer(xgb_val_prob >= threshold)
lgb_vote   <- as.integer(lgb_val_prob >= threshold)
lasso_vote <- as.integer(lasso_val_prob >= threshold)

# Sum votes (5 models — majority = 3+)
vote_sum         <- rf_vote + gbm_vote + xgb_vote + lgb_vote + lasso_vote
hard_vote_pred   <- factor(ifelse(vote_sum >= 3, "Churn", "Active"),
                           levels = c("Active", "Churn"))

cm_hard <- confusionMatrix(hard_vote_pred, val$Churn, positive = "Churn")
cat("Hard Voting F1:", round(cm_hard$byClass["F1"], 4), "\n")

# ── Soft voting: average predicted probabilities ──────────────────────
soft_prob <- (rf_val_prob + gbm_val_prob + xgb_val_prob +
              lgb_val_prob + lasso_val_prob) / 5

soft_vote_pred <- factor(ifelse(soft_prob >= threshold, "Churn", "Active"),
                         levels = c("Active", "Churn"))

cm_soft <- confusionMatrix(soft_vote_pred, val$Churn, positive = "Churn")
cat("Soft Voting AUC:", round(auc(roc(val$Churn, soft_prob)), 4), "\n")
cat("Soft Voting F1: ", round(cm_soft$byClass["F1"], 4), "\n")
```

### 5b. Weighted Average Ensemble

```r
# ── Compute weights from individual validation AUC ───────────────────
auc_rf    <- as.numeric(auc(roc(val$Churn, rf_val_prob)))
auc_gbm   <- as.numeric(auc(roc(val$Churn, gbm_val_prob)))
auc_xgb   <- as.numeric(auc(roc(val$Churn, xgb_val_prob)))
auc_lgb   <- as.numeric(auc(roc(val$Churn, lgb_val_prob)))
auc_lasso <- as.numeric(auc(roc(val$Churn, lasso_val_prob)))

aucs       <- c(rf = auc_rf, gbm = auc_gbm, xgb = auc_xgb,
                lgb = auc_lgb, lasso = auc_lasso)
weights    <- aucs / sum(aucs)   # normalise so they sum to 1
cat("Model weights:\n"); print(round(weights, 4))

# ── Weighted average probability ─────────────────────────────────────
weighted_prob <- weights["rf"]    * rf_val_prob    +
                 weights["gbm"]   * gbm_val_prob   +
                 weights["xgb"]   * xgb_val_prob   +
                 weights["lgb"]   * lgb_val_prob   +
                 weights["lasso"] * lasso_val_prob

weighted_pred <- factor(ifelse(weighted_prob >= threshold, "Churn", "Active"),
                        levels = c("Active", "Churn"))

cat("Weighted Avg AUC:", round(auc(roc(val$Churn, weighted_prob)), 4), "\n")
cat("Weighted Avg F1: ", round(confusionMatrix(weighted_pred, val$Churn,
                               positive = "Churn")$byClass["F1"], 4), "\n")

# ── Optimise weights via optim() ──────────────────────────────────────
prob_matrix <- cbind(rf_val_prob, gbm_val_prob, xgb_val_prob,
                     lgb_val_prob, lasso_val_prob)

y_val_bin <- as.integer(val$Churn == "Churn")

obj_fn <- function(w) {
  w      <- abs(w) / sum(abs(w))     # project to simplex
  p      <- prob_matrix %*% w
  roc_v  <- roc(y_val_bin, as.numeric(p), quiet = TRUE)
  -as.numeric(auc(roc_v))            # minimise negative AUC
}

set.seed(42)
opt_res  <- optim(rep(0.2, 5), obj_fn, method = "Nelder-Mead")
opt_w    <- abs(opt_res$par) / sum(abs(opt_res$par))
names(opt_w) <- c("rf", "gbm", "xgb", "lgb", "lasso")
cat("Optimised weights:\n"); print(round(opt_w, 4))
```

### 5c. Stacking (Meta-Learning)

> **Critical:** Base model predictions used to train the meta-learner **must** come from out-of-fold (OOF) predictions on training data. Using in-sample predictions causes severe data leakage.

```r
# ── Step 1: Generate OOF predictions from base models ────────────────
set.seed(42)
n_folds   <- 5
folds     <- createFolds(train$Churn, k = n_folds, list = TRUE)
n_train   <- nrow(train)

oof_rf    <- numeric(n_train)
oof_gbm   <- numeric(n_train)
oof_xgb   <- numeric(n_train)
oof_lgb   <- numeric(n_train)
oof_lasso <- numeric(n_train)

X_train_mat <- model.matrix(Churn ~ . - 1, data = train)
y_train_bin  <- as.integer(train$Churn == "Churn")
train_gbm_df <- train; train_gbm_df$Churn <- y_train_bin

for (fold_i in seq_len(n_folds)) {
  cat("Fold", fold_i, "of", n_folds, "\n")
  
  idx_val   <- folds[[fold_i]]
  idx_train <- setdiff(seq_len(n_train), idx_val)

  # Fold subsets
  tr <- train[idx_train, ];  va <- train[idx_val, ]
  tr_gbm <- train_gbm_df[idx_train, ];  va_gbm <- train_gbm_df[idx_val, ]
  X_tr <- X_train_mat[idx_train, ];     X_va <- X_train_mat[idx_val, ]
  y_tr <- y_train_bin[idx_train];       y_va <- y_train_bin[idx_val]

  # ── Random Forest ──────────────────────────────────────────────────
  m_rf <- randomForest(Churn ~ ., data = tr, ntree = 300,
                       mtry = floor(sqrt(ncol(tr) - 1)), nodesize = 5)
  oof_rf[idx_val] <- predict(m_rf, newdata = va, type = "prob")[, "Churn"]

  # ── GBM ────────────────────────────────────────────────────────────
  m_gbm <- gbm(Churn ~ ., data = tr_gbm, distribution = "bernoulli",
               n.trees = 300, interaction.depth = 4, shrinkage = 0.05,
               n.minobsinnode = 10, verbose = FALSE)
  oof_gbm[idx_val] <- predict(m_gbm, newdata = va_gbm,
                               n.trees = 300, type = "response")

  # ── XGBoost ────────────────────────────────────────────────────────
  dtr <- xgb.DMatrix(X_tr, label = y_tr)
  dva <- xgb.DMatrix(X_va, label = y_va)
  m_xgb <- xgb.train(
    params  = list(objective = "binary:logistic", eval_metric = "auc",
                   eta = 0.05, max_depth = 5, subsample = 0.8,
                   colsample_bytree = 0.8, min_child_weight = 5),
    data = dtr, nrounds = 200, verbose = 0)
  oof_xgb[idx_val] <- predict(m_xgb, dva)

  # ── LightGBM ───────────────────────────────────────────────────────
  dtr_lgb <- lgb.Dataset(X_tr, label = y_tr)
  m_lgb <- lgb.train(
    params  = list(objective = "binary", metric = "auc",
                   learning_rate = 0.05, num_leaves = 31,
                   min_data_in_leaf = 20, verbose = -1),
    data = dtr_lgb, nrounds = 200, verbose = -1)
  oof_lgb[idx_val] <- predict(m_lgb, X_va)

  # ── LASSO ──────────────────────────────────────────────────────────
  m_lasso <- cv.glmnet(X_tr, y_tr, family = "binomial", alpha = 1,
                       nfolds = 3, type.measure = "auc")
  oof_lasso[idx_val] <- as.numeric(
    predict(m_lasso, newx = X_va, s = "lambda.min", type = "response"))
}

cat("OOF AUC-RF:   ", round(auc(roc(train$Churn, oof_rf)),    4), "\n")
cat("OOF AUC-GBM:  ", round(auc(roc(train$Churn, oof_gbm)),   4), "\n")
cat("OOF AUC-XGB:  ", round(auc(roc(train$Churn, oof_xgb)),   4), "\n")
cat("OOF AUC-LGB:  ", round(auc(roc(train$Churn, oof_lgb)),   4), "\n")
cat("OOF AUC-LASSO:", round(auc(roc(train$Churn, oof_lasso)), 4), "\n")

# ── Step 2: Train meta-learner on OOF predictions ────────────────────
meta_train <- data.frame(
  rf    = oof_rf,
  gbm   = oof_gbm,
  xgb   = oof_xgb,
  lgb   = oof_lgb,
  lasso = oof_lasso,
  y     = y_train_bin
)

# Meta-learner: LASSO logistic regression (prevents meta overfitting)
set.seed(42)
meta_X <- as.matrix(meta_train[, 1:5])
meta_y <- meta_train$y

meta_cv <- cv.glmnet(meta_X, meta_y, family = "binomial",
                     alpha = 1, nfolds = 5, type.measure = "auc")
meta_model <- glmnet(meta_X, meta_y, family = "binomial",
                     alpha = 1, lambda = meta_cv$lambda.min)

# ── Step 3: Retrain base models on FULL training set ─────────────────
set.seed(42)
rf_final  <- randomForest(Churn ~ ., data = train, ntree = 500,
                           mtry = floor(sqrt(ncol(train) - 1)), nodesize = 5)
train_gbm_full <- train; train_gbm_full$Churn <- y_train_bin
gbm_final <- gbm(Churn ~ ., data = train_gbm_full, distribution = "bernoulli",
                  n.trees = 300, interaction.depth = 4, shrinkage = 0.05,
                  n.minobsinnode = 10, verbose = FALSE)
xgb_final <- xgb.train(
  params  = list(objective = "binary:logistic", eval_metric = "auc",
                 eta = 0.05, max_depth = 5, subsample = 0.8,
                 colsample_bytree = 0.8, min_child_weight = 5),
  data = xgb.DMatrix(X_train_mat, label = y_train_bin),
  nrounds = best_n_xgb, verbose = 0)
dtrain_lgb_full <- lgb.Dataset(X_train_mat, label = y_train_bin)
lgb_final <- lgb.train(
  params  = list(objective = "binary", metric = "auc",
                 learning_rate = 0.05, num_leaves = 31,
                 min_data_in_leaf = 20, verbose = -1),
  data = dtrain_lgb_full, nrounds = lgb_base$best_iter, verbose = -1)
lasso_final <- glmnet(X_train_mat, y_train_bin, family = "binomial",
                       alpha = 1, lambda = best_lambda)

# ── Step 4: Generate test predictions from base models ───────────────
X_test_mat <- model.matrix(Churn ~ . - 1, data = test)

test_rf    <- predict(rf_final, newdata = test, type = "prob")[, "Churn"]
test_gbm   <- predict(gbm_final, newdata = cbind(test[-which(names(test) == "Churn")],
               Churn = 0), n.trees = 300, type = "response")
test_xgb   <- predict(xgb_final, xgb.DMatrix(X_test_mat))
test_lgb   <- predict(lgb_final, X_test_mat)
test_lasso <- as.numeric(predict(lasso_final, newx = X_test_mat, type = "response"))

# ── Step 5: Meta-learner predicts final ensemble probability ──────────
meta_test <- as.matrix(data.frame(
  rf = test_rf, gbm = test_gbm, xgb = test_xgb,
  lgb = test_lgb, lasso = test_lasso))

stack_prob <- as.numeric(predict(meta_model, newx = meta_test, type = "response"))
stack_pred <- factor(ifelse(stack_prob >= threshold, "Churn", "Active"),
                     levels = c("Active", "Churn"))

cat("Stack AUC-ROC:", round(auc(roc(test$Churn, stack_prob)), 4), "\n")
cat("Stack F1:     ", round(confusionMatrix(stack_pred, test$Churn,
                            positive = "Churn")$byClass["F1"], 4), "\n")
```

### 5d. Blending

Blending is a simpler variant of stacking that uses a held-out **validation set** instead of cross-validation for OOF predictions. Faster but slightly more prone to overfit.

```r
# Base models are already trained on train set
# val set is used as the "blend set" — not seen during base model training

blend_train <- data.frame(
  rf    = rf_val_prob,
  gbm   = gbm_val_prob,
  xgb   = xgb_val_prob,
  lgb   = lgb_val_prob,
  lasso = lasso_val_prob,
  y     = as.integer(val$Churn == "Churn")
)

# Train meta-learner on blend set
set.seed(42)
blend_X   <- as.matrix(blend_train[, 1:5])
blend_y   <- blend_train$y

blend_cv  <- cv.glmnet(blend_X, blend_y, family = "binomial",
                       alpha = 1, nfolds = 5, type.measure = "auc")
blend_meta <- glmnet(blend_X, blend_y, family = "binomial",
                     alpha = 1, lambda = blend_cv$lambda.min)

# Predict on test set
blend_test_X <- as.matrix(data.frame(
  rf = test_rf, gbm = test_gbm, xgb = test_xgb,
  lgb = test_lgb, lasso = test_lasso))

blend_prob <- as.numeric(predict(blend_meta, newx = blend_test_X,
                                  type = "response"))
cat("Blend AUC-ROC:", round(auc(roc(test$Churn, blend_prob)), 4), "\n")
```

### 5e. caretEnsemble (automated stacking)

```r
library(caretEnsemble)

set.seed(42)
ctrl_ens <- trainControl(
  method          = "cv",
  number          = 5,
  classProbs      = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = "final",
  index           = createFolds(train$Churn, k = 5)  # same folds for all models
)

# ── Train a list of base models ───────────────────────────────────────
model_list <- caretList(
  Churn ~ .,
  data       = train,
  trControl  = ctrl_ens,
  metric     = "ROC",
  methodList = c("rf", "gbm", "glmnet"),
  tuneList   = list(
    xgbTree = caretModelSpec(
      method   = "xgbTree",
      tuneGrid = expand.grid(
        nrounds          = 200,
        max_depth        = 5,
        eta              = 0.05,
        gamma            = 0.1,
        colsample_bytree = 0.8,
        min_child_weight = 5,
        subsample        = 0.8
      )
    )
  )
)

# ── Correlation between model predictions (diversity check) ──────────
modelCor(resamples(model_list))
# Low correlation = more diverse = better ensemble candidates

# ── Stack with a LASSO meta-learner ──────────────────────────────────
stack_glmnet <- caretStack(
  model_list,
  method    = "glmnet",
  metric    = "ROC",
  trControl = trainControl(
    method          = "cv",
    number          = 5,
    classProbs      = TRUE,
    summaryFunction = twoClassSummary
  )
)

print(stack_glmnet)

# ── Predict ───────────────────────────────────────────────────────────
stack_prob_ens <- predict(stack_glmnet, newdata = test, type = "prob")[, "Churn"]
cat("caretEnsemble Stack AUC:", round(auc(roc(test$Churn, stack_prob_ens)), 4), "\n")
```

---

## 6. Regularisation Across Ensemble Members

```r
# ── Random Forest ─────────────────────────────────────────────────────
rf_reg <- randomForest(
  Churn ~ ., data = train,
  ntree    = 500,
  mtry     = 3,            # lower mtry → more diverse, less correlated trees
  nodesize = 15,           # larger leaves → less overfit
  maxnodes = 40,           # hard cap on leaves per tree
  sampsize = round(nrow(train) * 0.65),
  replace  = FALSE
)

# ── GBM ──────────────────────────────────────────────────────────────
gbm_reg <- gbm(
  Churn ~ ., data = train_gbm_full, distribution = "bernoulli",
  n.trees           = 1000,
  interaction.depth = 3,       # shallow trees (weak learners)
  shrinkage         = 0.01,    # very small learning rate
  n.minobsinnode    = 20,      # large min leaf
  bag.fraction      = 0.6,
  cv.folds          = 5,
  verbose           = FALSE
)

# ── XGBoost ───────────────────────────────────────────────────────────
xgb_reg_params <- list(
  objective        = "binary:logistic",
  eval_metric      = "auc",
  eta              = 0.01,       # small learning rate
  max_depth        = 4,
  subsample        = 0.7,
  colsample_bytree = 0.7,
  min_child_weight = 10,         # large → regularises
  gamma            = 0.5,        # requires meaningful gain to split
  lambda           = 2.0,        # L2 weight regularisation
  alpha            = 0.5         # L1 weight regularisation
)

# ── LightGBM ──────────────────────────────────────────────────────────
lgb_reg_params <- list(
  objective         = "binary",
  metric            = "auc",
  learning_rate     = 0.01,
  num_leaves        = 15,        # lower = less complex
  max_depth         = 5,
  min_data_in_leaf  = 30,
  feature_fraction  = 0.7,
  bagging_fraction  = 0.7,
  bagging_freq      = 5,
  lambda_l1         = 0.5,
  lambda_l2         = 2.0,
  min_gain_to_split = 0.1,       # minimum gain required to split
  verbose           = -1
)

# ── Meta-learner regularisation ───────────────────────────────────────
# Use lambda.1se instead of lambda.min for a sparser, more regularised meta-model
meta_model_reg <- glmnet(meta_X, meta_y, family = "binomial",
                          alpha  = 1,
                          lambda = meta_cv$lambda.1se)  # ← more regularised
```

### Regularisation reference

| Model | Key regularisation parameters | Effect |
|---|---|---|
| Random Forest | `nodesize` ↑, `mtry` ↓, `maxnodes` ↓ | Simpler trees, less overfit |
| GBM | `shrinkage` ↓, `n.minobsinnode` ↑, `bag.fraction` ↓ | Slower, more careful learning |
| XGBoost | `lambda` ↑, `alpha` ↑, `gamma` ↑, `min_child_weight` ↑ | L1/L2 + structural |
| LightGBM | `lambda_l1/l2` ↑, `min_data_in_leaf` ↑, `num_leaves` ↓ | Leaf-wise regularisation |
| LASSO | `lambda.1se` instead of `lambda.min` | Sparser, more conservative |
| Meta-learner | `alpha = 1` (LASSO) | Automatic model selection |

---

## 7. Handling Class Imbalance

```r
# ── Strategy 1: Class weights in each base model ──────────────────────
n_total  <- nrow(train)
n_churn  <- sum(train$Churn == "Churn")
n_active <- n_total - n_churn

# Compute weights
w_vec <- ifelse(train$Churn == "Churn",
                n_total / (2 * n_churn),
                n_total / (2 * n_active))

# randomForest
rf_wt <- randomForest(Churn ~ ., data = train, ntree = 500,
                       classwt = c(Active = n_total / (2 * n_active),
                                   Churn  = n_total / (2 * n_churn)))

# XGBoost — scale_pos_weight = ratio of negatives to positives
xgb_params_wt <- modifyList(xgb_params,
  list(scale_pos_weight = n_active / n_churn))  # e.g. 5.8 if 85/15 split

# LightGBM — is_unbalance or scale_pos_weight
lgb_params_wt <- modifyList(lgb_params,
  list(is_unbalance = TRUE))                    # auto-computes weights

# GBM — weights argument
train_gbm_wt <- train_gbm_full
gbm_wt <- gbm(Churn ~ ., data = train_gbm_wt, distribution = "bernoulli",
               n.trees = 300, interaction.depth = 4, shrinkage = 0.05,
               weights = w_vec, verbose = FALSE)

# ── Strategy 2: ROSE balanced training set ────────────────────────────
train_rose <- ROSE(Churn ~ ., data = train, seed = 42)$data

# ── Strategy 3: SMOTE balanced training set ───────────────────────────
library(DMwR2)
train_smote <- SMOTE(Churn ~ ., data = train,
                     perc.over = 200, perc.under = 150)

# ── Strategy 4: Threshold optimisation per model ──────────────────────
find_best_threshold <- function(prob, actual, metric = "F1") {
  thresholds <- seq(0.1, 0.9, by = 0.02)
  scores <- sapply(thresholds, function(t) {
    pred <- factor(ifelse(prob >= t, "Churn", "Active"),
                   levels = c("Active", "Churn"))
    cm   <- confusionMatrix(pred, actual, positive = "Churn")
    cm$byClass[metric]
  })
  thresholds[which.max(scores)]
}

thr_rf    <- find_best_threshold(test_rf,    test$Churn)
thr_stack <- find_best_threshold(stack_prob, test$Churn)
cat("Best threshold (RF):", thr_rf, "\n")
cat("Best threshold (Stack):", thr_stack, "\n")
```

---

## 8. Model Inspection — Under the Hood

### Feature importance comparison across models

```r
# ── Random Forest (MDA + MDGini) ──────────────────────────────────────
rf_imp <- data.frame(
  feature   = rownames(importance(rf_final)),
  RF_MDA    = importance(rf_final, type = 1)[, 1],
  RF_MDGini = importance(rf_final, type = 2)[, 1]
)

# ── GBM relative influence ────────────────────────────────────────────
gbm_imp <- summary(gbm_final, n.trees = 300, plotit = FALSE)
names(gbm_imp) <- c("feature", "GBM_RelInf")

# ── XGBoost importance ────────────────────────────────────────────────
xgb_imp <- xgb.importance(model = xgb_final)
# Types: "Gain" (reduction in loss), "Cover" (samples covered),
#        "Frequency" (times used in splits)

# ── LightGBM importance ───────────────────────────────────────────────
lgb_imp_gain <- lgb.importance(lgb_final, percentage = TRUE)

# ── LASSO coefficients ────────────────────────────────────────────────
lasso_coef <- coef(lasso_final, s = best_lambda)
lasso_coef_df <- data.frame(
  feature = rownames(lasso_coef),
  coef    = as.numeric(lasso_coef)
) %>% dplyr::filter(coef != 0) %>% dplyr::arrange(desc(abs(coef)))

# ── Combine top-10 importances into comparison table ─────────────────
imp_combined <- Reduce(function(a, b) merge(a, b, by = "feature", all = TRUE),
  list(
    rf_imp[, c("feature", "RF_MDA")],
    gbm_imp[, c("feature", "GBM_RelInf")],
    data.frame(feature = xgb_imp$Feature, XGB_Gain = xgb_imp$Gain)
  )
)

print(imp_combined[order(-imp_combined$RF_MDA, na.last = TRUE), ])
```

### XGBoost tree inspection

```r
# ── Dump first 3 trees as text ────────────────────────────────────────
xgb.dump(xgb_final, with_stats = TRUE)[1:50]

# ── Plot tree 1 (requires DiagrammeR) ────────────────────────────────
# install.packages("DiagrammeR")
xgb.plot.tree(model = xgb_final, trees = 0, render = TRUE)

# ── SHAP values (game-theoretic feature attribution) ─────────────────
shap_vals  <- predict(xgb_final,
                      xgb.DMatrix(X_test_mat),
                      predcontrib = TRUE)

# SHAP for first observation
shap_obs1  <- shap_vals[1, -ncol(shap_vals)]   # drop BIAS column
shap_df    <- data.frame(
  feature = colnames(X_test_mat),
  shap    = shap_obs1
) %>% dplyr::arrange(desc(abs(shap)))

print(head(shap_df, 15))

# SHAP summary plot
xgb.plot.shap(X_test_mat, model = xgb_final, top_n = 10)
```

### GBM partial dependence

```r
# ── Partial dependence: how does tenure affect P(Churn)? ─────────────
plot(gbm_final, i.var = "tenure",
     n.trees = best_n_gbm,
     main    = "Partial Dependence: tenure → P(Churn)",
     ylab    = "Log-odds contribution")

# ── 2-variable interaction PDP ────────────────────────────────────────
plot(gbm_final,
     i.var   = c("tenure", "MonthlyCharges"),
     n.trees = best_n_gbm,
     main    = "2-Way PDP: tenure x MonthlyCharges")
```

### Calibration check

```r
# Are predicted probabilities well calibrated?
calibration_df <- data.frame(
  prob  = stack_prob,
  label = as.integer(test$Churn == "Churn")
)

# Bin predictions into deciles
calibration_df$bin <- cut(calibration_df$prob,
                           breaks = seq(0, 1, by = 0.1),
                           include.lowest = TRUE)

cal_summary <- aggregate(cbind(prob, label) ~ bin,
                          data = calibration_df,
                          FUN  = mean)

ggplot(cal_summary, aes(x = prob, y = label)) +
  geom_point(color = "#1D9E75", size = 3) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "#888880") +
  geom_line(color = "#1D9E75") +
  labs(title = "Calibration Plot — Stacking Ensemble",
       x = "Mean Predicted Probability", y = "Fraction of Positives (Actual)") +
  xlim(0, 1) + ylim(0, 1)
```

---

## 9. Hyperparameter Tuning

### Grid search — each base model

```r
# ── XGBoost grid via caret ────────────────────────────────────────────
set.seed(42)
xgb_grid <- expand.grid(
  nrounds          = c(200, 400),
  max_depth        = c(4, 6),
  eta              = c(0.01, 0.05),
  gamma            = c(0, 0.1),
  colsample_bytree = c(0.7, 0.9),
  min_child_weight = c(5, 10),
  subsample        = c(0.7, 0.9)
)

ctrl_xgb <- trainControl(
  method          = "cv",
  number          = 5,
  classProbs      = TRUE,
  summaryFunction = twoClassSummary,
  verboseIter     = FALSE
)

# Note: large grid — use parallel or subsample grid for speed
xgb_caret <- train(
  x         = X_train,
  y         = factor(y_train, labels = c("Active", "Churn")),
  method    = "xgbTree",
  metric    = "ROC",
  trControl = ctrl_xgb,
  tuneGrid  = xgb_grid[sample(nrow(xgb_grid), 20), ]  # random subsample of grid
)

print(xgb_caret$bestTune)

# ── GBM grid ──────────────────────────────────────────────────────────
gbm_grid <- expand.grid(
  n.trees           = c(200, 500),
  interaction.depth = c(3, 5),
  shrinkage         = c(0.01, 0.05),
  n.minobsinnode    = c(10, 20)
)

gbm_caret <- train(
  Churn ~ ., data = train,
  method    = "gbm",
  metric    = "ROC",
  trControl = ctrl_xgb,
  tuneGrid  = gbm_grid,
  verbose   = FALSE
)

print(gbm_caret$bestTune)
```

### Parallel tuning with `doParallel`

```r
library(doParallel)

# Register cores
cl <- makeCluster(parallel::detectCores() - 1)
registerDoParallel(cl)

ctrl_par <- trainControl(
  method          = "cv",
  number          = 5,
  classProbs      = TRUE,
  summaryFunction = twoClassSummary,
  allowParallel   = TRUE    # ← enable parallel fold evaluation
)

rf_tuned <- train(
  Churn ~ ., data = train,
  method    = "rf",
  metric    = "ROC",
  trControl = ctrl_par,
  tuneGrid  = expand.grid(mtry = c(3, 5, 7, 9))
)

stopCluster(cl)
registerDoSEQ()

print(rf_tuned$bestTune)
```

### Tuning the meta-learner

```r
# Tune the stacking meta-learner's regularisation
alpha_grid  <- c(0, 0.25, 0.5, 0.75, 1)   # 0=Ridge, 1=LASSO

meta_results <- sapply(alpha_grid, function(a) {
  cv_m <- cv.glmnet(meta_X, meta_y,
                    family = "binomial",
                    alpha  = a,
                    nfolds = 5,
                    type.measure = "auc")
  max(cv_m$cvm)   # max mean AUC across lambdas
})

best_alpha <- alpha_grid[which.max(meta_results)]
cat("Best meta alpha:", best_alpha,
    "| AUC:", round(max(meta_results), 4), "\n")
```

---

## 10. Save & Load Ensemble Models

```r
# ── Save all components as a single bundle ────────────────────────────
ensemble_bundle <- list(
  # Base models
  rf_model    = rf_final,
  gbm_model   = gbm_final,
  gbm_ntrees  = best_n_gbm,
  xgb_model   = xgb_base,
  lgb_model   = lgb_base,
  lasso_model = lasso_final,

  # Meta-learner
  meta_model  = meta_model,
  blend_meta  = blend_meta,

  # Weights for weighted average
  opt_weights = opt_w,

  # Optimal thresholds
  threshold_stack  = thr_stack,
  threshold_rf     = thr_rf,

  # Preprocessing info
  train_medians    = train_medians,
  cat_col_levels   = lapply(train[cat_cols], levels),
  feature_cols     = setdiff(names(train), "Churn"),
  matrix_cols      = colnames(X_train_mat),

  # Metadata
  version          = "v1.0",
  trained_on       = Sys.Date(),
  train_rows       = nrow(train),
  auc_stack_test   = round(auc(roc(test$Churn, stack_prob)), 4)
)

saveRDS(ensemble_bundle, "models/ensemble_churn_v1.rds")

# ── Reload ────────────────────────────────────────────────────────────
bundle <- readRDS("models/ensemble_churn_v1.rds")
cat("Loaded ensemble version:", bundle$version, "\n")
cat("Trained on:", as.character(bundle$trained_on), "\n")
cat("Stack AUC:", bundle$auc_stack_test, "\n")

# ── Save XGBoost model separately (binary format) ─────────────────────
xgb.save(xgb_final, "models/xgb_churn_v1.model")
xgb_reloaded <- xgb.load("models/xgb_churn_v1.model")

# ── Save LightGBM model separately ───────────────────────────────────
lgb.save(lgb_final, "models/lgb_churn_v1.txt")
lgb_reloaded <- lgb.load("models/lgb_churn_v1.txt")
```

---

## 11. Predict on New Data & Join Back

```r
# ── Load saved bundle ─────────────────────────────────────────────────
bundle <- readRDS("models/ensemble_churn_v1.rds")

# ── Load new data ─────────────────────────────────────────────────────
new_data <- read.csv("new_customers.csv", stringsAsFactors = TRUE)

# Save CustomerIDs before removing them
cust_ids <- new_data$CustomerID
new_data$CustomerID <- NULL

# ── Apply SAME feature engineering ───────────────────────────────────
new_data$tenure_sq   <- new_data$tenure^2
new_data$avg_charges <- new_data$TotalCharges / pmax(new_data$tenure, 1)
new_data$high_value  <- ifelse(new_data$MonthlyCharges > 70, 1L, 0L)

# Impute using training medians
for (col in names(bundle$train_medians)) {
  if (col %in% names(new_data)) {
    new_data[[col]][is.na(new_data[[col]])] <- bundle$train_medians[[col]]
  }
}

# Align factor levels to training
for (col in names(bundle$cat_col_levels)) {
  if (col %in% names(new_data)) {
    new_data[[col]] <- factor(new_data[[col]],
                              levels = bundle$cat_col_levels[[col]])
  }
}

# Build numeric matrix for XGBoost / LightGBM / LASSO
X_new <- model.matrix(~ . - 1, data = new_data)
# Ensure same columns as training matrix
missing_cols <- setdiff(bundle$matrix_cols, colnames(X_new))
extra_cols   <- setdiff(colnames(X_new), bundle$matrix_cols)
if (length(missing_cols) > 0) {
  for (mc in missing_cols) X_new <- cbind(X_new, setNames(data.frame(0), mc))
}
X_new <- X_new[, bundle$matrix_cols]   # reorder to match training

# ── Score all base models ─────────────────────────────────────────────
new_rf <- predict(bundle$rf_model, newdata = new_data, type = "prob")[, "Churn"]

new_gbm <- predict(bundle$gbm_model,
                   newdata  = cbind(new_data, Churn = 0),
                   n.trees  = bundle$gbm_ntrees,
                   type     = "response")

new_xgb   <- predict(bundle$xgb_model, xgb.DMatrix(X_new))
new_lgb   <- predict(bundle$lgb_model, X_new)
new_lasso <- as.numeric(predict(bundle$lasso_model,
                                 newx = X_new, type = "response"))

# ── Ensemble predictions ──────────────────────────────────────────────

# 1. Soft voting (simple average)
soft_prob_new <- (new_rf + new_gbm + new_xgb + new_lgb + new_lasso) / 5

# 2. Weighted average
weighted_prob_new <- bundle$opt_weights["rf"]    * new_rf    +
                     bundle$opt_weights["gbm"]   * new_gbm   +
                     bundle$opt_weights["xgb"]   * new_xgb   +
                     bundle$opt_weights["lgb"]   * new_lgb   +
                     bundle$opt_weights["lasso"] * new_lasso

# 3. Stacking (meta-learner)
meta_new_X <- as.matrix(data.frame(
  rf = new_rf, gbm = new_gbm, xgb = new_xgb,
  lgb = new_lgb, lasso = new_lasso))

stack_prob_new <- as.numeric(predict(bundle$meta_model,
                                      newx  = meta_new_X,
                                      type  = "response"))

# ── Apply optimal threshold for binary label ──────────────────────────
thr <- bundle$threshold_stack

Prediction <- factor(ifelse(stack_prob_new >= thr, "Churn", "Active"),
                     levels = c("Active", "Churn"))

# ── Build scored data frame ───────────────────────────────────────────
scored <- data.frame(
  prob_rf       = round(new_rf,          4),
  prob_gbm      = round(new_gbm,         4),
  prob_xgb      = round(new_xgb,         4),
  prob_lgb      = round(new_lgb,         4),
  prob_lasso    = round(new_lasso,       4),
  prob_soft_avg = round(soft_prob_new,   4),
  prob_weighted = round(weighted_prob_new, 4),
  prob_stack    = round(stack_prob_new,  4),
  Prediction    = as.character(Prediction)
)

# ── Restore CustomerID and join back ──────────────────────────────────
new_data$CustomerID <- cust_ids

final_df <- cbind(new_data, scored)

# Sort by churn risk (highest first)
final_df <- final_df[order(-final_df$prob_stack), ]

# ── Preview ───────────────────────────────────────────────────────────
print(head(final_df[, c("CustomerID", "tenure", "MonthlyCharges",
                         "prob_stack", "Prediction")], 10))
# CustomerID  tenure  MonthlyCharges  prob_stack  Prediction
# C0001           3         82.45       0.9124       Churn
# C0003           8         91.10       0.8867       Churn
# C0012           5         78.30       0.7643       Churn
# C0002          42         55.30       0.0891       Active
# C0004          60         34.20       0.0312       Active

cat("Total scored:    ", nrow(final_df), "\n")
cat("Predicted Churn: ", sum(final_df$Prediction == "Churn"), "\n")
cat("Predicted Active:", sum(final_df$Prediction == "Active"), "\n")

# ── Write output ──────────────────────────────────────────────────────
write.csv(final_df, "output/ensemble_churn_predictions.csv", row.names = FALSE)
```

---

## 12. Compare All Models

```r
# ── Collect test AUC-ROC for every model & ensemble ───────────────────
compare_models <- function(prob_list, actuals) {
  data.frame(
    Model   = names(prob_list),
    AUC_ROC = sapply(prob_list, function(p)
                round(as.numeric(auc(roc(actuals, p, quiet = TRUE))), 4)),
    F1      = sapply(prob_list, function(p) {
                pred <- factor(ifelse(p >= 0.5, "Churn", "Active"),
                               levels = c("Active", "Churn"))
                round(confusionMatrix(pred, actuals,
                      positive = "Churn")$byClass["F1"], 4)
              })
  )
}

prob_list <- list(
  "Random Forest"   = test_rf,
  "GBM"             = test_gbm,
  "XGBoost"         = test_xgb,
  "LightGBM"        = test_lgb,
  "LASSO"           = test_lasso,
  "Hard Voting"     = as.numeric(vote_sum / 5),
  "Soft Voting"     = (test_rf + test_gbm + test_xgb + test_lgb + test_lasso) / 5,
  "Weighted Avg"    = as.numeric(prob_matrix %*% opt_w),
  "Stacking"        = stack_prob,
  "Blending"        = blend_prob
)

results_df <- compare_models(prob_list, test$Churn)
results_df <- results_df[order(-results_df$AUC_ROC), ]
print(results_df, row.names = FALSE)

# ── Visual comparison ─────────────────────────────────────────────────
ggplot(results_df, aes(x = reorder(Model, AUC_ROC), y = AUC_ROC,
                        fill = AUC_ROC)) +
  geom_col() +
  geom_text(aes(label = AUC_ROC), hjust = -0.1, size = 3.5) +
  coord_flip(ylim = c(min(results_df$AUC_ROC) - 0.02, 1.0)) +
  scale_fill_gradient(low = "#9FE1CB", high = "#0F6E56") +
  labs(title = "Model Comparison — AUC-ROC on Test Set",
       x = NULL, y = "AUC-ROC") +
  theme_minimal() +
  theme(legend.position = "none")

# ── ROC curves overlaid ───────────────────────────────────────────────
roc_rf    <- roc(test$Churn, test_rf,    quiet = TRUE)
roc_xgb   <- roc(test$Churn, test_xgb,  quiet = TRUE)
roc_stack <- roc(test$Churn, stack_prob, quiet = TRUE)

plot(roc_rf,    col = "#1D9E75", lwd = 2,
     main = "ROC Curves — Base vs Ensemble")
plot(roc_xgb,   col = "#534AB7", lwd = 2, add = TRUE)
plot(roc_stack, col = "#A32D2D", lwd = 2.5, add = TRUE)
legend("bottomright",
       legend = c(paste("RF     AUC:", round(auc(roc_rf), 3)),
                  paste("XGBoost AUC:", round(auc(roc_xgb), 3)),
                  paste("Stack  AUC:", round(auc(roc_stack), 3))),
       col  = c("#1D9E75", "#534AB7", "#A32D2D"),
       lwd  = c(2, 2, 2.5), bty = "n")
```

---

## 13. Quick-Reference Cheatsheet

### Ensemble method summary

| Method | Trains on | Combines via | Leakage risk | Complexity |
|---|---|---|---|---|
| Hard voting | Independent | Majority vote | None | Low |
| Soft voting | Independent | Average probs | None | Low |
| Weighted avg | Independent | Weighted sum | Low (val set) | Low |
| Blending | Train → val split | Meta-model on val | Medium | Medium |
| Stacking (OOF) | K-fold CV | Meta-model on OOF | Low (if done right) | High |
| caretEnsemble | K-fold CV | Auto meta-learner | Low | Medium |

### Model arguments at a glance

| Argument | RF | GBM | XGBoost | LightGBM |
|---|---|---|---|---|
| Trees | `ntree` | `n.trees` | `nrounds` | `nrounds` |
| Learning rate | — | `shrinkage` | `eta` | `learning_rate` |
| Tree depth | — | `interaction.depth` | `max_depth` | `max_depth` |
| Min leaf size | `nodesize` | `n.minobsinnode` | `min_child_weight` | `min_data_in_leaf` |
| Row sampling | `sampsize` | `bag.fraction` | `subsample` | `bagging_fraction` |
| Col sampling | `mtry` | — | `colsample_bytree` | `feature_fraction` |
| L2 | — | — | `lambda` | `lambda_l2` |
| L1 | — | — | `alpha` | `lambda_l1` |
| Class weight | `classwt` | `weights` | `scale_pos_weight` | `is_unbalance` |

### Key evaluation functions

| Metric | Function | Package |
|---|---|---|
| AUC-ROC | `auc(roc(...))` | `pROC` |
| AUC-PR | `pr.curve(...)` | `PRROC` |
| Confusion matrix / F1 | `confusionMatrix(...)` | `caret` |
| SHAP values | `predict(..., predcontrib=TRUE)` | `xgboost` |
| Partial dependence | `plot(gbm, i.var=...)` | `gbm` |
| Feature importance | `importance()` / `xgb.importance()` | `randomForest` / `xgboost` |

### Recommended file structure

```
project/
├── data/
│   ├── churn.csv
│   └── new_customers.csv
├── models/
│   ├── ensemble_churn_v1.rds    # full bundle (all models + metadata)
│   ├── xgb_churn_v1.model       # XGBoost binary
│   └── lgb_churn_v1.txt         # LightGBM text
├── output/
│   └── ensemble_churn_predictions.csv
└── ensemble_churn_model.R
```

### Decision guide — which ensemble to use

```
Start here
    │
    ├── Do you have < 1 hour to train?
    │       └── YES → Soft voting or Weighted average
    │
    ├── Do you want maximum predictive performance?
    │       └── YES → Stacking with OOF predictions
    │                  (use LASSO as meta-learner to prevent overfit)
    │
    ├── Do you have a clean validation set held out?
    │       └── YES → Blending (faster than full stacking)
    │
    ├── Do you want automated model selection?
    │       └── YES → caretEnsemble
    │
    └── Are your base models highly correlated (modelCor > 0.95)?
            └── YES → Add a more diverse base learner before ensembling
                       (e.g. add LASSO if you only have tree models)
```

---

*Base R Ensemble Models · RF · GBM · XGBoost · LightGBM · Stacking · Blending · Churn Prediction*
