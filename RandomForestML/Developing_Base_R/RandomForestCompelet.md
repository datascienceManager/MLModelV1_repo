# Random Forest in Base R — Complete Guide
### Churn Prediction · Binary Output: `"Churn"` / `"Active"`

> **Stack:** `randomForest` · `caret` · `ROSE` / `DMwR2` · `ranger` (fast alternative) · base R

---

## Table of Contents

1. [Setup & Packages](#1-setup--packages)
2. [Load & Explore Data](#2-load--explore-data)
3. [Feature Engineering & Train/Test Split](#3-feature-engineering--traintest-split)
4. [Build the Random Forest Model](#4-build-the-random-forest-model)
5. [Evaluate the Model](#5-evaluate-the-model)
6. [Handling Class Imbalance](#6-handling-class-imbalance)
7. [Regularisation](#7-regularisation)
8. [Tree Details — Under the Hood](#8-tree-details--under-the-hood)
9. [Hyperparameter Tuning](#9-hyperparameter-tuning)
10. [Save & Load the Model](#10-save--load-the-model)
11. [Predict on New Data & Join Back](#11-predict-on-new-data--join-back)
12. [Quick-Reference Cheatsheet](#12-quick-reference-cheatsheet)

---

## 1. Setup & Packages

```r
# Core packages
install.packages("randomForest")   # classic RF implementation
install.packages("ranger")         # fast C++ RF — drop-in replacement
install.packages("caret")          # unified train/tune/evaluate wrapper
install.packages("ROSE")           # oversampling / undersampling
install.packages("DMwR2")          # SMOTE (synthetic minority oversampling)
install.packages("pROC")           # AUC-ROC curves
install.packages("ggplot2")        # visualisation

library(randomForest)
library(ranger)
library(caret)
library(ROSE)
library(pROC)
library(ggplot2)
```

> **`randomForest` vs `ranger`:** Both use the same algorithm. `ranger` is 10–100× faster on large datasets and supports parallelism natively. The API is very similar — examples use `randomForest` but `ranger` equivalents are shown where they differ.

---

## 2. Load & Explore Data

```r
# Load data
df <- read.csv("churn.csv", stringsAsFactors = TRUE)

# First look
str(df)
summary(df)
head(df)

# ── Class balance check ──────────────────────────────────────────────
table(df$Churn)
prop.table(table(df$Churn)) * 100
# e.g.  Active: 85.4%   Churn: 14.6%  ← imbalanced!

# ── Missing values ───────────────────────────────────────────────────
colSums(is.na(df))

# ── Numeric summary by class ─────────────────────────────────────────
aggregate(. ~ Churn, data = df[, sapply(df, is.numeric)], FUN = mean)
```

---

## 3. Feature Engineering & Train/Test Split

```r
# ── 1. Target variable: factor with meaningful levels ────────────────
df$Churn <- factor(df$Churn,
                   levels = c(0, 1),
                   labels = c("Active", "Churn"))

# ── 2. Feature engineering ───────────────────────────────────────────
df$tenure_sq   <- df$tenure^2                               # non-linear tenure effect
df$avg_charges <- df$TotalCharges / pmax(df$tenure, 1)      # charge per month
df$high_value  <- ifelse(df$MonthlyCharges > 70, 1L, 0L)   # binary flag

# ── 3. Handle missing values ─────────────────────────────────────────
# Median imputation for numeric columns
num_cols <- sapply(df, is.numeric)
df[num_cols] <- lapply(df[num_cols], function(x) {
  x[is.na(x)] <- median(x, na.rm = TRUE)
  x
})

# ── 4. Encode categorical variables ──────────────────────────────────
# randomForest handles factors natively — just ensure they are factors
cat_cols <- c("Contract", "InternetService", "PaymentMethod")
df[cat_cols] <- lapply(df[cat_cols], factor)

# ── 5. Remove ID column (not a predictor) ────────────────────────────
df$CustomerID <- NULL

# ── 6. Stratified train / test split (80/20) ─────────────────────────
set.seed(42)
train_idx <- createDataPartition(df$Churn, p = 0.80, list = FALSE)
train <- df[ train_idx, ]
test  <- df[-train_idx, ]

cat("Train rows:", nrow(train), " | Test rows:", nrow(test), "\n")
prop.table(table(train$Churn))   # confirm stratification
```

---

## 4. Build the Random Forest Model

### Using `randomForest`

```r
set.seed(42)

rf_model <- randomForest(
  formula    = Churn ~ .,
  data       = train,
  ntree      = 500,          # number of trees (more = more stable)
  mtry       = floor(sqrt(ncol(train) - 1)),  # features per split (sqrt for classification)
  nodesize   = 5,            # minimum samples in terminal leaf
  maxnodes   = NULL,         # NULL = grow fully (control via nodesize)
  importance = TRUE,         # compute feature importance
  keep.forest = TRUE,        # keep trees for prediction
  sampsize   = nrow(train),  # bootstrap sample size (default = nrow)
  replace    = TRUE          # sampling with replacement (bagging)
)

print(rf_model)
# OOB estimate of  error rate: X.XX%
# Confusion matrix:
#         Active Churn class.error
# Active    XXXX   XXX      0.0XX
# Churn      XXX   XXX      0.XXX
```

### Using `ranger` (faster alternative)

```r
set.seed(42)

rf_ranger <- ranger(
  formula        = Churn ~ .,
  data           = train,
  num.trees      = 500,
  mtry           = floor(sqrt(ncol(train) - 1)),
  min.node.size  = 5,
  importance     = "impurity",    # "impurity" (Gini) or "permutation"
  probability    = TRUE,          # return class probabilities
  num.threads    = parallel::detectCores() - 1,  # parallel training
  seed           = 42
)

print(rf_ranger)
```

---

## 5. Evaluate the Model

```r
# ── Predictions ──────────────────────────────────────────────────────
pred_class <- predict(rf_model, newdata = test, type = "class")
pred_prob  <- predict(rf_model, newdata = test, type = "prob")

# ── Confusion matrix ─────────────────────────────────────────────────
cm <- confusionMatrix(pred_class, test$Churn, positive = "Churn")
print(cm)
# Reports: Accuracy, Sensitivity (Recall), Specificity,
#          Precision (Pos Pred Value), F1, Kappa

# ── AUC-ROC ──────────────────────────────────────────────────────────
roc_obj <- roc(
  response  = test$Churn,
  predictor = pred_prob[, "Churn"],
  levels    = c("Active", "Churn"),
  direction = "<"
)
cat("AUC-ROC:", auc(roc_obj), "\n")
plot(roc_obj, main = "ROC Curve", col = "#1D9E75", lwd = 2)

# ── Precision-Recall (better for imbalanced data) ────────────────────
library(PRROC)
pr <- pr.curve(
  scores.class0 = pred_prob[test$Churn == "Churn",  "Churn"],
  scores.class1 = pred_prob[test$Churn == "Active", "Churn"],
  curve = TRUE
)
plot(pr, main = "Precision-Recall Curve")
cat("AUC-PR:", pr$auc.integral, "\n")

# ── OOB error (free estimate — no separate val set needed) ───────────
plot(rf_model,
     main = "OOB Error vs Number of Trees",
     col  = c("black", "#1D9E75", "#A32D2D"))
legend("topright",
       legend = c("OOB Overall", "Active", "Churn"),
       col    = c("black", "#1D9E75", "#A32D2D"),
       lty    = 1)
```

> **Metrics priority for imbalanced churn data:**
> 1. AUC-PR (most informative)
> 2. F1-Score
> 3. AUC-ROC
> 4. ~~Accuracy~~ (misleading — avoid as primary metric)

---

## 6. Handling Class Imbalance

### Strategy 1 · Class weights (`classwt` in `randomForest`)

```r
# Compute inverse-frequency weights
n_total  <- nrow(train)
n_churn  <- sum(train$Churn == "Churn")
n_active <- sum(train$Churn == "Active")

w_churn  <- n_total / (2 * n_churn)
w_active <- n_total / (2 * n_active)

set.seed(42)
rf_weighted <- randomForest(
  Churn ~ .,
  data     = train,
  ntree    = 500,
  mtry     = floor(sqrt(ncol(train) - 1)),
  classwt  = c(Active = w_active, Churn = w_churn),  # ← key argument
  importance = TRUE
)
```

### Strategy 2 · Oversample minority (ROSE)

```r
library(ROSE)

# Synthetic oversampling + undersampling combined
train_rose <- ROSE(
  Churn ~ .,
  data = train,
  seed = 42,
  N    = nrow(train)   # keep same total size
)$data

table(train_rose$Churn)   # confirm balance

set.seed(42)
rf_rose <- randomForest(Churn ~ ., data = train_rose, ntree = 500)
```

### Strategy 3 · SMOTE (synthetic minority oversampling)

```r
library(DMwR2)

# SMOTE: generate synthetic minority samples via k-NN interpolation
train_smote <- SMOTE(
  Churn ~ .,
  data  = train,
  perc.over = 200,   # oversample minority by 200% (3× original)
  perc.under = 150   # undersample majority to 150% of minority
)

table(train_smote$Churn)

set.seed(42)
rf_smote <- randomForest(Churn ~ ., data = train_smote, ntree = 500)
```

### Strategy 4 · Manual oversampling (no extra packages)

```r
majority <- train[train$Churn == "Active", ]
minority  <- train[train$Churn == "Churn",  ]

# Oversample minority to match majority size
set.seed(42)
minority_over <- minority[sample(nrow(minority),
                                 size    = nrow(majority),
                                 replace = TRUE), ]

train_balanced <- rbind(majority, minority_over)
train_balanced <- train_balanced[sample(nrow(train_balanced)), ]  # shuffle

table(train_balanced$Churn)
```

### Strategy 5 · Threshold tuning (post-model)

```r
# Default threshold = 0.5 — lower to catch more churners
pred_prob <- predict(rf_model, newdata = test, type = "prob")

# Custom threshold
threshold <- 0.35
pred_adjusted <- factor(
  ifelse(pred_prob[, "Churn"] >= threshold, "Churn", "Active"),
  levels = c("Active", "Churn")
)

# Evaluate adjusted predictions
confusionMatrix(pred_adjusted, test$Churn, positive = "Churn")

# ── Find optimal threshold via F1 ────────────────────────────────────
thresholds <- seq(0.1, 0.9, by = 0.05)
f1_scores  <- sapply(thresholds, function(t) {
  pred <- factor(ifelse(pred_prob[, "Churn"] >= t, "Churn", "Active"),
                 levels = c("Active", "Churn"))
  cm  <- confusionMatrix(pred, test$Churn, positive = "Churn")
  cm$byClass["F1"]
})

best_threshold <- thresholds[which.max(f1_scores)]
cat("Best threshold:", best_threshold, " | Best F1:", max(f1_scores, na.rm = TRUE), "\n")

plot(thresholds, f1_scores, type = "l", col = "#1D9E75", lwd = 2,
     xlab = "Threshold", ylab = "F1 Score",
     main = "F1 Score vs Classification Threshold")
abline(v = best_threshold, col = "#A32D2D", lty = 2)
```

| Strategy | Best for | Notes |
|---|---|---|
| `classwt` | Mild imbalance < 10:1 | Simplest — try first |
| ROSE | Moderate imbalance | Combines over + undersampling |
| SMOTE | Small minority class | Synthetic interpolation via k-NN |
| Manual oversample | Any ratio | No extra packages needed |
| Threshold tuning | Any | Post-hoc, tune precision/recall trade-off |

---

## 7. Regularisation

> **Note:** Random Forests do not have L1/L2 penalties (those are for linear models). Regularisation is achieved via **structural constraints** on tree growth and **subsampling**.

```r
set.seed(42)

rf_regularised <- randomForest(
  Churn ~ .,
  data     = train,

  # ── Tree complexity ─────────────────────────────────────────────────
  ntree    = 500,          # more trees = more stable, not more overfit
  mtry     = 4,            # fewer features per split → more regularised
  maxnodes = 50,           # hard cap on leaf nodes per tree

  # ── Leaf size (most important regularisation lever) ─────────────────
  nodesize = 10,           # min samples in terminal node (default=1)
                           # increase → simpler trees → less overfit

  # ── Subsampling ─────────────────────────────────────────────────────
  sampsize = round(nrow(train) * 0.7),  # use only 70% of data per tree
  replace  = FALSE,        # sampling WITHOUT replacement (stricter)

  importance = TRUE
)
```

### Pre-processing regularisation (remove noisy features)

```r
# ── Remove near-zero variance features ──────────────────────────────
nzv_cols <- nearZeroVar(train, names = TRUE)
train_clean <- train[, !names(train) %in% nzv_cols]

# ── Remove highly correlated features (r > 0.90) ────────────────────
num_data  <- train_clean[, sapply(train_clean, is.numeric)]
cor_matrix <- cor(num_data, use = "complete.obs")
high_cor  <- findCorrelation(cor_matrix, cutoff = 0.90, names = TRUE)
train_clean <- train_clean[, !names(train_clean) %in% high_cor]

cat("Removed NZV:", length(nzv_cols),
    "| Removed high-cor:", length(high_cor), "\n")

# ── Recursive Feature Elimination (RFE via caret) ───────────────────
set.seed(42)
ctrl_rfe <- rfeControl(
  functions = rfFuncs,
  method    = "cv",
  number    = 5,
  verbose   = FALSE
)

rfe_result <- rfe(
  x      = train_clean[, names(train_clean) != "Churn"],
  y      = train_clean$Churn,
  sizes  = c(5, 10, 15, 20),
  rfeControl = ctrl_rfe
)

print(rfe_result)
cat("Optimal features:", predictors(rfe_result), "\n")
```

| Parameter | Values | Effect |
|---|---|---|
| `nodesize` ↑ | 5 → 20 | Larger leaves → simpler model → less overfit |
| `maxnodes` ↓ | 100 → 30 | Fewer splits → less complex trees |
| `mtry` ↓ | sqrt → 2 | Less feature info per split → more diverse trees |
| `sampsize` ↓ | 100% → 63% | Less data per tree → higher variance between trees |
| `replace = FALSE` | — | Subsampling without replacement (more regularising) |
| `ntree` ↑ | 100 → 1000 | More stable — does NOT cause overfitting |

---

## 8. Tree Details — Under the Hood

### Extract individual tree structure

```r
# ── Get the node table of a single tree ─────────────────────────────
tree_1 <- getTree(rf_model,
                  k     = 1,      # which tree (1 to ntree)
                  labelVar = TRUE)  # use feature names instead of indices

head(tree_1, 20)
# Columns: left daughter, right daughter, split var,
#          split point, status (-1=terminal), prediction

# ── Count nodes and depth for every tree ────────────────────────────
n_trees  <- rf_model$ntree
depths   <- numeric(n_trees)
n_nodes  <- numeric(n_trees)

for (i in seq_len(n_trees)) {
  t         <- getTree(rf_model, k = i)
  n_nodes[i] <- nrow(t)
  # Depth = longest path from root (row 1) to a terminal node
  terminal  <- which(t[, "status"] == -1)
  depths[i] <- max(sapply(terminal, function(leaf) {
    depth <- 0
    node  <- leaf
    while (node != 1) {
      node  <- which(t[, "left daughter"] == node |
                     t[, "right daughter"] == node)
      depth <- depth + 1
    }
    depth
  }))
}

tree_stats <- data.frame(tree = 1:n_trees, depth = depths, nodes = n_nodes)
summary(tree_stats)

ggplot(tree_stats, aes(x = depth)) +
  geom_histogram(bins = 20, fill = "#1D9E75", color = "white") +
  labs(title = "Distribution of Tree Depths",
       x = "Tree Depth", y = "Count")
```

### Feature importance

```r
# ── Mean Decrease Accuracy (permutation importance) ──────────────────
# How much accuracy drops when a feature is randomly shuffled
imp <- importance(rf_model, type = 1)  # type=1: MDA, type=2: MDG

# ── Mean Decrease Gini ───────────────────────────────────────────────
# How much each feature reduces node impurity across all trees
imp_gini <- importance(rf_model, type = 2)

# Combine into a data frame
imp_df <- data.frame(
  feature  = rownames(imp),
  MDA      = imp[, "MeanDecreaseAccuracy"],
  MDGini   = imp_gini[, "MeanDecreaseGini"]
) %>%
  dplyr::arrange(desc(MDA))

print(imp_df)

# Visualise
varImpPlot(rf_model, main = "Variable Importance", type = 1, col = "#0F6E56")

# ggplot version
ggplot(imp_df[1:15, ], aes(x = reorder(feature, MDA), y = MDA)) +
  geom_col(fill = "#1D9E75") +
  coord_flip() +
  labs(title = "Feature Importance (Mean Decrease Accuracy)",
       x = NULL, y = "MDA")
```

### Proximity matrix (observe cluster structure)

```r
# Proximity = fraction of trees where two observations share a terminal node
# Computationally heavy — use a sample
set.seed(42)
sample_idx <- sample(nrow(train), 500)
train_sample <- train[sample_idx, ]

rf_prox <- randomForest(
  Churn ~ .,
  data      = train_sample,
  ntree     = 200,
  proximity = TRUE   # ← enable proximity matrix
)

# Multidimensional scaling of proximity
prox_mds <- cmdscale(1 - rf_prox$proximity, k = 2)
plot(prox_mds,
     col  = c("#1D9E75", "#A32D2D")[as.integer(train_sample$Churn)],
     pch  = 19, cex = 0.5,
     main = "RF Proximity MDS (Green=Active, Red=Churn)",
     xlab = "MDS Dim 1", ylab = "MDS Dim 2")
```

### Partial dependence plots

```r
# How does the predicted probability of Churn change as tenure varies?
partialPlot(
  rf_model,
  pred.data = train,
  x.var     = "tenure",
  which.class = "Churn",
  main  = "Partial Dependence: tenure → P(Churn)",
  xlab  = "tenure",
  ylab  = "Log-odds of Churn",
  col   = "#1D9E75", lwd = 2
)
```

### OOB vote matrix

```r
# Each row = one training observation
# Columns = fraction of trees voting for each class
oob_votes <- rf_model$votes
head(oob_votes)

# OOB predicted class for each training row
oob_pred <- rf_model$predicted
table(oob_pred, train$Churn)
```

---

## 9. Hyperparameter Tuning

### Manual grid search (full control)

```r
# ── Define the grid ──────────────────────────────────────────────────
param_grid <- expand.grid(
  ntree    = c(200, 500),
  mtry     = c(3, 5, 7),
  nodesize = c(1, 5, 10),
  stringsAsFactors = FALSE
)

# ── Evaluate each combination using OOB error ────────────────────────
set.seed(42)
results <- apply(param_grid, 1, function(p) {
  m <- randomForest(
    Churn ~ .,
    data     = train,
    ntree    = as.integer(p["ntree"]),
    mtry     = as.integer(p["mtry"]),
    nodesize = as.integer(p["nodesize"])
  )
  # OOB error rate (overall)
  oob_err <- m$err.rate[m$ntree, "OOB"]
  cat("ntree:", p["ntree"], "| mtry:", p["mtry"],
      "| nodesize:", p["nodesize"], "| OOB:", round(oob_err, 4), "\n")
  oob_err
})

param_grid$oob_error <- results
best_params <- param_grid[which.min(results), ]
print(best_params)
```

### Tuning via `caret` (cross-validation)

```r
# caret's tuneRF only tunes mtry — use trainControl for full CV
set.seed(42)

ctrl <- trainControl(
  method          = "cv",
  number          = 5,
  classProbs      = TRUE,
  summaryFunction = twoClassSummary,   # ROC, Sens, Spec
  savePredictions = "final",
  verboseIter     = FALSE
)

# Grid (caret's randomForest wrapper only exposes mtry directly)
tune_grid <- expand.grid(mtry = c(2, 4, 6, 8, 10))

rf_caret <- train(
  Churn ~ .,
  data      = train,
  method    = "rf",
  metric    = "ROC",             # optimise AUC-ROC
  trControl = ctrl,
  tuneGrid  = tune_grid,
  ntree     = 300,
  nodesize  = 5
)

print(rf_caret)
plot(rf_caret, main = "mtry vs AUC-ROC (5-fold CV)")

# Best mtry
rf_caret$bestTune
```

### `tuneRF` — automated mtry search

```r
# Built-in helper that searches around the default mtry
set.seed(42)

best_mtry <- tuneRF(
  x           = train[, names(train) != "Churn"],
  y           = train$Churn,
  ntreeTry    = 300,
  stepFactor  = 1.5,    # multiply/divide mtry by this each step
  improve     = 0.01,   # minimum OOB improvement to continue
  trace       = TRUE,
  plot        = TRUE,
  doBest      = FALSE   # TRUE = returns the best model directly
)

cat("Best mtry:", best_mtry[which.min(best_mtry[,2]), 1], "\n")
```

### Full ranger-based grid search (parallel, fast)

```r
library(ranger)

set.seed(42)
ranger_grid <- expand.grid(
  num.trees     = c(300, 500),
  mtry          = c(3, 5, 7),
  min.node.size = c(1, 5, 10),
  stringsAsFactors = FALSE
)

ranger_results <- apply(ranger_grid, 1, function(p) {
  m <- ranger(
    Churn ~ .,
    data          = train,
    num.trees     = as.integer(p["num.trees"]),
    mtry          = as.integer(p["mtry"]),
    min.node.size = as.integer(p["min.node.size"]),
    probability   = TRUE,
    num.threads   = parallel::detectCores() - 1,
    seed          = 42
  )
  m$prediction.error   # OOB Brier score
})

ranger_grid$oob_brier <- ranger_results
best_ranger <- ranger_grid[which.min(ranger_results), ]
print(best_ranger)
```

| Parameter | Typical range | Tuning approach |
|---|---|---|
| `ntree` | 100 – 1000 | Set high (500+) first, OOB stabilises it |
| `mtry` | 2 – p (num features) | `tuneRF` or manual grid |
| `nodesize` | 1 – 20 | Manual grid; 5–10 for large datasets |
| `maxnodes` | 10 – 100 | Use if trees are too deep |
| `sampsize` | 50% – 100% | Use with `replace = FALSE` |

---

## 10. Save & Load the Model

### Save with `saveRDS` (recommended)

```r
# ── Save ─────────────────────────────────────────────────────────────
saveRDS(rf_model, file = "models/rf_churn_v1.rds")

# ── Load in a new session ────────────────────────────────────────────
rf_loaded <- readRDS("models/rf_churn_v1.rds")

# Verify
predict(rf_loaded, newdata = head(test), type = "class")
```

### Save with metadata

```r
# Bundle model + metadata together
model_bundle <- list(
  model      = rf_model,
  version    = "v1.2",
  trained_on = Sys.Date(),
  features   = setdiff(names(train), "Churn"),
  ntree      = rf_model$ntree,
  mtry       = rf_model$mtry,
  nodesize   = 5,
  auc_roc    = as.numeric(auc(roc_obj)),
  train_rows = nrow(train),
  threshold  = best_threshold     # optimal classification threshold
)

saveRDS(model_bundle, "models/rf_churn_v1_bundle.rds")

# Reload
bundle      <- readRDS("models/rf_churn_v1_bundle.rds")
rf_loaded   <- bundle$model
threshold   <- bundle$threshold
cat("Model version:", bundle$version, "\n")
cat("AUC-ROC:", bundle$auc_roc, "\n")
```

### Export for portability

```r
# ── PMML export (score outside R) ───────────────────────────────────
install.packages("pmml")
library(pmml)

pmml_model <- pmml(rf_model)
saveXML(pmml_model, file = "models/rf_churn.pmml")

# ── Export as plain R object (base R, no package needed to load) ─────
# The RDS file IS the R object — no special package needed to read it:
# rf <- readRDS("rf_churn_v1.rds") works in any R session with randomForest loaded
```

---

## 11. Predict on New Data & Join Back

```r
# ── 1. Load new data ─────────────────────────────────────────────────
new_data <- read.csv("new_customers.csv", stringsAsFactors = TRUE)

# ── 2. Apply SAME feature engineering as training ────────────────────
new_data$tenure_sq   <- new_data$tenure^2
new_data$avg_charges <- new_data$TotalCharges / pmax(new_data$tenure, 1)
new_data$high_value  <- ifelse(new_data$MonthlyCharges > 70, 1L, 0L)

# Impute missing values using training medians
train_medians <- sapply(train[, num_cols], median, na.rm = TRUE)
for (col in names(train_medians)) {
  if (col %in% names(new_data)) {
    new_data[[col]][is.na(new_data[[col]])] <- train_medians[[col]]
  }
}

# Ensure factor levels match training data
for (col in cat_cols) {
  if (col %in% names(new_data)) {
    new_data[[col]] <- factor(new_data[[col]],
                              levels = levels(train[[col]]))
  }
}

# ── 3. Score ─────────────────────────────────────────────────────────
pred_class <- predict(rf_loaded, newdata = new_data, type = "class")
pred_prob  <- predict(rf_loaded, newdata = new_data, type = "prob")

# ── 4. Apply optimal threshold ───────────────────────────────────────
threshold   <- bundle$threshold     # from saved bundle
pred_tuned  <- factor(
  ifelse(pred_prob[, "Churn"] >= threshold, "Churn", "Active"),
  levels = c("Active", "Churn")
)

# ── 5. Build results data frame ──────────────────────────────────────
scored <- data.frame(
  CustomerID = new_data$CustomerID,     # keep original ID
  prob_churn = round(pred_prob[, "Churn"], 4),
  Prediction = as.character(pred_tuned)  # "Churn" or "Active"
)

# ── 6. Join back to original dataset ─────────────────────────────────
final_df <- merge(
  new_data,            # original dataset (all columns)
  scored,              # add: prob_churn, Prediction
  by   = "CustomerID",
  all.x = TRUE         # left join — keep all original rows
)

# ── 7. Preview ───────────────────────────────────────────────────────
print(head(final_df[, c("CustomerID", "tenure", "MonthlyCharges",
                         "prob_churn", "Prediction")], 10))
# CustomerID  tenure  MonthlyCharges  prob_churn  Prediction
# C0001          3         82.45        0.8742       Churn
# C0002         42         55.30        0.1035       Active
# C0003          8         91.10        0.7124       Churn
# C0004         60         34.20        0.0341       Active
# C0005         15         67.90        0.4609       Active

# ── 8. Rank customers by churn risk ──────────────────────────────────
final_df <- final_df[order(-final_df$prob_churn), ]

# ── 9. Write output ──────────────────────────────────────────────────
write.csv(final_df, "output/churn_predictions.csv", row.names = FALSE)

cat("Total scored:", nrow(final_df), "\n")
cat("Predicted Churn:", sum(final_df$Prediction == "Churn"), "\n")
cat("Predicted Active:", sum(final_df$Prediction == "Active"), "\n")
```

---

## 12. Quick-Reference Cheatsheet

### Core function arguments

| Argument | `randomForest` | `ranger` | Meaning |
|---|---|---|---|
| Trees | `ntree` | `num.trees` | Number of trees |
| Features/split | `mtry` | `mtry` | Features sampled per split |
| Min leaf size | `nodesize` | `min.node.size` | Smallest terminal node |
| Max leaf nodes | `maxnodes` | — | Hard cap on leaves per tree |
| Row sampling | `sampsize` | `sample.fraction` | Rows used per tree |
| With replacement | `replace` | `replace` | Bagging vs subsampling |
| Class weights | `classwt` | `class.weights` | Imbalance handling |
| Importances | `importance = TRUE` | `importance = "impurity"` | Enable feature importance |
| Probabilities | `type = "prob"` | `probability = TRUE` | Return class probabilities |

### Evaluation metrics

| Metric | Function | Notes |
|---|---|---|
| AUC-ROC | `pROC::auc()` | Overall discrimination ability |
| AUC-PR | `PRROC::pr.curve()` | Best for imbalanced data |
| Confusion matrix | `caret::confusionMatrix()` | F1, Sensitivity, Specificity |
| OOB error | `rf_model$err.rate` | Free estimate — no val set needed |

### When to use `ranger` vs `randomForest`

| Scenario | Use |
|---|---|
| Dataset > 50k rows | `ranger` (10–100× faster) |
| Need parallel training | `ranger` (`num.threads`) |
| Need proximity matrix | `randomForest` (`proximity = TRUE`) |
| Need partial dependence | `randomForest` (`partialPlot`) |
| Production/PMML export | `randomForest` + `pmml` package |
| Default / learning | `randomForest` (more examples online) |

### File structure recommendation

```
project/
├── data/
│   ├── churn.csv
│   └── new_customers.csv
├── models/
│   ├── rf_churn_v1.rds          # model object
│   └── rf_churn_v1_bundle.rds   # model + metadata
├── output/
│   └── churn_predictions.csv    # scored new data
└── rf_churn_model.R             # this script
```

---

*Base R Random Forest · randomForest / ranger · Churn Prediction Reference*
