

========== Part - 1 =========

Now I'll build a comprehensive, beautifully formatted interactive guide for the Random Forest + sparklyr workflow in R.Here's a complete end-to-end reference — click through each tab for the full code. Here's a summary of what's covered:

**Tab 1 — Setup**: Connect to Spark (local or cluster), install packages, load data, and check class balance.

**Tab 2 — Build**: Feature engineering, stratified train/test split, `ml_random_forest_classifier()` with all key arguments, evaluation with AUC-ROC/F1, and feature importance.

**Tab 3 — Imbalance**: Four strategies — `weight_col` (native, try first), oversampling via `sdf_sample`, undersampling, and post-model threshold tuning. Includes guidance on which metric to use (avoid accuracy — use AUC-PR or F1 for imbalanced data).

**Tab 4 — Regularisation**: Spark RF doesn't have L1/L2 penalties — regularisation happens through structural constraints: `max_depth`, `min_instances_per_node`, `min_info_gain`, `subsampling_rate`, and `feature_subset_strategy`. Also shows `recipes`-based preprocessing.

**Tab 5 — Tree Details**: How to pull individual trees using `invoke("toDebugString")`, inspect node counts and depths per tree, decode `rawPrediction` (the raw vote counts from each tree), and extract probability vectors.

**Tab 6 — Tune**: Full cross-validation (`ml_cross_validator`) and faster train-validation split (`ml_train_validation_split`) with parameter grids. Shows how to extract the best model and confirm its parameters.

**Tab 7 — Save**: `ml_save` / `ml_load` for Spark-native format, MLeap bundle for portable JVM scoring without Spark, and metadata versioning.

**Tab 8 — Predict**: Score new data, extract `prob_churn` + binary `"Churn"/"Active"` label, join back to the original dataset on `CustomerID`, and write output to CSV, Parquet, or Hive.


=========== Part - 2 ===========

Now I'll build a comprehensive, beautifully formatted interactive guide for the Random Forest + sparklyr workflow in R.Here's a complete end-to-end reference — click through each tab for the full code. Here's a summary of what's covered:

**Tab 1 — Setup**: Connect to Spark (local or cluster), install packages, load data, and check class balance.

**Tab 2 — Build**: Feature engineering, stratified train/test split, `ml_random_forest_classifier()` with all key arguments, evaluation with AUC-ROC/F1, and feature importance.

**Tab 3 — Imbalance**: Four strategies — `weight_col` (native, try first), oversampling via `sdf_sample`, undersampling, and post-model threshold tuning. Includes guidance on which metric to use (avoid accuracy — use AUC-PR or F1 for imbalanced data).

**Tab 4 — Regularisation**: Spark RF doesn't have L1/L2 penalties — regularisation happens through structural constraints: `max_depth`, `min_instances_per_node`, `min_info_gain`, `subsampling_rate`, and `feature_subset_strategy`. Also shows `recipes`-based preprocessing.

**Tab 5 — Tree Details**: How to pull individual trees using `invoke("toDebugString")`, inspect node counts and depths per tree, decode `rawPrediction` (the raw vote counts from each tree), and extract probability vectors.

**Tab 6 — Tune**: Full cross-validation (`ml_cross_validator`) and faster train-validation split (`ml_train_validation_split`) with parameter grids. Shows how to extract the best model and confirm its parameters.

**Tab 7 — Save**: `ml_save` / `ml_load` for Spark-native format, MLeap bundle for portable JVM scoring without Spark, and metadata versioning.

**Tab 8 — Predict**: Score new data, extract `prob_churn` + binary `"Churn"/"Active"` label, join back to the original dataset on `CustomerID`, and write output to CSV, Parquet, or Hive.

=========== Part - 3 ===========

Imbalance 

Now I'll build a comprehensive, beautifully formatted interactive guide for the Random Forest + sparklyr workflow in R.Here's a complete end-to-end reference — click through each tab for the full code. Here's a summary of what's covered:

**Tab 1 — Setup**: Connect to Spark (local or cluster), install packages, load data, and check class balance.

**Tab 2 — Build**: Feature engineering, stratified train/test split, `ml_random_forest_classifier()` with all key arguments, evaluation with AUC-ROC/F1, and feature importance.

**Tab 3 — Imbalance**: Four strategies — `weight_col` (native, try first), oversampling via `sdf_sample`, undersampling, and post-model threshold tuning. Includes guidance on which metric to use (avoid accuracy — use AUC-PR or F1 for imbalanced data).

**Tab 4 — Regularisation**: Spark RF doesn't have L1/L2 penalties — regularisation happens through structural constraints: `max_depth`, `min_instances_per_node`, `min_info_gain`, `subsampling_rate`, and `feature_subset_strategy`. Also shows `recipes`-based preprocessing.

**Tab 5 — Tree Details**: How to pull individual trees using `invoke("toDebugString")`, inspect node counts and depths per tree, decode `rawPrediction` (the raw vote counts from each tree), and extract probability vectors.

**Tab 6 — Tune**: Full cross-validation (`ml_cross_validator`) and faster train-validation split (`ml_train_validation_split`) with parameter grids. Shows how to extract the best model and confirm its parameters.

**Tab 7 — Save**: `ml_save` / `ml_load` for Spark-native format, MLeap bundle for portable JVM scoring without Spark, and metadata versioning.

**Tab 8 — Predict**: Score new data, extract `prob_churn` + binary `"Churn"/"Active"` label, join back to the original dataset on `CustomerID`, and write output to CSV, Parquet, or Hive.


=========== Part - 3 ===========

Regularize 

Now I'll build a comprehensive, beautifully formatted interactive guide for the Random Forest + sparklyr workflow in R.Here's a complete end-to-end reference — click through each tab for the full code. Here's a summary of what's covered:

**Tab 1 — Setup**: Connect to Spark (local or cluster), install packages, load data, and check class balance.

**Tab 2 — Build**: Feature engineering, stratified train/test split, `ml_random_forest_classifier()` with all key arguments, evaluation with AUC-ROC/F1, and feature importance.

**Tab 3 — Imbalance**: Four strategies — `weight_col` (native, try first), oversampling via `sdf_sample`, undersampling, and post-model threshold tuning. Includes guidance on which metric to use (avoid accuracy — use AUC-PR or F1 for imbalanced data).

**Tab 4 — Regularisation**: Spark RF doesn't have L1/L2 penalties — regularisation happens through structural constraints: `max_depth`, `min_instances_per_node`, `min_info_gain`, `subsampling_rate`, and `feature_subset_strategy`. Also shows `recipes`-based preprocessing.

**Tab 5 — Tree Details**: How to pull individual trees using `invoke("toDebugString")`, inspect node counts and depths per tree, decode `rawPrediction` (the raw vote counts from each tree), and extract probability vectors.

**Tab 6 — Tune**: Full cross-validation (`ml_cross_validator`) and faster train-validation split (`ml_train_validation_split`) with parameter grids. Shows how to extract the best model and confirm its parameters.

**Tab 7 — Save**: `ml_save` / `ml_load` for Spark-native format, MLeap bundle for portable JVM scoring without Spark, and metadata versioning.

**Tab 8 — Predict**: Score new data, extract `prob_churn` + binary `"Churn"/"Active"` label, join back to the original dataset on `CustomerID`, and write output to CSV, Parquet, or Hive.


=========== Part - 4 ===========

Tree Details

Now I'll build a comprehensive, beautifully formatted interactive guide for the Random Forest + sparklyr workflow in R.Here's a complete end-to-end reference — click through each tab for the full code. Here's a summary of what's covered:

**Tab 1 — Setup**: Connect to Spark (local or cluster), install packages, load data, and check class balance.

**Tab 2 — Build**: Feature engineering, stratified train/test split, `ml_random_forest_classifier()` with all key arguments, evaluation with AUC-ROC/F1, and feature importance.

**Tab 3 — Imbalance**: Four strategies — `weight_col` (native, try first), oversampling via `sdf_sample`, undersampling, and post-model threshold tuning. Includes guidance on which metric to use (avoid accuracy — use AUC-PR or F1 for imbalanced data).

**Tab 4 — Regularisation**: Spark RF doesn't have L1/L2 penalties — regularisation happens through structural constraints: `max_depth`, `min_instances_per_node`, `min_info_gain`, `subsampling_rate`, and `feature_subset_strategy`. Also shows `recipes`-based preprocessing.

**Tab 5 — Tree Details**: How to pull individual trees using `invoke("toDebugString")`, inspect node counts and depths per tree, decode `rawPrediction` (the raw vote counts from each tree), and extract probability vectors.

**Tab 6 — Tune**: Full cross-validation (`ml_cross_validator`) and faster train-validation split (`ml_train_validation_split`) with parameter grids. Shows how to extract the best model and confirm its parameters.

**Tab 7 — Save**: `ml_save` / `ml_load` for Spark-native format, MLeap bundle for portable JVM scoring without Spark, and metadata versioning.

**Tab 8 — Predict**: Score new data, extract `prob_churn` + binary `"Churn"/"Active"` label, join back to the original dataset on `CustomerID`, and write output to CSV, Parquet, or Hive.



=========== Part - 5 ===========

Tuning


Now I'll build a comprehensive, beautifully formatted interactive guide for the Random Forest + sparklyr workflow in R.Here's a complete end-to-end reference — click through each tab for the full code. Here's a summary of what's covered:

**Tab 1 — Setup**: Connect to Spark (local or cluster), install packages, load data, and check class balance.

**Tab 2 — Build**: Feature engineering, stratified train/test split, `ml_random_forest_classifier()` with all key arguments, evaluation with AUC-ROC/F1, and feature importance.

**Tab 3 — Imbalance**: Four strategies — `weight_col` (native, try first), oversampling via `sdf_sample`, undersampling, and post-model threshold tuning. Includes guidance on which metric to use (avoid accuracy — use AUC-PR or F1 for imbalanced data).

**Tab 4 — Regularisation**: Spark RF doesn't have L1/L2 penalties — regularisation happens through structural constraints: `max_depth`, `min_instances_per_node`, `min_info_gain`, `subsampling_rate`, and `feature_subset_strategy`. Also shows `recipes`-based preprocessing.

**Tab 5 — Tree Details**: How to pull individual trees using `invoke("toDebugString")`, inspect node counts and depths per tree, decode `rawPrediction` (the raw vote counts from each tree), and extract probability vectors.

**Tab 6 — Tune**: Full cross-validation (`ml_cross_validator`) and faster train-validation split (`ml_train_validation_split`) with parameter grids. Shows how to extract the best model and confirm its parameters.

**Tab 7 — Save**: `ml_save` / `ml_load` for Spark-native format, MLeap bundle for portable JVM scoring without Spark, and metadata versioning.

**Tab 8 — Predict**: Score new data, extract `prob_churn` + binary `"Churn"/"Active"` label, join back to the original dataset on `CustomerID`, and write output to CSV, Parquet, or Hive.


=========== Part - 6 ===========

Saving


Now I'll build a comprehensive, beautifully formatted interactive guide for the Random Forest + sparklyr workflow in R.Here's a complete end-to-end reference — click through each tab for the full code. Here's a summary of what's covered:

**Tab 1 — Setup**: Connect to Spark (local or cluster), install packages, load data, and check class balance.

**Tab 2 — Build**: Feature engineering, stratified train/test split, `ml_random_forest_classifier()` with all key arguments, evaluation with AUC-ROC/F1, and feature importance.

**Tab 3 — Imbalance**: Four strategies — `weight_col` (native, try first), oversampling via `sdf_sample`, undersampling, and post-model threshold tuning. Includes guidance on which metric to use (avoid accuracy — use AUC-PR or F1 for imbalanced data).

**Tab 4 — Regularisation**: Spark RF doesn't have L1/L2 penalties — regularisation happens through structural constraints: `max_depth`, `min_instances_per_node`, `min_info_gain`, `subsampling_rate`, and `feature_subset_strategy`. Also shows `recipes`-based preprocessing.

**Tab 5 — Tree Details**: How to pull individual trees using `invoke("toDebugString")`, inspect node counts and depths per tree, decode `rawPrediction` (the raw vote counts from each tree), and extract probability vectors.

**Tab 6 — Tune**: Full cross-validation (`ml_cross_validator`) and faster train-validation split (`ml_train_validation_split`) with parameter grids. Shows how to extract the best model and confirm its parameters.

**Tab 7 — Save**: `ml_save` / `ml_load` for Spark-native format, MLeap bundle for portable JVM scoring without Spark, and metadata versioning.

**Tab 8 — Predict**: Score new data, extract `prob_churn` + binary `"Churn"/"Active"` label, join back to the original dataset on `CustomerID`, and write output to CSV, Parquet, or Hive.


=========== Part - 6 ===========

Prediction



Now I'll build a comprehensive, beautifully formatted interactive guide for the Random Forest + sparklyr workflow in R.Here's a complete end-to-end reference — click through each tab for the full code. Here's a summary of what's covered:

**Tab 1 — Setup**: Connect to Spark (local or cluster), install packages, load data, and check class balance.

**Tab 2 — Build**: Feature engineering, stratified train/test split, `ml_random_forest_classifier()` with all key arguments, evaluation with AUC-ROC/F1, and feature importance.

**Tab 3 — Imbalance**: Four strategies — `weight_col` (native, try first), oversampling via `sdf_sample`, undersampling, and post-model threshold tuning. Includes guidance on which metric to use (avoid accuracy — use AUC-PR or F1 for imbalanced data).

**Tab 4 — Regularisation**: Spark RF doesn't have L1/L2 penalties — regularisation happens through structural constraints: `max_depth`, `min_instances_per_node`, `min_info_gain`, `subsampling_rate`, and `feature_subset_strategy`. Also shows `recipes`-based preprocessing.

**Tab 5 — Tree Details**: How to pull individual trees using `invoke("toDebugString")`, inspect node counts and depths per tree, decode `rawPrediction` (the raw vote counts from each tree), and extract probability vectors.

**Tab 6 — Tune**: Full cross-validation (`ml_cross_validator`) and faster train-validation split (`ml_train_validation_split`) with parameter grids. Shows how to extract the best model and confirm its parameters.

**Tab 7 — Save**: `ml_save` / `ml_load` for Spark-native format, MLeap bundle for portable JVM scoring without Spark, and metadata versioning.

**Tab 8 — Predict**: Score new data, extract `prob_churn` + binary `"Churn"/"Active"` label, join back to the original dataset on `CustomerID`, and write output to CSV, Parquet, or Hive.






