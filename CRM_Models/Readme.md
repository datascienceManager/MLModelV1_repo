
**Data** — 5,000 synthetic subscribers are generated locally and pushed to Spark via `copy_to()`. Every feature is designed to mimic real OTT signals (viewing hours, completion rate, support tickets, payment failures, plan type, tenure).

**7 models built:**

| # | Model | Algorithm | Package |
|---|---|---|---|
| 1 | Churn prediction | Logistic Regression + Random Forest | `sparklyr` MLlib |
| 2 | Customer LTV | Linear Regression + Gradient Boosted Trees | `sparklyr` MLlib |
| 3 | Content recommendation | ALS Collaborative Filtering | `sparklyr` MLlib |
| 4 | Conversion propensity | Logistic Regression | `sparklyr` MLlib |
| 5 | Upsell propensity | Logistic Regression | `sparklyr` MLlib |
| 6 | Fraud detection | Mahalanobis distance (outlier scoring) | base R `stats` |
| 7 | Demand forecasting | ARIMA with seasonal component | base R `stats` |

**Outputs:**
- Confusion matrix + Accuracy / Precision / Recall / F1 for classifiers
- RMSE / MAE / R² for regression models
- A **unified master score table** joining all model outputs per subscriber, with a CRM `action_priority` label (VIP save, high-priority save, fraud investigate, etc.)
- Three `ggplot2` charts — churn distribution by plan, LTV vs churn scatter, and ARIMA forecast
- Results written to Parquet (`/tmp/ott_master_scores/`) and CSV

**To run:** install `sparklyr`, `dplyr`, `ggplot2`, call `spark_install()` once, then `source("OTT_Predictive_Models.R")`. The quick reference block at the bottom documents every Spark/dplyr/base R function used.
