# =============================================================================
#  OTT PLATFORM — PREDICTIVE MODELS SUITE
#  Packages: dplyr, sparklyr (primary) + base R / stats for modelling
#  Models covered:
#    1. Churn Prediction          (Logistic Regression + Random Forest)
#    2. Customer Lifetime Value   (Linear Regression + Gradient Boosted Trees)
#    3. Content Recommendation    (ALS Collaborative Filtering via Spark)
#    4. Conversion Propensity     (Logistic Regression)
#    5. Upsell / Plan Upgrade     (Logistic Regression)
#    6. Fraud Detection           (Isolation Forest proxy via GLM)
#    7. Content Demand Forecast   (Time-series ARIMA via base R)
# =============================================================================

# ── 0. DEPENDENCIES ──────────────────────────────────────────────────────────
library(sparklyr)
library(dplyr)
library(ggplot2)

# Optional but recommended
# install.packages(c("sparklyr","dplyr","ggplot2"))
# sparklyr::spark_install(version = "3.3")   # run once


# ── 1. SPARK SESSION ─────────────────────────────────────────────────────────
sc <- spark_connect(
  master = "local",
  config = spark_config()   # add spark.executor.memory etc. for cluster
)
cat("Spark connected — version:", spark_version(sc), "\n")


# =============================================================================
# SECTION A — SAMPLE DATA GENERATION
# All data is created locally then copied to Spark via copy_to()
# =============================================================================

set.seed(42)
N <- 5000   # number of subscribers

# ── A1. Subscriber master ─────────────────────────────────────────────────────
subscribers_local <- data.frame(
  subscriber_id    = paste0("SUB", sprintf("%05d", 1:N)),
  age              = sample(18:65, N, replace = TRUE),
  gender           = sample(c("M","F","Other"), N, replace = TRUE, prob = c(.48,.48,.04)),
  country          = sample(c("QA","AE","SA","KW","BH"), N, replace = TRUE,
                            prob = c(.35,.25,.20,.10,.10)),
  plan             = sample(c("trial","basic","standard","premium"), N, replace = TRUE,
                            prob = c(.15,.25,.40,.20)),
  tenure_months    = sample(1:60, N, replace = TRUE),
  monthly_fee_usd  = round(runif(N, 5, 25), 2),
  payment_method   = sample(c("card","wallet","bank"), N, replace = TRUE),
  stringsAsFactors = FALSE
)

# ── A2. Viewing behaviour ─────────────────────────────────────────────────────
viewing_local <- data.frame(
  subscriber_id       = subscribers_local$subscriber_id,
  avg_daily_hours     = pmax(0, rnorm(N, 1.8, 1.2)),
  days_active_last30  = sample(0:30, N, replace = TRUE),
  genres_watched      = sample(1:10, N, replace = TRUE),
  completion_rate     = round(runif(N, 0.3, 1.0), 2),
  content_starts      = sample(5:200, N, replace = TRUE),
  device_type         = sample(c("smart_tv","mobile","tablet","web"), N, replace = TRUE),
  stringsAsFactors    = FALSE
)

# ── A3. Support & account events ─────────────────────────────────────────────
support_local <- data.frame(
  subscriber_id      = subscribers_local$subscriber_id,
  support_tickets    = rpois(N, 0.8),
  payment_failures   = rpois(N, 0.3),
  login_failures     = rpois(N, 0.4),
  plan_changes       = sample(0:3, N, replace = TRUE),
  stringsAsFactors   = FALSE
)

# ── A4. Churn label (target) ──────────────────────────────────────────────────
# Churn probability is a function of real signals
churn_logit <- with(
  list(
    v = viewing_local,
    s = subscribers_local,
    sp = support_local
  ),
  -2.5
  + 1.2 * (v$days_active_last30 < 10)
  - 0.05 * v$avg_daily_hours
  + 0.8 * (s$plan == "trial")
  - 0.5 * (s$plan == "premium")
  + 0.3 * sp$support_tickets
  + 0.6 * sp$payment_failures
  - 0.02 * s$tenure_months
  + rnorm(N, 0, 0.5)
)
churn_prob <- 1 / (1 + exp(-churn_logit))
churn_label <- as.integer(rbinom(N, 1, churn_prob))

churn_labels_local <- data.frame(
  subscriber_id = subscribers_local$subscriber_id,
  churned       = churn_label,
  stringsAsFactors = FALSE
)

# ── A5. Content ratings (for recommendation) ─────────────────────────────────
n_content  <- 200
n_ratings  <- 15000
ratings_local <- data.frame(
  subscriber_id = sample(subscribers_local$subscriber_id, n_ratings, replace = TRUE),
  content_id    = sample(1:n_content, n_ratings, replace = TRUE),
  rating        = round(runif(n_ratings, 1, 5), 1),
  stringsAsFactors = FALSE
) |> distinct(subscriber_id, content_id, .keep_all = TRUE)


# ── A6. Push to Spark ─────────────────────────────────────────────────────────
subs_sdf     <- copy_to(sc, subscribers_local,    "subscribers",    overwrite = TRUE)
viewing_sdf  <- copy_to(sc, viewing_local,         "viewing",        overwrite = TRUE)
support_sdf  <- copy_to(sc, support_local,         "support",        overwrite = TRUE)
churn_sdf    <- copy_to(sc, churn_labels_local,    "churn_labels",   overwrite = TRUE)
ratings_sdf  <- copy_to(sc, ratings_local,         "ratings",        overwrite = TRUE)

cat("Sample data pushed to Spark.\n")


# =============================================================================
# MODEL 1 — CHURN PREDICTION
# =============================================================================
cat("\n===== MODEL 1: CHURN PREDICTION =====\n")

# ── 1a. Feature table (all joins in Spark via dplyr) ─────────────────────────
churn_features_sdf <- subs_sdf |>
  left_join(viewing_sdf,  by = "subscriber_id") |>
  left_join(support_sdf,  by = "subscriber_id") |>
  left_join(churn_sdf,    by = "subscriber_id") |>
  mutate(
    is_trial    = as.integer(plan == "trial"),
    is_premium  = as.integer(plan == "premium"),
    low_activity = as.integer(days_active_last30 < 10)
  ) |>
  select(
    subscriber_id, churned,
    age, tenure_months, monthly_fee_usd,
    avg_daily_hours, days_active_last30, genres_watched,
    completion_rate, content_starts,
    support_tickets, payment_failures, plan_changes,
    is_trial, is_premium, low_activity
  )

# ── 1b. Train / test split (Spark) ───────────────────────────────────────────
splits <- churn_features_sdf |>
  sdf_random_split(training = 0.8, testing = 0.2, seed = 42)
churn_train <- splits$training
churn_test  <- splits$testing

# ── 1c. Logistic Regression (Spark MLlib) ────────────────────────────────────
churn_lr <- churn_train |>
  ml_logistic_regression(
    formula       = churned ~ age + tenure_months + monthly_fee_usd +
      avg_daily_hours + days_active_last30 + genres_watched +
      completion_rate + support_tickets + payment_failures +
      plan_changes + is_trial + is_premium + low_activity,
    max_iter      = 100,
    reg_param     = 0.01,
    elastic_net_param = 0.5
  )

cat("Logistic Regression coefficients:\n")
print(churn_lr$coefficients)

# ── 1d. Evaluate on test set ─────────────────────────────────────────────────
churn_lr_preds <- ml_predict(churn_lr, churn_test)

# Collect metrics using base R
preds_local <- churn_lr_preds |>
  select(churned, prediction, probability_1) |>
  collect()

# Confusion matrix (base R table)
conf_mat <- table(
  Actual    = preds_local$churned,
  Predicted = preds_local$prediction
)
cat("\nChurn LR — Confusion Matrix:\n")
print(conf_mat)

# Derived metrics (base R)
TP <- conf_mat["1","1"]; FP <- conf_mat["0","1"]
TN <- conf_mat["0","0"]; FN <- conf_mat["1","0"]
accuracy  <- (TP + TN) / sum(conf_mat)
precision <- TP / (TP + FP)
recall    <- TP / (TP + FN)
f1        <- 2 * precision * recall / (precision + recall)

cat(sprintf("\nAccuracy : %.3f\nPrecision: %.3f\nRecall   : %.3f\nF1 Score : %.3f\n",
            accuracy, precision, recall, f1))

# ── 1e. Random Forest (Spark MLlib) ──────────────────────────────────────────
churn_rf <- churn_train |>
  ml_random_forest_classifier(
    formula    = churned ~ age + tenure_months + monthly_fee_usd +
      avg_daily_hours + days_active_last30 + genres_watched +
      completion_rate + support_tickets + payment_failures +
      plan_changes + is_trial + is_premium + low_activity,
    num_trees  = 100,
    max_depth  = 8,
    seed       = 42
  )

churn_rf_preds <- ml_predict(churn_rf, churn_test) |>
  select(churned, prediction, probability_1) |>
  collect()

rf_conf <- table(Actual = churn_rf_preds$churned, Predicted = churn_rf_preds$prediction)
cat("\nChurn RF — Confusion Matrix:\n")
print(rf_conf)

# Feature importance (base R)
fi <- ml_feature_importances(churn_rf)
cat("\nTop 5 Features by Importance:\n")
print(head(fi[order(-fi$importance), ], 5))

# ── 1f. Score ALL subscribers → churn risk table ─────────────────────────────
churn_scores_sdf <- ml_predict(churn_rf, churn_features_sdf) |>
  select(subscriber_id, churn_probability = probability_1) |>
  mutate(
    risk_segment = case_when(
      churn_probability >= 0.70 ~ "High",
      churn_probability >= 0.40 ~ "Medium",
      TRUE                      ~ "Low"
    )
  )

cat("\nChurn risk distribution:\n")
churn_scores_sdf |>
  group_by(risk_segment) |>
  summarise(n = n(), avg_prob = mean(churn_probability, na.rm = TRUE)) |>
  collect() |>
  print()


# =============================================================================
# MODEL 2 — CUSTOMER LIFETIME VALUE (LTV)
# =============================================================================
cat("\n===== MODEL 2: CUSTOMER LIFETIME VALUE =====\n")

# ── 2a. Feature + target engineering ─────────────────────────────────────────
ltv_features_sdf <- subs_sdf |>
  left_join(viewing_sdf,  by = "subscriber_id") |>
  left_join(support_sdf,  by = "subscriber_id") |>
  mutate(
    # Simple LTV proxy: monthly fee × projected tenure × engagement multiplier
    engagement_score  = (avg_daily_hours / 3) * completion_rate,
    estimated_ltv_usd = monthly_fee_usd *
                        pmax(1, tenure_months + (12 * engagement_score)) *
                        pmax(0.5, 1 - 0.1 * support_tickets) +
                        rnorm(1, 0, 20),
    is_premium  = as.integer(plan == "premium"),
    is_trial    = as.integer(plan == "trial")
  ) |>
  select(
    subscriber_id, estimated_ltv_usd,
    age, tenure_months, monthly_fee_usd,
    avg_daily_hours, completion_rate, genres_watched,
    support_tickets, plan_changes, is_premium, is_trial,
    engagement_score
  )

# ── 2b. Split ────────────────────────────────────────────────────────────────
ltv_splits <- ltv_features_sdf |>
  sdf_random_split(training = 0.8, testing = 0.2, seed = 42)
ltv_train  <- ltv_splits$training
ltv_test   <- ltv_splits$testing

# ── 2c. Linear Regression (Spark MLlib) ──────────────────────────────────────
ltv_lr <- ltv_train |>
  ml_linear_regression(
    formula   = estimated_ltv_usd ~ age + tenure_months + monthly_fee_usd +
      avg_daily_hours + completion_rate + genres_watched +
      support_tickets + plan_changes + is_premium + is_trial +
      engagement_score,
    reg_param = 0.01,
    elastic_net_param = 0.3
  )

cat("LTV Linear Regression — Model Summary:\n")
print(summary(ltv_lr))

# ── 2d. Gradient Boosted Trees regression (Spark MLlib) ──────────────────────
ltv_gbt <- ltv_train |>
  ml_gbt_regressor(
    formula    = estimated_ltv_usd ~ age + tenure_months + monthly_fee_usd +
      avg_daily_hours + completion_rate + genres_watched +
      support_tickets + plan_changes + is_premium + is_trial +
      engagement_score,
    max_iter   = 50,
    max_depth  = 5,
    step_size  = 0.1,
    seed       = 42
  )

# ── 2e. Evaluate (RMSE, MAE via base R) ──────────────────────────────────────
ltv_preds <- ml_predict(ltv_gbt, ltv_test) |>
  select(estimated_ltv_usd, prediction) |>
  collect()

rmse <- sqrt(mean((ltv_preds$estimated_ltv_usd - ltv_preds$prediction)^2))
mae  <- mean(abs(ltv_preds$estimated_ltv_usd - ltv_preds$prediction))
r2   <- cor(ltv_preds$estimated_ltv_usd, ltv_preds$prediction)^2
cat(sprintf("\nLTV GBT — RMSE: %.2f  MAE: %.2f  R²: %.3f\n", rmse, mae, r2))

# ── 2f. Score all subscribers ─────────────────────────────────────────────────
ltv_scores_sdf <- ml_predict(ltv_gbt, ltv_features_sdf) |>
  select(subscriber_id, predicted_ltv_usd = prediction) |>
  mutate(
    ltv_tier = case_when(
      predicted_ltv_usd >= 500 ~ "Platinum",
      predicted_ltv_usd >= 250 ~ "Gold",
      predicted_ltv_usd >= 100 ~ "Silver",
      TRUE                     ~ "Bronze"
    )
  )

cat("\nLTV tier distribution:\n")
ltv_scores_sdf |>
  group_by(ltv_tier) |>
  summarise(n = n(), avg_ltv = mean(predicted_ltv_usd, na.rm = TRUE)) |>
  collect() |>
  print()


# =============================================================================
# MODEL 3 — CONTENT RECOMMENDATION (ALS Collaborative Filtering)
# =============================================================================
cat("\n===== MODEL 3: CONTENT RECOMMENDATION (ALS) =====\n")

# ── 3a. Encode subscriber_id to integer index for ALS ────────────────────────
ratings_indexed_sdf <- ratings_sdf |>
  mutate(
    user_id    = as.integer(substr(subscriber_id, 4, 8)),  # extract numeric part
    item_id    = as.integer(content_id),
    rating_dbl = as.double(rating)
  ) |>
  select(user_id, item_id, rating_dbl)

# ── 3b. ALS model (Spark MLlib) ──────────────────────────────────────────────
als_model <- ratings_indexed_sdf |>
  ml_als(
    rating_col        = "rating_dbl",
    user_col          = "user_id",
    item_col          = "item_id",
    rank              = 10,       # latent factors
    max_iter          = 10,
    reg_param         = 0.1,
    cold_start_strategy = "drop",
    seed              = 42
  )

cat("ALS model trained. Rank:", als_model$rank, "\n")

# ── 3c. Generate top-5 recommendations per user ──────────────────────────────
top5_recs <- ml_recommend(als_model, items_for_users = 5)

cat("\nSample recommendations (first 3 users):\n")
top5_recs |>
  head(3) |>
  collect() |>
  print()

# ── 3d. RMSE on training data (base R) ───────────────────────────────────────
als_preds <- ml_predict(als_model, ratings_indexed_sdf) |>
  filter(!is.nan(prediction)) |>
  select(rating_dbl, prediction) |>
  collect()

als_rmse <- sqrt(mean((als_preds$rating_dbl - als_preds$prediction)^2, na.rm = TRUE))
cat(sprintf("\nALS Training RMSE: %.4f\n", als_rmse))


# =============================================================================
# MODEL 4 — CONVERSION PROPENSITY (Trial → Paid)
# =============================================================================
cat("\n===== MODEL 4: CONVERSION PROPENSITY =====\n")

# ── 4a. Filter to trial users; label = converted ─────────────────────────────
# Simulate conversion: higher engagement → more likely to convert
trial_sdf <- subs_sdf |>
  filter(plan == "trial") |>
  left_join(viewing_sdf,  by = "subscriber_id") |>
  left_join(support_sdf,  by = "subscriber_id") |>
  mutate(
    converted = as.integer(
      (avg_daily_hours * 0.4 +
       completion_rate * 0.3 +
       (days_active_last30 / 30) * 0.3 +
       rnorm(1, 0, 0.2)) > 0.45
    )
  ) |>
  select(
    subscriber_id, converted,
    age, avg_daily_hours, days_active_last30,
    completion_rate, content_starts, genres_watched,
    support_tickets
  )

cat("Trial subscribers:", sdf_nrow(trial_sdf), "\n")

# ── 4b. Train / test ──────────────────────────────────────────────────────────
conv_splits <- trial_sdf |>
  sdf_random_split(training = 0.8, testing = 0.2, seed = 42)

conv_model <- conv_splits$training |>
  ml_logistic_regression(
    formula  = converted ~ age + avg_daily_hours + days_active_last30 +
      completion_rate + content_starts + genres_watched + support_tickets,
    max_iter = 100
  )

# ── 4c. Evaluate ─────────────────────────────────────────────────────────────
conv_preds <- ml_predict(conv_model, conv_splits$testing) |>
  select(converted, prediction, probability_1) |>
  collect()

conv_acc <- mean(conv_preds$converted == conv_preds$prediction)
cat(sprintf("Conversion model accuracy: %.3f\n", conv_acc))

# ── 4d. Score all trial users → conversion score ─────────────────────────────
conversion_scores_sdf <- ml_predict(conv_model, trial_sdf) |>
  select(subscriber_id, conversion_probability = probability_1) |>
  mutate(
    priority = case_when(
      conversion_probability >= 0.7 ~ "Hot lead — immediate offer",
      conversion_probability >= 0.4 ~ "Warm lead — nurture campaign",
      TRUE                          ~ "Cold lead — organic only"
    )
  )

cat("\nConversion priority distribution:\n")
conversion_scores_sdf |>
  group_by(priority) |>
  summarise(n = n()) |>
  collect() |>
  print()


# =============================================================================
# MODEL 5 — UPSELL PROPENSITY (Basic/Standard → Premium)
# =============================================================================
cat("\n===== MODEL 5: UPSELL PROPENSITY =====\n")

# ── 5a. Filter non-premium; label = upgraded ─────────────────────────────────
upsell_sdf <- subs_sdf |>
  filter(plan %in% c("basic", "standard")) |>
  left_join(viewing_sdf,  by = "subscriber_id") |>
  left_join(support_sdf,  by = "subscriber_id") |>
  mutate(
    upgraded = as.integer(
      (avg_daily_hours * 0.35 +
       genres_watched * 0.04 +
       completion_rate * 0.25 +
       tenure_months  * 0.005 +
       rnorm(1, 0, 0.2)) > 0.5
    ),
    is_standard = as.integer(plan == "standard")
  ) |>
  select(
    subscriber_id, upgraded, is_standard,
    age, tenure_months, monthly_fee_usd,
    avg_daily_hours, genres_watched, completion_rate,
    content_starts, plan_changes, support_tickets
  )

# ── 5b. Train model ──────────────────────────────────────────────────────────
upsell_splits <- upsell_sdf |>
  sdf_random_split(training = 0.8, testing = 0.2, seed = 42)

upsell_model <- upsell_splits$training |>
  ml_logistic_regression(
    formula  = upgraded ~ is_standard + age + tenure_months +
      monthly_fee_usd + avg_daily_hours + genres_watched +
      completion_rate + content_starts + plan_changes + support_tickets,
    max_iter = 100,
    reg_param = 0.01
  )

# ── 5c. Evaluate + score ─────────────────────────────────────────────────────
upsell_preds <- ml_predict(upsell_model, upsell_splits$testing) |>
  select(upgraded, prediction, probability_1) |>
  collect()

upsell_acc <- mean(upsell_preds$upgraded == upsell_preds$prediction)
cat(sprintf("Upsell model accuracy: %.3f\n", upsell_acc))

upsell_scores_sdf <- ml_predict(upsell_model, upsell_sdf) |>
  select(subscriber_id, upsell_probability = probability_1) |>
  arrange(desc(upsell_probability))

cat("\nTop 10 upsell candidates:\n")
upsell_scores_sdf |> head(10) |> collect() |> print()


# =============================================================================
# MODEL 6 — FRAUD DETECTION
# =============================================================================
cat("\n===== MODEL 6: FRAUD DETECTION =====\n")

# ── 6a. Engineer fraud signals ────────────────────────────────────────────────
# Simulate: accounts with many login failures, payment failures, multiple devices
# and very low content consumption may be shared/fraudulent accounts
fraud_features_sdf <- subs_sdf |>
  left_join(viewing_sdf,  by = "subscriber_id") |>
  left_join(support_sdf,  by = "subscriber_id") |>
  mutate(
    # anomaly score proxy features
    unusual_login  = as.integer(login_failures > 5),
    payment_issues = as.integer(payment_failures > 2),
    low_content    = as.integer(avg_daily_hours < 0.1 & content_starts < 10),
    multi_device   = as.integer(device_type == "web" & content_starts > 100),
    fraud_score_raw = (login_failures * 0.3 +
                       payment_failures * 0.4 +
                       as.integer(avg_daily_hours < 0.1) * 0.3 +
                       rnorm(1, 0, 0.5))
  )

# ── 6b. GLM anomaly score → binary flag (base R threshold) ───────────────────
# Collect a sample for base R isolation-forest proxy
fraud_local <- fraud_features_sdf |>
  select(subscriber_id, login_failures, payment_failures,
         avg_daily_hours, content_starts, support_tickets,
         fraud_score_raw) |>
  collect()

# Standardise (base R scale)
fraud_scaled <- scale(
  fraud_local[, c("login_failures","payment_failures",
                  "avg_daily_hours","content_starts","support_tickets")]
)

# Mahalanobis distance as outlier score (base R)
fraud_cov   <- cov(fraud_scaled)
fraud_center <- colMeans(fraud_scaled)
mahal_dist  <- mahalanobis(fraud_scaled, center = fraud_center, cov = fraud_cov)

fraud_local$outlier_score <- mahal_dist
fraud_local$is_suspicious <- as.integer(mahal_dist > quantile(mahal_dist, 0.95))

cat(sprintf("\nFraud detection:\n  Flagged suspicious: %d (%.1f%% of base)\n",
            sum(fraud_local$is_suspicious),
            100 * mean(fraud_local$is_suspicious)))

# Push fraud scores to Spark
fraud_scores_sdf <- copy_to(sc,
  fraud_local |> select(subscriber_id, outlier_score, is_suspicious),
  "fraud_scores", overwrite = TRUE
)


# =============================================================================
# MODEL 7 — CONTENT DEMAND FORECASTING (ARIMA, base R)
# =============================================================================
cat("\n===== MODEL 7: CONTENT DEMAND FORECASTING (ARIMA) =====\n")

# ── 7a. Simulate weekly new-subscription time series ─────────────────────────
set.seed(42)
n_weeks  <- 104   # 2 years of weekly data
trend    <- seq(500, 900, length.out = n_weeks)
seasonal <- 80 * sin(2 * pi * (1:n_weeks) / 52)   # annual cycle
noise    <- rnorm(n_weeks, 0, 40)
weekly_subs <- round(trend + seasonal + noise)

ts_subs <- ts(weekly_subs, frequency = 52, start = c(2022, 1))

cat("Weekly subscriptions — summary:\n")
print(summary(as.numeric(ts_subs)))

# ── 7b. Fit ARIMA (base R stats::arima via auto-search) ──────────────────────
# Simple grid search for ARIMA(p,d,q) — lightweight, no forecast package needed
best_aic <- Inf
best_order <- c(1, 1, 1)

for (p in 0:2) {
  for (d in 0:1) {
    for (q in 0:2) {
      tryCatch({
        fit <- arima(ts_subs, order = c(p, d, q),
                     seasonal = list(order = c(1, 0, 0), period = 52),
                     method = "ML")
        if (AIC(fit) < best_aic) {
          best_aic   <- AIC(fit)
          best_order <- c(p, d, q)
          best_fit   <- fit
        }
      }, error = function(e) NULL)
    }
  }
}

cat(sprintf("\nBest ARIMA order: (%d,%d,%d) — AIC: %.2f\n",
            best_order[1], best_order[2], best_order[3], best_aic))

# ── 7c. 12-week forecast (base R predict.Arima) ───────────────────────────────
forecast_steps <- 12
fc <- predict(best_fit, n.ahead = forecast_steps)

forecast_df <- data.frame(
  week          = seq_len(forecast_steps),
  forecast      = round(fc$pred),
  lower_95      = round(fc$pred - 1.96 * fc$se),
  upper_95      = round(fc$pred + 1.96 * fc$se)
)

cat("\n12-Week Subscription Demand Forecast:\n")
print(forecast_df)


# =============================================================================
# SECTION B — UNIFIED SCORING TABLE
# Join all model scores into one master subscriber view
# =============================================================================
cat("\n===== UNIFIED SUBSCRIBER SCORE TABLE =====\n")

master_scores_sdf <- subs_sdf |>
  select(subscriber_id, plan, tenure_months, monthly_fee_usd) |>
  left_join(
    churn_scores_sdf |> select(subscriber_id, churn_probability, risk_segment),
    by = "subscriber_id"
  ) |>
  left_join(
    ltv_scores_sdf |> select(subscriber_id, predicted_ltv_usd, ltv_tier),
    by = "subscriber_id"
  ) |>
  left_join(
    fraud_scores_sdf |> select(subscriber_id, outlier_score, is_suspicious),
    by = "subscriber_id"
  ) |>
  mutate(
    # Combined priority score for CRM targeting
    action_priority = case_when(
      is_suspicious == 1                               ~ "Investigate fraud",
      risk_segment == "High" & ltv_tier %in% c("Platinum","Gold") ~ "VIP save — immediate",
      risk_segment == "High"                           ~ "High-priority save",
      risk_segment == "Medium" & ltv_tier == "Platinum"~ "Proactive engagement",
      TRUE                                             ~ "Standard monitoring"
    )
  )

cat("\nAction priority distribution across all subscribers:\n")
master_scores_sdf |>
  group_by(action_priority) |>
  summarise(
    n            = n(),
    avg_ltv      = round(mean(predicted_ltv_usd, na.rm = TRUE), 2),
    avg_churn_pr = round(mean(churn_probability, na.rm = TRUE), 3)
  ) |>
  arrange(desc(n)) |>
  collect() |>
  print()


# =============================================================================
# SECTION C — VISUALISATIONS (ggplot2, local data)
# =============================================================================

# ── C1. Churn probability distribution by plan ────────────────────────────────
plot_data <- churn_scores_sdf |>
  left_join(subs_sdf |> select(subscriber_id, plan), by = "subscriber_id") |>
  collect()

p1 <- ggplot(plot_data, aes(x = churn_probability, fill = plan)) +
  geom_density(alpha = 0.5, color = NA) +
  facet_wrap(~plan, ncol = 2) +
  scale_fill_manual(values = c(trial = "#E24B4A", basic = "#F59E0B",
                                standard = "#378ADD", premium = "#1D9E75")) +
  labs(title = "Churn probability distribution by plan",
       x = "Churn probability", y = "Density") +
  theme_minimal(base_size = 13) +
  theme(legend.position = "none")
print(p1)

# ── C2. LTV vs churn risk scatter ─────────────────────────────────────────────
scatter_data <- master_scores_sdf |>
  select(predicted_ltv_usd, churn_probability, risk_segment) |>
  collect() |>
  sample_n(min(1000, nrow(collect(master_scores_sdf))))   # sample for plot speed

p2 <- ggplot(scatter_data,
             aes(x = predicted_ltv_usd, y = churn_probability, colour = risk_segment)) +
  geom_point(alpha = 0.4, size = 1.5) +
  scale_colour_manual(values = c(Low = "#1D9E75", Medium = "#F59E0B", High = "#E24B4A")) +
  labs(title = "LTV vs Churn Risk — subscriber segmentation",
       x = "Predicted LTV (USD)", y = "Churn probability",
       colour = "Risk segment") +
  theme_minimal(base_size = 13)
print(p2)

# ── C3. ARIMA forecast plot (base R graphics) ─────────────────────────────────
plot(ts_subs,
     main = "Weekly subscriptions + 12-week ARIMA forecast",
     ylab = "New subscriptions", xlab = "Week",
     col = "#378ADD", lwd = 1.5, xlim = c(2022, 2024.5))
lines(ts(forecast_df$forecast,
         start = end(ts_subs) + c(0, 1),
         frequency = 52),
      col = "#E24B4A", lwd = 2, lty = 2)
legend("topleft", legend = c("Actual","Forecast"),
       col = c("#378ADD","#E24B4A"), lty = c(1,2), lwd = 2)


# =============================================================================
# SECTION D — SAVE OUTPUTS
# =============================================================================

# ── D1. Write master scores back to Spark (could go to Parquet / Delta) ──────
cat("\nWriting master scores to local Parquet...\n")
master_scores_sdf |>
  spark_write_parquet(path = "/tmp/ott_master_scores", mode = "overwrite")
cat("Written: /tmp/ott_master_scores/\n")

# ── D2. Collect and write CSV locally ────────────────────────────────────────
master_local <- master_scores_sdf |> collect()
write.csv(master_local, "/tmp/ott_master_scores.csv", row.names = FALSE)
cat("CSV written: /tmp/ott_master_scores.csv\n")

# ── D3. Save ARIMA forecast ───────────────────────────────────────────────────
write.csv(forecast_df, "/tmp/ott_demand_forecast.csv", row.names = FALSE)
cat("Demand forecast CSV written: /tmp/ott_demand_forecast.csv\n")


# =============================================================================
# SECTION E — DISCONNECT SPARK
# =============================================================================
spark_disconnect(sc)
cat("\nSpark session closed. All models complete.\n")


# =============================================================================
# QUICK REFERENCE — FUNCTION SUMMARY
# =============================================================================
#
#  SPARK / DPLYR FUNCTIONS USED
#  ─────────────────────────────────────────────────────────────────────────────
#  spark_connect()          Open Spark session
#  copy_to()                Push R data.frame → Spark DataFrame
#  sdf_random_split()       Train/test split in Spark
#  sdf_nrow()               Row count of Spark DataFrame
#  collect()                Pull Spark DataFrame → R data.frame
#  left_join()              SQL-style join (runs in Spark)
#  mutate() / select()      Column transforms / selection (dplyr → Spark SQL)
#  filter() / arrange()     Row filter / sort
#  group_by() + summarise() Aggregation
#  case_when()              Multi-condition CASE WHEN
#  spark_write_parquet()    Write to Parquet from Spark
#
#  SPARK MLLIB (ml_*) FUNCTIONS USED
#  ─────────────────────────────────────────────────────────────────────────────
#  ml_logistic_regression()         Binary classification
#  ml_random_forest_classifier()    Ensemble classification
#  ml_gbt_regressor()               Gradient boosted trees regression
#  ml_linear_regression()           OLS / regularised regression
#  ml_als()                         Alternating Least Squares (recommendation)
#  ml_recommend()                   Top-N recommendations from ALS
#  ml_predict()                     Score new data from any model
#  ml_feature_importances()         Variable importance from tree models
#
#  BASE R / STATS FUNCTIONS USED
#  ─────────────────────────────────────────────────────────────────────────────
#  scale()                  Standardise matrix
#  cov()                    Covariance matrix
#  mahalanobis()            Outlier distance metric
#  arima()                  ARIMA time-series model
#  predict.Arima()          Multi-step forecast
#  AIC()                    Model selection criterion
#  table()                  Confusion matrix
#  cor()                    Pearson correlation (R²)
#
# =============================================================================
