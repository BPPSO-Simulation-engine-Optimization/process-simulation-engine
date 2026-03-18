# Processing Time Prediction

Predicts the time (in seconds) between consecutive events in a process. Used by the simulation engine to sample realistic durations.

---

## Integrated Models (used in simulation)

These are trained with `ProcessingTimeTrainer` and used via `ProcessingTimePredictionClass`. Set `method` when creating the predictor.

### 1. Distribution (`method="distribution"`)

**What it is:** Per-transition log-normal distributions. One distribution per *(prev_activity, prev_lifecycle, curr_activity, curr_lifecycle)*.

**Training:** `fit_distributions()` — computes processing times between consecutive events per case, groups by transition key, fits `scipy.stats.lognorm` (mu, sigma from log(times)). Needs at least `min_observations` (default 2) per transition.

**Prediction:** Sample from the distribution for that transition. Fallback: same activity pair without lifecycle, or global log-normal from fallback mean/std.

**Use when:** Simple baseline, no ML stack, few features, or you want pure transition-based timing.

---

### 2. ML — Random Forest (`method="ml"`)

**What it is:** Single Random Forest regressor predicting processing time (seconds) from context features.

**Features (from event log / context):**
- Categorical: `prev_activity`, `prev_lifecycle`, `curr_activity`, `curr_lifecycle`, `prev_resource`, `curr_resource`, `case:LoanGoal`, `case:ApplicationType`
- Numerical: `hour`, `weekday`, `month`, `day_of_year`, `event_position_in_case`, `case_duration_so_far`, plus booleans like `Accepted`, `Selected`

**Training:** `train_ml_model()` — extracts (X, y) from event log, label-encodes categories, MinMax-scales numerics, removes outliers (mean + 3*std), fits `sklearn.ensemble.RandomForestRegressor` (default n_estimators=500, max_depth=30, etc.).

**Prediction:** One vector per call via `_context_to_features` + `_prepare_single_vector` (or legacy `_prepare_features`), then `ml_model.predict`. Output is clamped to [0, 86400] seconds.

**Use when:** You want rich context (time, case, resources) and a single, interpretable model.

---

### 3. Probabilistic ML — LSTM (`method="probabilistic_ml"`)

**What it is:** Gaussian LSTM that predicts a mean and log-variance for the next processing time (in log space), so you get a full distribution per step.

**Inputs:**
- **Sequence:** Last `sequence_length` (default 10) events as one-hot: activity + lifecycle + resource per step, padded left.
- **Context:** hour, weekday, month, day_of_year, event position, case duration so far, case:LoanGoal, case:ApplicationType (normalized).

**Target:** Log(processing_time + 1), then normalized with train mean/std.

**Architecture (TensorFlow/Keras):**
- Sequence branch: `Input(sequence_length, feature_dim)` → LSTM(128, dropout=0.3, recurrent_dropout=0.2) → vector.
- Context branch: `Input(context_dim)` → Dense(64, relu) → Dropout(0.3) → vector.
- Concatenate → Dense(128, relu) → Dropout(0.4) → Dense(64, relu) → Dropout(0.3) → two heads: `mean` (Dense(1)), `log_variance` (Dense(1)).
- Loss: Gaussian negative log-likelihood (mean + log_var).

**Training:** `train_probabilistic_ml_model()` — `_extract_sequences()` builds sequences per case, optional cache; outliers removed; train/val split; log+normalize target; fit with Adam (lr=0.0005), EarlyStopping on val_loss. Saves encoders, y_mean, y_std, and Keras model.

**Prediction:** Build sequence + context from `event_history` and current event → model predicts (mean_norm, log_var) → invert normalization and exp → sample from Gaussian(mean, std) and return max(0, sample).

**Use when:** You want sequence-aware, probabilistic predictions and have enough data and TensorFlow available.

---

## XGBoost Approach (not integrated)

Lives under `XGBoost_approach (Not integrated)/`. Not wired into `ProcessingTimePredictionClass`; use only for experiments or separate pipelines.

### 4. ModelTrainer (single XGBoost)

**What it is:** One XGBoost regressor in a sklearn Pipeline: `ColumnTransformer` (OneHotEncoder for categoricals + passthrough for numericals) → `XGBRegressor` (e.g. n_estimators=300, max_depth=5, learning_rate=0.05). Target is typically log(processing time).

**Features:** Configurable; defaults include `event`, `lifecycle:transition`, `event_index`, `hour`, `weekday`.

**Usage:** `split_data_grouped(X, y, groups)` for case-aware split, then `train_model(X_train, y_train)`, then `predict(X)`.

---

### 5. ActivitySpecificModel

**What it is:** One model per activity (or per “next” activity). For each activity, can use:
- **Fixed 1s:** e.g. A_Cancelled, A_Submitted — no training.
- **W-activities:** special handling for schedule→complete/ate_abort time.
- **Quantile regression:** `QuantileModelTrainer` for that activity (see below).
- **Outlier separation:** train one model on “normal” and one on “outlier” samples, then route at predict time.
- **Standard:** single `ModelTrainer` (XGBoost) for that activity.

**Usage:** `prepare_activity_data(df)` then train/load per activity; `predict_for_activity(activity, X)` routes to the right model.

---

### 6. QuantileModelTrainer

**What it is:** Multiple quantile regressions (e.g. 0.25, 0.5, 0.75). Per quantile, either:
- **LightGBM** (if installed) with `objective='quantile', alpha=quantile`, or
- **sklearn GradientBoostingRegressor** with `loss='quantile', alpha=quantile`.

**Outputs:** `predict_quantiles(X)` → DataFrame of quantile columns; `predict_median(X)` or `predict_lowest_quantile(X)` for single summaries.

**Use when:** You need full conditional quantiles (e.g. for robust or risk-aware simulation), especially for multimodal or skewed processing times.

---

### 7. ClassificationRegressionModel

**What it is:** Two-stage: (1) Random Forest classifier into speed buckets (e.g. fast &lt;5h, medium 5–25h, slow 25–45h, very_slow &gt;45h), (2) one XGBoost regressor per bucket on log(target).

**Usage:** `train_classifier_regressor(X_train, y_train, y_train_original)` then `predict(X)` — classify then regress per class.

**Use when:** Processing times are clearly multimodal (fast vs slow paths) and a single regressor underperforms.

---

## Quick reference

| Model                     | Integrated | Type              | Output           |
|---------------------------|------------|-------------------|------------------|
| Distribution              | Yes        | Log-normal sample | Single value     |
| ML (Random Forest)        | Yes        | Regression        | Single value     |
| Probabilistic ML (LSTM)   | Yes        | Gaussian (mean+var) | Sampled value  |
| ModelTrainer (XGBoost)    | No         | Regression        | Single value     |
| ActivitySpecificModel     | No         | Per-activity mix  | Single value     |
| QuantileModelTrainer      | No         | Quantile regression | Quantiles / median |
| ClassificationRegression  | No         | Class + regress   | Single value     |

---

## Training and saving (integrated)

```python
from processing_time_prediction.ProcessingTimeTrainer import ProcessingTimeTrainer

trainer = ProcessingTimeTrainer(data_log_df, method="ml")  # or "distribution" or "probabilistic_ml"
trainer.train(cache_path="models/processing_time_model_lstm", force_recompute=False)
trainer.save_model("models/processing_time_model_ml")
```

## Benchmarking

`benchmark_models.py` compares the three integrated models on a case-based train/test split and reports MAE, RMSE, R², MAPE, Median AE, and MdAPE.

```bash
cd processing_time_prediction
python benchmark_models.py --log "Dataset/BPI Challenge 2017.xes"
python benchmark_models.py --log eventlog.xes --test_samples 2000 --output results.csv
python benchmark_models.py --log eventlog.xes --skip_train --model_dir models
```

- **--log**: Path to event log (.xes or .csv). Default: `Dataset/BPI Challenge 2017.xes`
- **--train_ratio**: Fraction of cases for training (default 0.8).
- **--test_samples**: Cap number of test samples (default: use all).
- **--model_dir**: Where to save/load models (default `models`).
- **--output**: Write results table to this CSV path.
- **--skip_train**: Only evaluate; load existing models from `model_dir`.

---

## Loading and predicting (integrated)

```python
from processing_time_prediction.ProcessingTimePredictionClass import ProcessingTimePredictionClass

predictor = ProcessingTimePredictionClass(method="ml", model_path="models/processing_time_model_ml")
seconds = predictor.predict(
    prev_activity="A_Complete",
    prev_lifecycle="complete",
    curr_activity="A_Accepted",
    curr_lifecycle="complete",
    context={"resource_1": "User 1", "event_position_in_case": 5, ...}
)
```

For probabilistic LSTM you can also use `get_probabilistic_distribution(...)` to get mean/std for the next step.
