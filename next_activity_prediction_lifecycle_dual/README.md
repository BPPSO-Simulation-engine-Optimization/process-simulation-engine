# Dual Next-Event Prediction (Activity + Lifecycle)

This module trains a **joint model** that predicts:
- the next activity (`concept:name`)
- the next lifecycle transition (`lifecycle:transition`)

It supports two data modes:
1. `start_complete`: train on only `start` and `complete` lifecycle events
2. `full_lifecycle`: train on every available lifecycle transition

For each mode, it automatically compares two methodologies:
- `baseline`: normal training
- `balanced`: class-balanced sample weighting for both prediction heads

The output includes a single ranking file so you can select the most balanced model.

---

## Folder Layout

`next_activity_prediction_lifecycle_dual/models` is created after training:

- `start_complete/baseline/`
- `start_complete/balanced/`
- `full_lifecycle/baseline/`
- `full_lifecycle/balanced/`
- `comparison_summary.json`

Each model folder contains:
- `model.keras`
- `checkpoints/best_model.keras`
- `metadata.json`
- `history.json`
- `metrics.json`

---

## What "Balanced Performance" Means Here

Each trained model is scored with:
- activity accuracy
- activity macro-F1
- lifecycle accuracy
- lifecycle macro-F1
- joint accuracy (activity+lifecycle correct together)

Final ranking uses:

`balanced_score = mean(activity_macro_f1, lifecycle_macro_f1, joint_accuracy)`

This avoids favoring only frequent labels and pushes the model toward stable behavior across both tasks.

---

## Notebook Usage (Recommended)

Open and run:

`next_activity_prediction_lifecycle_dual/dual_lifecycle_experiment.ipynb`

The notebook handles:
- setup/config
- training all 4 combinations
- loading and ranking results by balanced performance

---

## Script Usage (Optional)

### 1) Train all variants and compare methodologies

```bash
python -m next_activity_prediction_lifecycle_dual.trainer --log-path "Dataset/your_log.xes"
```

or with CSV:

```bash
python -m next_activity_prediction_lifecycle_dual.trainer --log-path "Dataset/your_log.csv"
```

### 2) Optional training arguments

```bash
python -m next_activity_prediction_lifecycle_dual.trainer \
  --log-path "Dataset/your_log.xes" \
  --epochs 50 \
  --batch-size 64 \
  --sequence-length 60 \
  --model-root "next_activity_prediction_lifecycle_dual/models"
```

---

## Picking the Best Model

After training, open:

`next_activity_prediction_lifecycle_dual/models/comparison_summary.json`

Use:
- `best_model` for the top-ranked configuration
- `all_results` to inspect trade-offs between `baseline` and `balanced` in both lifecycle modes

---

## Notes

- If your log has no `lifecycle:transition` column, lifecycle is set to `unknown`.
- The module appends explicit end tokens for both activity and lifecycle to model case completion.
- TensorFlow/Keras, pandas, numpy, scikit-learn, and pm4py are expected from project requirements.
