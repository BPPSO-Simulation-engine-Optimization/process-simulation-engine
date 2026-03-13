import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.utils import get_token


def _load_best_model_dir(model_root: Path) -> Path:
    summary_path = model_root / "comparison_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Could not find {summary_path}. Run training first or pass --model-dir explicitly."
        )

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    best = summary.get("best_model", {})
    model_dir = best.get("model_dir")
    if not model_dir:
        raise ValueError("comparison_summary.json has no best_model.model_dir")
    return Path(model_dir)


def _build_model_card(repo_id: str, metrics: dict | None, source_model_dir: Path) -> str:
    metrics_lines = ""
    if metrics:
        metrics_lines = "\n".join(
            [
                f"- Activity accuracy: `{metrics.get('activity_accuracy', 'n/a')}`",
                f"- Activity macro-F1: `{metrics.get('activity_macro_f1', 'n/a')}`",
                f"- Lifecycle accuracy: `{metrics.get('lifecycle_accuracy', 'n/a')}`",
                f"- Lifecycle macro-F1: `{metrics.get('lifecycle_macro_f1', 'n/a')}`",
                f"- Joint accuracy: `{metrics.get('joint_accuracy', 'n/a')}`",
                f"- Balanced score: `{metrics.get('balanced_score', 'n/a')}`",
            ]
        )
    else:
        metrics_lines = "- Metrics not provided in this export."

    return f"""---
library_name: tf-keras
tags:
  - process-mining
  - next-activity-prediction
  - sequence-modeling
  - tensorflow
---

# {repo_id.split('/')[-1]}

Dual-head next-event model that predicts:
- next activity (`concept:name`)
- next lifecycle transition (`lifecycle:transition`)

This repository was exported from:
`{source_model_dir}`

## Included files

- `model.keras`
- `metadata.json`
- `metrics.json` (if available)
- `history.json` (if available)

## Metrics

{metrics_lines}

## Usage (Python, platform-independent)

```python
import json
import numpy as np
from huggingface_hub import hf_hub_download
from tensorflow import keras

repo_id = "{repo_id}"
model_path = hf_hub_download(repo_id=repo_id, filename="model.keras")
metadata_path = hf_hub_download(repo_id=repo_id, filename="metadata.json")

with open(metadata_path, "r", encoding="utf-8") as f:
    metadata = json.load(f)

model = keras.models.load_model(model_path)
sequence_length = int(metadata["sequence_length"])
activity_to_idx = metadata["activity_to_idx"]
lifecycle_to_idx = metadata["lifecycle_to_idx"]

def pad(xs, n):
    return ([0] * (n - len(xs)) + xs)[-n:]

# Example history:
activity_hist = ["A_Create Application", "A_Submitted"]
lifecycle_hist = ["complete", "complete"]

X_act = np.array([pad([activity_to_idx.get(a, 0) for a in activity_hist], sequence_length)], dtype=np.int32)
X_life = np.array([pad([lifecycle_to_idx.get(l, 0) for l in lifecycle_hist], sequence_length)], dtype=np.int32)

pred_activity_probs, pred_lifecycle_probs = model.predict([X_act, X_life], verbose=0)
next_activity_idx = int(np.argmax(pred_activity_probs[0]))
next_lifecycle_idx = int(np.argmax(pred_lifecycle_probs[0]))

idx_to_activity = {{int(k): v for k, v in metadata["idx_to_activity"].items()}}
idx_to_lifecycle = {{int(k): v for k, v in metadata["idx_to_lifecycle"].items()}}

print("next_activity:", idx_to_activity.get(next_activity_idx))
print("next_lifecycle:", idx_to_lifecycle.get(next_lifecycle_idx))
```
"""


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        shutil.copy2(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload a trained next_activity_prediction_lifecycle_dual model to Hugging Face."
    )
    parser.add_argument("--repo-id", required=True, help="Hugging Face repo id, e.g. username/repo-name")
    parser.add_argument(
        "--model-root",
        default="next_activity_prediction_lifecycle_dual/models",
        help="Root folder containing comparison_summary.json and trained models",
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Specific trained model folder (contains model.keras + metadata.json). "
        "If omitted, best model from comparison_summary.json is used.",
    )
    parser.add_argument("--private", action="store_true", help="Create/update private model repo")
    parser.add_argument("--token", default=None, help="HF token. If omitted, HF_TOKEN env var is used.")
    args = parser.parse_args()

    token = args.token or os.getenv("HF_TOKEN") or get_token()
    if not token:
        raise ValueError("Missing Hugging Face token. Set HF_TOKEN, run hf login, or pass --token.")

    model_root = Path(args.model_root)
    model_dir = Path(args.model_dir) if args.model_dir else _load_best_model_dir(model_root)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory does not exist: {model_dir}")

    required_files = ["model.keras", "metadata.json"]
    missing = [name for name in required_files if not (model_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required files in {model_dir}: {missing}")

    metrics = None
    metrics_path = model_dir / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics_doc = json.load(f)
            metrics = metrics_doc.get("metrics", metrics_doc)

    api = HfApi()
    api.create_repo(repo_id=args.repo_id, repo_type="model", private=args.private, exist_ok=True, token=token)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        _copy_if_exists(model_dir / "model.keras", tmp / "model.keras")
        _copy_if_exists(model_dir / "metadata.json", tmp / "metadata.json")
        _copy_if_exists(model_dir / "metrics.json", tmp / "metrics.json")
        _copy_if_exists(model_dir / "history.json", tmp / "history.json")
        _copy_if_exists(model_dir / "checkpoints" / "best_model.keras", tmp / "best_model.keras")

        model_card = _build_model_card(args.repo_id, metrics, model_dir)
        (tmp / "README.md").write_text(model_card, encoding="utf-8")

        api.upload_folder(
            folder_path=str(tmp),
            repo_id=args.repo_id,
            repo_type="model",
            token=token,
            commit_message="Upload dual lifecycle next-activity model",
        )

    print(f"Upload completed: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
