# Process Transformer Predictor

This module implements a Next Activity Predictor using a Transformer architecture.

## Dependencies

This predictor requires:
- `tensorflow`
- `tf_keras` (for legacy Keras 2 model support)
- `huggingface_hub` (for auto-downloading models)

These are included in the project's `requirements.txt`.

## Model Loading

The model is automatically downloaded from HuggingFace (`lgk03/bpic17-process-transformer_v1`) if not found locally in `models/process_transformer`.

## Configuration

The predictor supports:
- **Temperature**: Controls randomness (default 1.0).
- **Repetition Penalty**: Reduces probability of repeating activities.

## Known Issues

- **Keras 3 Compatibility**: The pre-trained model was saved with Keras 2. The code handles this by explicitly importing `tf_keras` if installed.
- **Performance**: On Apple Silicon (M1/M2), `model.predict()` can be slow in loops. The implementation uses `model(x, training=False)` for optimized inference.
