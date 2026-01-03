import os
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model


def _expand_and_cast(x):
    return tf.cast(tf.expand_dims(x, axis=-1), tf.float32)


class ModelPersistence:
    """Saves and loads model bundles to/from disk."""

    MODEL_FILE = "model.keras"
    ACTIVITY_ENCODER_FILE = "activity_encoder.pkl"
    RESOURCE_ENCODER_FILE = "resource_encoder.pkl"
    LABEL_ENCODER_FILE = "label_encoder.pkl"
    CONTEXT_ENCODERS_FILE = "context_encoders.pkl"
    METADATA_FILE = "metadata.pkl"

    @classmethod
    def save(cls, bundle, directory):
        os.makedirs(directory, exist_ok=True)

        bundle["model"].save(os.path.join(directory, cls.MODEL_FILE))
        joblib.dump(bundle["activity_encoder"], os.path.join(directory, cls.ACTIVITY_ENCODER_FILE))
        joblib.dump(bundle["resource_encoder"], os.path.join(directory, cls.RESOURCE_ENCODER_FILE))
        joblib.dump(bundle["label_encoder"], os.path.join(directory, cls.LABEL_ENCODER_FILE))
        joblib.dump(bundle["context_encoders"], os.path.join(directory, cls.CONTEXT_ENCODERS_FILE))

        metadata = {
            "context_keys": bundle["context_keys"],
            "max_seq_len": bundle["max_seq_len"],
        }
        joblib.dump(metadata, os.path.join(directory, cls.METADATA_FILE))

    @classmethod
    def load(cls, directory):
        model = load_model(
            os.path.join(directory, cls.MODEL_FILE),
            custom_objects={"expand_and_cast": _expand_and_cast},
            safe_mode=False,
        )

        activity_enc = joblib.load(os.path.join(directory, cls.ACTIVITY_ENCODER_FILE))
        resource_enc = joblib.load(os.path.join(directory, cls.RESOURCE_ENCODER_FILE))
        label_enc = joblib.load(os.path.join(directory, cls.LABEL_ENCODER_FILE))
        context_encs = joblib.load(os.path.join(directory, cls.CONTEXT_ENCODERS_FILE))
        metadata = joblib.load(os.path.join(directory, cls.METADATA_FILE))

        return {
            "model": model,
            "activity_encoder": activity_enc,
            "resource_encoder": resource_enc,
            "label_encoder": label_enc,
            "context_encoders": context_encs,
            "context_keys": metadata["context_keys"],
            "max_seq_len": metadata["max_seq_len"],
        }

    @classmethod
    def save_all(cls, models_dict, base_directory):
        os.makedirs(base_directory, exist_ok=True)
        for dp_name, bundle in models_dict.items():
            safe_name = dp_name.replace(" ", "_").replace(":", "_")
            dp_dir = os.path.join(base_directory, safe_name)
            cls.save(bundle, dp_dir)

    @classmethod
    def load_all(cls, base_directory):
        models = {}
        for entry in os.listdir(base_directory):
            dp_dir = os.path.join(base_directory, entry)
            if os.path.isdir(dp_dir) and os.path.exists(os.path.join(dp_dir, cls.MODEL_FILE)):
                dp_name = entry.replace("_", " ")
                models[dp_name] = cls.load(dp_dir)
        return models

