import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.preprocessing.sequence import pad_sequences


class SequenceEncoder:
    """Encodes and pads activity, resource, and duration sequences."""

    def __init__(self):
        self.activity_encoder = LabelEncoder()
        self.resource_encoder = LabelEncoder()
        self.label_encoder = LabelEncoder()
        self.max_seq_len = None

    def _collect_unique(self, sequences):
        unique = set()
        for seq in sequences:
            unique.update(seq)
        return sorted(unique)

    def fit(self, df):
        activities = self._collect_unique(df["sequence"])
        resources = self._collect_unique(df["sequence_resources"])

        self.activity_encoder.fit(activities)
        self.resource_encoder.fit(resources)
        self.label_encoder.fit(df["label"])

        encoded_acts = [self.activity_encoder.transform(seq) for seq in df["sequence"]]
        self.max_seq_len = max(len(seq) for seq in encoded_acts)

        return self

    def transform(self, df):
        act_encoded = [self.activity_encoder.transform(seq) for seq in df["sequence"]]
        res_encoded = [self.resource_encoder.transform(seq) for seq in df["sequence_resources"]]
        dur_seqs = df["sequence_durations"].tolist()

        X_acts = pad_sequences(act_encoded, maxlen=self.max_seq_len, padding="pre").astype("int32")
        X_res = pad_sequences(res_encoded, maxlen=self.max_seq_len, padding="pre").astype("int32")
        X_durs = pad_sequences(dur_seqs, maxlen=self.max_seq_len, padding="pre", dtype="float32").astype("float32")

        y = self.label_encoder.transform(df["label"]).astype("int32")

        return X_acts, X_durs, X_res, y

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)

    def add_unknown_token(self, token="UNKNOWN"):
        if token not in self.activity_encoder.classes_:
            self.activity_encoder.classes_ = np.append(self.activity_encoder.classes_, token)
        if token not in self.resource_encoder.classes_:
            self.resource_encoder.classes_ = np.append(self.resource_encoder.classes_, token)
        return self

    @property
    def num_activities(self):
        return len(self.activity_encoder.classes_)

    @property
    def num_resources(self):
        return len(self.resource_encoder.classes_)

    @property
    def num_classes(self):
        return len(self.label_encoder.classes_)


class ContextEncoder:
    """Encodes and scales context attributes."""

    def __init__(self, keys):
        self.keys = keys
        self.encoders = {}
        self.dim = 0

    def fit(self, df):
        for col in self.keys:
            col_data = df[col]
            if col_data.dtype == "object" or col_data.dtype.name == "string":
                enc = LabelEncoder()
                enc.fit(col_data.astype(str))
                self.encoders[col] = enc
            else:
                scaler = StandardScaler()
                scaler.fit(col_data.values.reshape(-1, 1))
                self.encoders[col] = scaler

        self.dim = len(self.keys)
        return self

    def transform(self, df):
        result = df[self.keys].copy()

        for col in self.keys:
            enc = self.encoders[col]
            if isinstance(enc, LabelEncoder):
                result[col] = enc.transform(df[col].astype(str))
            else:
                result[col] = enc.transform(df[col].values.reshape(-1, 1)).flatten()

        return result.to_numpy().astype("float32")

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)


def prepare_sequences_and_labels(df):
    """Legacy function wrapper for backward compatibility."""
    encoder = SequenceEncoder()
    X_acts, X_durs, X_res, y = encoder.fit_transform(df)
    return (
        X_acts,
        X_durs,
        X_res,
        encoder.activity_encoder,
        encoder.resource_encoder,
        encoder.label_encoder,
        y,
        encoder.max_seq_len,
    )


def prepare_context_attributes(df, context_keys):
    """Legacy function wrapper for backward compatibility."""
    encoder = ContextEncoder(context_keys)
    X_context = encoder.fit_transform(df)
    return X_context, encoder.dim, encoder.encoders

