from collections import defaultdict
import pandas as pd


class TrainingDataGenerator:
    """Generates enriched training datasets from event logs for decision points.

    Supports two modes:
    1. Pre-processed mode: If DataFrame has 'processing_time' column (from LifecycleFilter),
       uses actual processing times instead of computing inter-event durations.
    2. Legacy mode: Computes inter-event durations from consecutive timestamps.
    """

    CASE_COLUMNS = ["case:LoanGoal", "case:ApplicationType", "case:RequestedAmount"]
    PROCESSING_TIME_COLUMN = "processing_time"

    def __init__(self, df_log, decision_point_map, max_history=10, min_seq_count=20, min_class_count=10):
        self._log = df_log
        self._dp_map = decision_point_map
        self._max_history = max_history
        self._min_seq_count = min_seq_count
        self._min_class_count = min_class_count
        self._has_processing_time = self.PROCESSING_TIME_COLUMN in df_log.columns

    def _build_transition_lookup(self):
        transitions = defaultdict(set)
        for dp, cfg in self._dp_map.items():
            for prev in cfg["incoming"]:
                for nxt in cfg["outgoing"]:
                    transitions[(prev, nxt)].add(dp)

        return {key: list(dps)[0] for key, dps in transitions.items() if len(dps) == 1}

    def _extract_case_attributes(self):
        return (
            self._log.drop_duplicates("case:concept:name")[["case:concept:name"] + self.CASE_COLUMNS]
            .set_index("case:concept:name")
        )

    def _compute_durations_legacy(self, timestamps, start_idx, end_idx):
        """Compute inter-event durations (legacy mode)."""
        durations = []
        for j in range(start_idx + 1, end_idx + 1):
            delta = (timestamps[j] - timestamps[j - 1]).total_seconds()
            durations.append(delta)
        if len(durations) < (end_idx - start_idx + 1):
            durations.insert(0, 0.0)
        return durations

    def _get_durations(self, processing_times, timestamps, start_idx, end_idx):
        """
        Get durations for a sequence.

        If processing_time column exists, use actual processing times.
        Otherwise, fall back to inter-event durations (legacy mode).
        """
        if self._has_processing_time and processing_times is not None:
            # Use actual processing times from the collapsed events
            return processing_times[start_idx : end_idx + 1]
        else:
            # Legacy: compute inter-event durations
            return self._compute_durations_legacy(timestamps, start_idx, end_idx)

    def _process_case(self, case_id, group, transition_map, case_attrs):
        events = group["concept:name"].tolist()
        resources = group["org:resource"].tolist()
        timestamps = group["time:timestamp"].tolist()

        # Get processing times if available
        processing_times = None
        if self._has_processing_time:
            processing_times = group[self.PROCESSING_TIME_COLUMN].tolist()

        rows_by_dp = defaultdict(list)

        for i in range(len(events) - 1):
            prev_act, next_act = events[i], events[i + 1]
            dp = transition_map.get((prev_act, next_act))

            if dp is None:
                continue

            start = max(0, i - self._max_history + 1)
            seq = events[start : i + 1]
            seq_res = resources[start : i + 1]
            seq_ts = timestamps[start : i + 1]
            seq_dur = self._get_durations(processing_times, timestamps, start, i)

            case_feats = case_attrs.loc[case_id].to_dict() if case_id in case_attrs.index else {}

            rows_by_dp[dp].append({
                "case_id": case_id,
                "sequence": seq,
                "sequence_resources": seq_res,
                "sequence_durations": seq_dur,
                "sequence_timestamps": seq_ts,
                "label": next_act,
                **case_feats,
            })

        return rows_by_dp

    def _filter_dataframe(self, df):
        df["_seq_tuple"] = df["sequence"].apply(tuple)
        seq_counts = df["_seq_tuple"].value_counts()
        valid_seqs = set(seq_counts[seq_counts > self._min_seq_count].index)
        df = df[df["_seq_tuple"].isin(valid_seqs)].drop(columns=["_seq_tuple"])

        label_counts = df["label"].value_counts()
        valid_labels = label_counts[label_counts >= self._min_class_count].index
        df = df[df["label"].isin(valid_labels)]

        return df

    def generate(self):
        transition_map = self._build_transition_lookup()
        case_attrs = self._extract_case_attributes()

        dp_rows = defaultdict(list)
        sorted_log = self._log.sort_values(["case:concept:name", "time:timestamp"])

        for case_id, group in sorted_log.groupby("case:concept:name"):
            case_dp_rows = self._process_case(case_id, group, transition_map, case_attrs)
            for dp, rows in case_dp_rows.items():
                dp_rows[dp].extend(rows)

        result = {}
        for dp, rows in dp_rows.items():
            if not rows:
                continue

            df = pd.DataFrame(rows)
            df = self._filter_dataframe(df)

            if not df.empty and df["label"].nunique() >= 2:
                result[dp] = df

        return result


def generate_enriched_training_sets_simple(
    df_log, bpmn_decision_point_map, max_history_len=10, min_sequence_count=20, min_class_count=10
):
    """Legacy function wrapper for backward compatibility."""
    generator = TrainingDataGenerator(
        df_log,
        bpmn_decision_point_map,
        max_history=max_history_len,
        min_seq_count=min_sequence_count,
        min_class_count=min_class_count,
    )
    return generator.generate()

