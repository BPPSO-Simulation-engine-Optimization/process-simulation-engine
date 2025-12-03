import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


class DataPreparationClass:

    def __init__(self, data_log_df: pd.DataFrame):
        self.data_log_df = data_log_df

    def prepare_data(self) -> pd.DataFrame:

        print("------------------ Vor Data Preparation ------------------")
        print("Anzahl an Cases:   " + str(self.data_log_df["case:concept:name"].nunique()))
        print("Anzahl an Events:  " + str(len(self.data_log_df)))
        print("Anzahl an Spalten: " + str(self.data_log_df.shape[1]))

        df = self.parse_into_correct_data_types(self.data_log_df)
        df = self.delete_top1_percent_duration_outliers(df)
        df = self.extract_datetime_features(df)
        df = self.one_hot_encode_categorical_columns(df)
        df = self.encode_large_categorical_columns(df)
        df = self.scale_numerical_features(df)
        df = self.drop_unused_columns(df)

        print("\n------------------ Nach Data Preparation ------------------")
        print("Anzahl an Cases:   " + str(df["case:concept:name"].nunique()))
        print("Anzahl an Events:  " + str(len(df)))
        print("Anzahl an Spalten: " + str(df.shape[1]))

        return df

    # Datentypen korrekt setzen
    def parse_into_correct_data_types(self, df: pd.DataFrame) -> pd.DataFrame:

        df["time:timestamp"] = pd.to_datetime(df["time:timestamp"], errors="coerce")

        numeric_cols = [
            "case:RequestedAmount",
            "FirstWithdrawalAmount",
            "NumberOfTerms",
            "MonthlyCost",
            "CreditScore",
            "OfferedAmount"
        ]

        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    # Outlier entfernen (oberstes & unterstes 1% der Case-Duration)
    def delete_top1_percent_duration_outliers(self, df: pd.DataFrame) -> pd.DataFrame:

        case_durations = (
            df.groupby("case:concept:name")["time:timestamp"]
            .agg(lambda x: (x.max() - x.min()).total_seconds())
        )

        lower_bound = case_durations.quantile(0.01)
        upper_bound = case_durations.quantile(0.99)

        valid_cases = case_durations[
            (case_durations >= lower_bound) & (case_durations <= upper_bound)
        ].index

        df_filtered = df[df["case:concept:name"].isin(valid_cases)].copy()

        return df_filtered

    # One-Hot-Encoding
    def one_hot_encode_categorical_columns(self, df: pd.DataFrame) -> pd.DataFrame:

        cols_to_onehot = [
            "Action",
            "EventOrigin",
            "lifecycle:transition",
            "case:ApplicationType",
            "case:LoanGoal"
        ]

        df = pd.get_dummies(df, columns=cols_to_onehot, drop_first=True)
        return df

    # Label-Encoding für große Kategorische Spalten
    def encode_large_categorical_columns(self, df: pd.DataFrame) -> pd.DataFrame:

        cols_to_label = [
            "org:resource",
            "concept:name",
            "EventID",
            "case:concept:name",
            "OfferID"
        ]

        for col in cols_to_label:
            if col in df.columns:
                df[col] = df[col].astype("category").cat.codes

        return df

    # Datetime-Features extrahieren + zyklische Kodierung
    def extract_datetime_features(self, df: pd.DataFrame) -> pd.DataFrame:

        df["year"] = df["time:timestamp"].dt.year
        df["month"] = df["time:timestamp"].dt.month
        df["day"] = df["time:timestamp"].dt.day
        df["hour"] = df["time:timestamp"].dt.hour
        df["weekday"] = df["time:timestamp"].dt.weekday

        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

        return df

    # Numerische Features skalieren
    def scale_numerical_features(self, df):

        numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns

        scaler = MinMaxScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

        return df

    # Unnötige Spalten löschen
    def drop_unused_columns(self, df):

        cols_to_drop = ["time:timestamp"]

        return df.drop(columns=[c for c in cols_to_drop if c in df.columns])
