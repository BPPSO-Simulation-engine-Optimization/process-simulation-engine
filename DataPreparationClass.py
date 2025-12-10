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

        # 1. Initial Filtering
        df = self.parse_into_correct_data_types(self.data_log_df)
        df = self.filter_incomplete_cases(df)
        df = self.filter_missing_targets(df)
        df = self.delete_top1_percent_duration_outliers(df)

        # 2. Feature Engineering (that needs timestamps)
        df = self.unroll_loops(df)
        df = self.calculate_durations(df)
        df = self.extract_datetime_features(df)
        df = self.add_aggregate_features(df)

        # 3. Encoding and Scaling
        df = self.one_hot_encode_categorical_columns(df)
        df = self.encode_large_categorical_columns(df)
        
        # 4. Cleaning and Imputation
        df = self.impute_missing_values(df)
        df = self.remove_correlated_features(df)
        df = self.scale_numerical_features(df)

        # 5. Final Checks and Cleanup
        self.check_concept_drift(df)
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

    def filter_incomplete_cases(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Removes cases that do not have a definitive endpoint.
        Endpoints: A_Cancelled, A_Denied, A_Pending, O_Accepted, O_Refused, O_Cancelled
        Source: BPIC17 Report - Winner Academic: https://ais.win.tue.nl/bpi/2017/bpi2017_winner_academic.pdf
        """
        endpoints = [
            "A_Cancelled",
            "A_Denied",
            "A_Pending",
            "O_Accepted",
            "O_Refused",
            "O_Cancelled"
        ]
        
        # Identify cases that have at least one endpoint activity
        cases_with_endpoint = df[df["concept:name"].isin(endpoints)]["case:concept:name"].unique()
        return df[df["case:concept:name"].isin(cases_with_endpoint)].copy()

    def filter_missing_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Removes records missing the target variable -> offers where 'Selected' is empty.
        Source: BPIC17 Report - Winner Professional: https://ais.win.tue.nl/bpi/2017/bpi2017_winner_professional.pdf
        """    

        cases_with_target = df.groupby("case:concept:name")["Selected"].apply(lambda x: x.notna().any())
        valid_cases = cases_with_target[cases_with_target].index
        
        return df[df["case:concept:name"].isin(valid_cases)].copy()

    # Outlier entfernen (oberstes & unterstes 1% der Case-Duration) | TODO: to be discussed
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

    # TODO: to be discussed
    def unroll_loops(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Unrolls loops by appending the occurrence count to the activity name.
        e.g. W_Validate application -> W_Validate application 1
        """
        # Ensure sorted by time
        df = df.sort_values(["case:concept:name", "time:timestamp"])
        
        # Group by case and activity to get cumulative count
        df["occurrence"] = df.groupby(["case:concept:name", "concept:name"]).cumcount() + 1
        
        # Update concept:name
        df["concept:name"] = df["concept:name"].astype(str) + " " + df["occurrence"].astype(str)
        
        # Drop temporary column
        df = df.drop(columns=["occurrence"])
        
        return df

    def calculate_durations(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates time since case start for each event.
        """
        df = df.sort_values(["case:concept:name", "time:timestamp"])
        
        df["time_since_start"] = (
            df["time:timestamp"] - 
            df.groupby("case:concept:name")["time:timestamp"].transform("min")
        ).dt.total_seconds()
    
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

    def add_aggregate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds aggregate case-level features.
        """
    
        # Count of A_Incomplete per case
        incompletes = df[df["concept:name"].astype(str).str.contains("A_Incomplete", na=False)].groupby("case:concept:name").size()
        df["num_incomplete"] = df["case:concept:name"].map(incompletes).fillna(0)
        
        # Flag for W_Validate application
        has_validation = df[df["concept:name"].astype(str).str.contains("W_Validate application", na=False)]["case:concept:name"].unique()
        df["has_validation"] = df["case:concept:name"].isin(has_validation).astype(int)

        # TODO: to be discussed -> more/other?
        
        return df

    # One-Hot-Encoding
    def one_hot_encode_categorical_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        cols_to_onehot = [
            "Action",
            "EventOrigin", 
            "lifecycle:transition",
            "case:ApplicationType",
            "case:LoanGoal"
        ]
        df = pd.get_dummies(df, columns=cols_to_onehot, drop_first=False)
        return df


    # Label-Encoding für große Kategorische Spalten 
    # TODO: Label encoding problematisch für Neural Nets (impliziert fake Ordnung: User_0 < User_1 < User_2)
    # Alternativen für NNs:
    #   - Frequency Encoding: Map zu Häufigkeit (simpel, funktioniert gut)
    #   - Entity Embeddings: NN lernt eigene Repräsentationen (beste Performance, mehr Aufwand)
    #   - Target Encoding: Map zu avg target per category (powerful aber overfitting risk)
    # Für tree-based models ist label encoding OK
    def encode_large_categorical_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        cols_to_label = [
            "org:resource",
            "concept:name",
            # "case:concept:name", => case ID, no feature | only introduce noise
            # "OfferID", => IDs, no feature | only introduce noise
            # "EventID", => IDs, not feature | only introduce noise
        ]

        for col in cols_to_label:
            if col in df.columns:
                df[col] = df[col].astype("category").cat.codes

        return df

    def impute_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Imputes missing numerical values with the median.
        """
        numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
        for col in numeric_cols:
            if df[col].isna().any():
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
        return df

    def remove_correlated_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Removes highly correlated features (> 0.9 correlation).
        """
        # Only for numeric columns
        numeric_df = df.select_dtypes(include=["float64", "int64"])
        
        if numeric_df.empty:
            return df
            
        corr_matrix = numeric_df.corr().abs()
        
        # Select upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Find features with correlation greater than 0.9
        to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
        
        if to_drop:
            print(f"Dropping correlated features: {to_drop}")
            df = df.drop(columns=to_drop)
            
        return df

    def scale_numerical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Scale only true numerical features, exclude one-hot encoded and label-encoded columns.
        WARNING: Fit scaler only on train set to avoid data leakage!
        """

        # Define which columns are actually numerical (not encoded categoricals)
        true_numeric_cols = [
            "case:RequestedAmount",
            "FirstWithdrawalAmount", 
            "NumberOfTerms",
            "MonthlyCost",
            "CreditScore",
            "OfferedAmount",
            "time_since_start",
            "num_incomplete"
        ]
    
        # Only scale columns that exist in df
        cols_to_scale = [col for col in true_numeric_cols if col in df.columns]
    
        scaler = MinMaxScaler()
        df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
    
        return df

    def check_concept_drift(self, df: pd.DataFrame):
        """
        Checks for concept drift by comparing activity sets in the first and last 20% of the log.
        """
        df = df.sort_values("time:timestamp")
        n = len(df)
        chunk_size = int(n * 0.2)
        
        if chunk_size == 0:
            return
            
        first_chunk = df.iloc[:chunk_size]
        last_chunk = df.iloc[-chunk_size:]
        
        activities_first = set(first_chunk["concept:name"].unique())
        activities_last = set(last_chunk["concept:name"].unique())
        
        if activities_first != activities_last:
            print("WARNING: Potential Concept Drift detected!")
            print(f"Activities in first 20% but not last: {activities_first - activities_last}")
            print(f"Activities in last 20% but not first: {activities_last - activities_first}")
        else:
            print("No obvious structural concept drift (activity set change) detected.")
    
    # Unnötige Spalten löschen
    def drop_unused_columns(self, df):

        cols_to_drop = ["time:timestamp"]

        df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

        return df
    