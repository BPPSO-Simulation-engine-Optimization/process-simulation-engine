import pm4py
import pandas as pd
from pathlib import Path
from typing import Optional


class DataLoader:
    """Handles loading and preprocessing of process mining data."""

    def __init__(self):
        """Initialize the DataLoader."""
        pass

    def load_xes_to_dataframe(self, xes_file_path: str) -> pd.DataFrame:
        """
        Load XES file and convert to pandas DataFrame.

        Args:
            xes_file_path: Path to the XES file

        Returns:
            DataFrame with renamed columns (case_id, event, timestamp)
        """
        print(f"Loading XES file: {xes_file_path}")

        # Read XES file
        log = pm4py.read_xes(xes_file_path)

        # Convert to DataFrame
        df = pm4py.convert_to_dataframe(log)

        # Rename columns to standard names (same as load_csv_to_dataframe)
        rename_dict = {
            "case:concept:name": "case_id",
            "concept:name": "event",
            "time:timestamp": "timestamp"
        }

        df = df.rename(columns=rename_dict)

        # Convert timestamp to datetime and clean timezone
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
            
            # Convert to Europe/Berlin timezone and remove timezone info
            if df["timestamp"].dt.tz is not None:
                df["timestamp"] = (
                    df["timestamp"]
                    .dt.tz_convert("Europe/Berlin")
                    .dt.tz_localize(None)
                )
            else:
                # If already timezone-naive, just ensure it's datetime
                df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

            # Remove invalid timestamps
            df = df.dropna(subset=["timestamp"])

        print(f"Loaded {len(df)} events from {len(df['case_id'].unique())} cases")
        print(f"Columns: {list(df.columns)}")

        return df

    def save_dataframe_to_csv(self, df: pd.DataFrame, output_path: str) -> None:
        """
        Save DataFrame to CSV file.

        Args:
            df: DataFrame to save
            output_path: Path where to save the CSV
        """
        df.to_csv(output_path, index=False)
        print(f"Saved DataFrame to: {output_path}")

    def load_csv_to_dataframe(self, csv_file_path: str) -> pd.DataFrame:
        """
        Load CSV file into DataFrame with standard column renaming.

        Args:
            csv_file_path: Path to the CSV file

        Returns:
            DataFrame with renamed columns
        """
        print(f"Loading CSV file: {csv_file_path}")

        df = pd.read_csv(csv_file_path)

        # Rename columns to standard names
        rename_dict = {
            "case:concept:name": "case_id",
            "concept:name": "event",
            "time:timestamp": "timestamp"
        }

        df = df.rename(columns=rename_dict)

        # Convert timestamp to datetime and clean timezone
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        
        # Convert to Europe/Berlin timezone and remove timezone info
        if df["timestamp"].dt.tz is not None:
            df["timestamp"] = (
                df["timestamp"]
                .dt.tz_convert("Europe/Berlin")
                .dt.tz_localize(None)
            )
        else:
            # If already timezone-naive, just ensure it's datetime
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

        # Remove invalid timestamps
        df = df.dropna(subset=["timestamp"])

        print(f"Loaded {len(df)} events from {len(df['case_id'].unique())} cases")

        return df
