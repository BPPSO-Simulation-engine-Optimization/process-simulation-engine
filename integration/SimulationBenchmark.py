"""
SimulationBenchmark Class
Descriptive comparison of original and simulated event logs without scoring.
Extracts and tabulates key metrics from both logs side-by-side.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from collections import Counter
import warnings
import os

try:
    import pm4py
    PM4PY_AVAILABLE = True
except ImportError:
    PM4PY_AVAILABLE = False
    warnings.warn("pm4py not available. XES file loading will not work.")

warnings.filterwarnings('ignore')


class SimulationBenchmark:
    """
    Descriptive benchmark for comparing original and simulated event logs.
    No scoring - purely extracts and compares metrics in tabular form.
    
    Expected DataFrame schema:
    - case:concept:name (case ID)
    - concept:name (activity name)
    - time:timestamp (timestamp)
    - org:resource (optional, resource name)
    
    Accepts either:
    - .xes file paths (str) - will be loaded with pm4py
    - pandas DataFrames - used directly
    """
    
    def __init__(self, 
                 original_log: Union[str, pd.DataFrame], 
                 simulated_log: Union[str, pd.DataFrame]):
        """
        Initialize benchmark with two event logs.
        
        Args:
            original_log: Original event log as DataFrame or path to .xes file
            simulated_log: Simulated event log as DataFrame or path to .xes file
        """
        # Load logs (from file or use DataFrame directly)
        self.original_log = self._load_log(original_log, 'original')
        self.simulated_log = self._load_log(simulated_log, 'simulated')
        
        # Standardize column names
        self._standardize_columns()
        
        # Results storage
        self.results = {}
    
    def _load_log(self, log_input: Union[str, pd.DataFrame], log_name: str) -> pd.DataFrame:
        """
        Load event log from file or use DataFrame directly.
        
        Args:
            log_input: Either a file path (str) or pandas DataFrame
            log_name: Name for error messages
            
        Returns:
            pandas DataFrame with event log
        """
        if isinstance(log_input, pd.DataFrame):
            return log_input.copy()
        
        elif isinstance(log_input, str):
            if not os.path.exists(log_input):
                raise FileNotFoundError(f"{log_name} log file not found: {log_input}")
            
            if not PM4PY_AVAILABLE:
                raise ImportError(
                    "pm4py is required to load .xes files. "
                    "Install it with: pip install pm4py"
                )
            
            print(f"Loading {log_name} log from {log_input}...")
            
            # Load with pm4py
            if log_input.endswith('.xes') or log_input.endswith('.xes.gz'):
                log = pm4py.read_xes(log_input)
            elif log_input.endswith('.csv'):
                log = pd.read_csv(log_input)
            else:
                raise ValueError(
                    f"Unsupported file format for {log_name} log. "
                    "Supported formats: .xes, .xes.gz, .csv"
                )
            
            return log
        
        else:
            raise TypeError(
                f"{log_name} log must be either a file path (str) or pandas DataFrame"
            )
        
    def _standardize_columns(self):
        """Ensure consistent column naming."""
        # Map common column name variations
        col_mapping = {
            'case:concept:name': 'case_id',
            'concept:name': 'activity',
            'time:timestamp': 'timestamp',
            'org:resource': 'resource'
        }
        
        for log in [self.original_log, self.simulated_log]:
            for old_col, new_col in col_mapping.items():
                if old_col in log.columns:
                    log.rename(columns={old_col: new_col}, inplace=True)
            
            # Ensure timestamp is datetime
            if 'timestamp' in log.columns:
                log['timestamp'] = pd.to_datetime(log['timestamp'])
                
    def compute_all_metrics(self) -> Dict:
        """
        Compute all benchmark metrics and return comprehensive comparison.
        
        Returns:
            Dictionary with all comparison results
        """
        print("Computing benchmark metrics...")
        
        # Basic statistics
        print("  - Basic statistics...")
        self.results['basic_stats'] = self._compare_basic_stats()
        
        # Events per case
        print("  - Events per case...")
        self.results['events_per_case'] = self._compare_events_per_case()
        
        # Case throughput time
        print("  - Case throughput time...")
        self.results['throughput_time'] = self._compare_throughput_time()
        
        # Case arrivals
        print("  - Case arrivals...")
        self.results['arrivals'] = self._compare_arrivals()
        
        # Case completions
        print("  - Case completions...")
        self.results['completions'] = self._compare_completions()
        
        # Control flow (directly-follows graph)
        print("  - Control flow (DFG)...")
        self.results['dfg_top'] = self._compare_dfg_top()
        self.results['dfg_comparison'] = self._compare_dfg_detailed()
        
        # Trace variants
        print("  - Trace variants...")
        self.results['variants_top'] = self._compare_variants_top()
        self.results['variants_comparison'] = self._compare_variants_detailed()
        
        # Start/End activities
        print("  - Start/End activities...")
        self.results['start_activities'] = self._compare_start_activities()
        self.results['end_activities'] = self._compare_end_activities()
        
        # Activity durations (time to next event)
        print("  - Activity durations...")
        self.results['activity_durations'] = self._compare_activity_durations()
        
        # Resource statistics (if available)
        if 'resource' in self.original_log.columns and 'resource' in self.simulated_log.columns:
            print("  - Resource statistics...")
            self.results['resource_stats'] = self._compare_resource_stats()
            self.results['activity_resource'] = self._compare_activity_resource()
        
        print("Benchmark computation complete!")
        return self.results
    
    def _compare_basic_stats(self) -> pd.DataFrame:
        """Compare basic statistics between logs."""
        stats = []
        
        for name, log in [('Original', self.original_log), ('Simulated', self.simulated_log)]:
            n_events = len(log)
            n_cases = log['case_id'].nunique()
            n_activities = log['activity'].nunique()
            
            stat_dict = {
                'Log': name,
                'Number of Events': n_events,
                'Number of Cases': n_cases,
                'Number of Unique Activities': n_activities
            }
            
            # Add resources if available
            if 'resource' in log.columns:
                stat_dict['Number of Unique Resources'] = log['resource'].nunique()
            
            stats.append(stat_dict)
        
        return pd.DataFrame(stats)
    
    def _compare_events_per_case(self) -> pd.DataFrame:
        """Compare events per case distribution."""
        stats = []
        
        for name, log in [('Original', self.original_log), ('Simulated', self.simulated_log)]:
            events_per_case = log.groupby('case_id').size()
            
            stats.append({
                'Log': name,
                'Mean': events_per_case.mean(),
                'Median': events_per_case.median(),
                'P90': events_per_case.quantile(0.90),
                'Max': events_per_case.max()
            })
        
        return pd.DataFrame(stats)
    
    def _compare_throughput_time(self) -> pd.DataFrame:
        """Compare case throughput times (cycle times) in hours."""
        stats = []
        
        for name, log in [('Original', self.original_log), ('Simulated', self.simulated_log)]:
            case_times = log.groupby('case_id')['timestamp'].agg(['min', 'max'])
            throughput = (case_times['max'] - case_times['min']).dt.total_seconds() / 3600  # hours
            
            stats.append({
                'Log': name,
                'Mean (hours)': throughput.mean(),
                'Median (hours)': throughput.median(),
                'P75 (hours)': throughput.quantile(0.75),
                'P90 (hours)': throughput.quantile(0.90),
                'P95 (hours)': throughput.quantile(0.95),
                'Max (hours)': throughput.max()
            })
        
        return pd.DataFrame(stats)
    
    def _compare_arrivals(self) -> pd.DataFrame:
        """Compare case arrival patterns (daily aggregation)."""
        stats = []
        
        for name, log in [('Original', self.original_log), ('Simulated', self.simulated_log)]:
            # Get first event per case (arrival)
            arrivals = log.groupby('case_id')['timestamp'].min()
            daily_arrivals = arrivals.dt.date.value_counts().sort_index()
            
            stats.append({
                'Log': name,
                'Mean Daily Arrivals': daily_arrivals.mean(),
                'Median Daily Arrivals': daily_arrivals.median(),
                'P90 Daily Arrivals': daily_arrivals.quantile(0.90),
                'Max Daily Arrivals': daily_arrivals.max()
            })
        
        return pd.DataFrame(stats)
    
    def _compare_completions(self) -> pd.DataFrame:
        """Compare case completion patterns (daily aggregation)."""
        stats = []
        
        for name, log in [('Original', self.original_log), ('Simulated', self.simulated_log)]:
            # Get last event per case (completion)
            completions = log.groupby('case_id')['timestamp'].max()
            daily_completions = completions.dt.date.value_counts().sort_index()
            
            stats.append({
                'Log': name,
                'Mean Daily Completions': daily_completions.mean(),
                'Median Daily Completions': daily_completions.median(),
                'P90 Daily Completions': daily_completions.quantile(0.90),
                'Max Daily Completions': daily_completions.max()
            })
        
        return pd.DataFrame(stats)
    
    def _extract_dfg(self, log: pd.DataFrame) -> Counter:
        """Extract directly-follows graph as edge counter."""
        edges = []
        
        for case_id, case_df in log.groupby('case_id'):
            activities = case_df.sort_values('timestamp')['activity'].tolist()
            for i in range(len(activities) - 1):
                edges.append((activities[i], activities[i+1]))
        
        return Counter(edges)
    
    def _compare_dfg_top(self, top_n: int = 20) -> pd.DataFrame:
        """Compare top-N most frequent directly-follows edges."""
        orig_dfg = self._extract_dfg(self.original_log)
        sim_dfg = self._extract_dfg(self.simulated_log)
        
        # Get top edges from each
        orig_top = orig_dfg.most_common(top_n)
        sim_top = sim_dfg.most_common(top_n)
        
        results = []
        results.append({'Log': 'Original', 'Metric': 'Total Edges', 'Value': len(orig_dfg)})
        results.append({'Log': 'Simulated', 'Metric': 'Total Edges', 'Value': len(sim_dfg)})
        
        for i, (edge, count) in enumerate(orig_top, 1):
            results.append({
                'Log': 'Original',
                'Metric': f'Top {i} Edge',
                'Value': f'{edge[0]} → {edge[1]} ({count})'
            })
        
        for i, (edge, count) in enumerate(sim_top, 1):
            results.append({
                'Log': 'Simulated',
                'Metric': f'Top {i} Edge',
                'Value': f'{edge[0]} → {edge[1]} ({count})'
            })
        
        return pd.DataFrame(results)
    
    def _compare_dfg_detailed(self, top_n: int = 30) -> pd.DataFrame:
        """Detailed comparison of DFG edges (union of top edges from both logs)."""
        orig_dfg = self._extract_dfg(self.original_log)
        sim_dfg = self._extract_dfg(self.simulated_log)
        
        # Get union of top edges
        orig_top_edges = [edge for edge, _ in orig_dfg.most_common(top_n)]
        sim_top_edges = [edge for edge, _ in sim_dfg.most_common(top_n)]
        all_edges = list(set(orig_top_edges + sim_top_edges))
        
        # Calculate totals
        orig_total = sum(orig_dfg.values())
        sim_total = sum(sim_dfg.values())
        
        # Build comparison table
        comparison = []
        for edge in all_edges:
            orig_count = orig_dfg.get(edge, 0)
            sim_count = sim_dfg.get(edge, 0)
            
            # Calculate ranks
            orig_rank = sorted(orig_dfg.values(), reverse=True).index(orig_count) + 1 if orig_count > 0 else None
            sim_rank = sorted(sim_dfg.values(), reverse=True).index(sim_count) + 1 if sim_count > 0 else None
            
            comparison.append({
                'From': edge[0],
                'To': edge[1],
                'Original Count': orig_count,
                'Original Share (%)': 100 * orig_count / orig_total if orig_total > 0 else 0,
                'Original Rank': orig_rank,
                'Simulated Count': sim_count,
                'Simulated Share (%)': 100 * sim_count / sim_total if sim_total > 0 else 0,
                'Simulated Rank': sim_rank
            })
        
        # Sort by combined frequency
        df = pd.DataFrame(comparison)
        df['Total'] = df['Original Count'] + df['Simulated Count']
        df = df.sort_values('Total', ascending=False).drop('Total', axis=1)
        
        return df
    
    def _extract_variants(self, log: pd.DataFrame) -> Counter:
        """Extract trace variants as counter."""
        variants = []
        
        for case_id, case_df in log.groupby('case_id'):
            activities = case_df.sort_values('timestamp')['activity'].tolist()
            variant = ' → '.join(activities)
            variants.append(variant)
        
        return Counter(variants)
    
    def _compare_variants_top(self, top_n: int = 15) -> pd.DataFrame:
        """Compare top-N most frequent trace variants."""
        orig_variants = self._extract_variants(self.original_log)
        sim_variants = self._extract_variants(self.simulated_log)
        
        results = []
        results.append({'Log': 'Original', 'Metric': 'Total Variants', 'Value': len(orig_variants)})
        results.append({'Log': 'Simulated', 'Metric': 'Total Variants', 'Value': len(sim_variants)})
        
        for i, (variant, count) in enumerate(orig_variants.most_common(top_n), 1):
            results.append({
                'Log': 'Original',
                'Metric': f'Top {i} Variant',
                'Value': f'{variant} ({count})'
            })
        
        for i, (variant, count) in enumerate(sim_variants.most_common(top_n), 1):
            results.append({
                'Log': 'Simulated',
                'Metric': f'Top {i} Variant',
                'Value': f'{variant} ({count})'
            })
        
        return pd.DataFrame(results)
    
    def _compare_variants_detailed(self, top_n: int = 20) -> pd.DataFrame:
        """Detailed comparison of trace variants (union of top variants from both logs)."""
        orig_variants = self._extract_variants(self.original_log)
        sim_variants = self._extract_variants(self.simulated_log)
        
        # Get union of top variants
        orig_top_vars = [var for var, _ in orig_variants.most_common(top_n)]
        sim_top_vars = [var for var, _ in sim_variants.most_common(top_n)]
        all_variants = list(set(orig_top_vars + sim_top_vars))
        
        # Calculate totals
        orig_total = sum(orig_variants.values())
        sim_total = sum(sim_variants.values())
        
        # Build comparison table
        comparison = []
        for variant in all_variants:
            orig_count = orig_variants.get(variant, 0)
            sim_count = sim_variants.get(variant, 0)
            
            # Calculate ranks
            orig_rank = sorted(orig_variants.values(), reverse=True).index(orig_count) + 1 if orig_count > 0 else None
            sim_rank = sorted(sim_variants.values(), reverse=True).index(sim_count) + 1 if sim_count > 0 else None
            
            comparison.append({
                'Variant': variant,
                'Original Count': orig_count,
                'Original Share (%)': 100 * orig_count / orig_total if orig_total > 0 else 0,
                'Original Rank': orig_rank,
                'Simulated Count': sim_count,
                'Simulated Share (%)': 100 * sim_count / sim_total if sim_total > 0 else 0,
                'Simulated Rank': sim_rank
            })
        
        # Sort by combined frequency
        df = pd.DataFrame(comparison)
        df['Total'] = df['Original Count'] + df['Simulated Count']
        df = df.sort_values('Total', ascending=False).drop('Total', axis=1)
        
        return df
    
    def _compare_start_activities(self) -> pd.DataFrame:
        """Compare distribution of start activities."""
        orig_starts = self.original_log.groupby('case_id').apply(
            lambda x: x.sort_values('timestamp')['activity'].iloc[0]
        ).value_counts()
        
        sim_starts = self.simulated_log.groupby('case_id').apply(
            lambda x: x.sort_values('timestamp')['activity'].iloc[0]
        ).value_counts()
        
        # Get union of activities
        all_activities = list(set(orig_starts.index) | set(sim_starts.index))
        
        orig_total = orig_starts.sum()
        sim_total = sim_starts.sum()
        
        comparison = []
        for activity in all_activities:
            orig_count = orig_starts.get(activity, 0)
            sim_count = sim_starts.get(activity, 0)
            
            comparison.append({
                'Activity': activity,
                'Original Count': orig_count,
                'Original Share (%)': 100 * orig_count / orig_total if orig_total > 0 else 0,
                'Simulated Count': sim_count,
                'Simulated Share (%)': 100 * sim_count / sim_total if sim_total > 0 else 0
            })
        
        df = pd.DataFrame(comparison)
        df = df.sort_values('Original Count', ascending=False)
        
        return df
    
    def _compare_end_activities(self) -> pd.DataFrame:
        """Compare distribution of end activities."""
        orig_ends = self.original_log.groupby('case_id').apply(
            lambda x: x.sort_values('timestamp')['activity'].iloc[-1]
        ).value_counts()
        
        sim_ends = self.simulated_log.groupby('case_id').apply(
            lambda x: x.sort_values('timestamp')['activity'].iloc[-1]
        ).value_counts()
        
        # Get union of activities
        all_activities = list(set(orig_ends.index) | set(sim_ends.index))
        
        orig_total = orig_ends.sum()
        sim_total = sim_ends.sum()
        
        comparison = []
        for activity in all_activities:
            orig_count = orig_ends.get(activity, 0)
            sim_count = sim_ends.get(activity, 0)
            
            comparison.append({
                'Activity': activity,
                'Original Count': orig_count,
                'Original Share (%)': 100 * orig_count / orig_total if orig_total > 0 else 0,
                'Simulated Count': sim_count,
                'Simulated Share (%)': 100 * sim_count / sim_total if sim_total > 0 else 0
            })
        
        df = pd.DataFrame(comparison)
        df = df.sort_values('Original Count', ascending=False)
        
        return df
    
    def _compare_activity_durations(self) -> pd.DataFrame:
        """Compare activity durations (time to next event within case) in hours."""
        orig_durations = self._compute_activity_durations(self.original_log)
        sim_durations = self._compute_activity_durations(self.simulated_log)
        
        # Get union of activities
        all_activities = list(set(orig_durations.keys()) | set(sim_durations.keys()))
        
        comparison = []
        for activity in all_activities:
            orig_times = orig_durations.get(activity, [])
            sim_times = sim_durations.get(activity, [])
            
            comparison.append({
                'Activity': activity,
                'Original Mean (hours)': np.mean(orig_times) if len(orig_times) > 0 else np.nan,
                'Original Median (hours)': np.median(orig_times) if len(orig_times) > 0 else np.nan,
                'Original P90 (hours)': np.quantile(orig_times, 0.9) if len(orig_times) > 0 else np.nan,
                'Simulated Mean (hours)': np.mean(sim_times) if len(sim_times) > 0 else np.nan,
                'Simulated Median (hours)': np.median(sim_times) if len(sim_times) > 0 else np.nan,
                'Simulated P90 (hours)': np.quantile(sim_times, 0.9) if len(sim_times) > 0 else np.nan
            })
        
        df = pd.DataFrame(comparison)
        df = df.sort_values('Original Mean (hours)', ascending=False)
        
        return df
    
    def _compute_activity_durations(self, log: pd.DataFrame) -> Dict[str, List[float]]:
        """Compute time to next event for each activity (in hours)."""
        durations = {}
        
        for case_id, case_df in log.groupby('case_id'):
            case_df = case_df.sort_values('timestamp')
            activities = case_df['activity'].tolist()
            timestamps = case_df['timestamp'].tolist()
            
            for i in range(len(activities) - 1):
                activity = activities[i]
                duration = (timestamps[i+1] - timestamps[i]).total_seconds() / 3600  # hours
                
                if activity not in durations:
                    durations[activity] = []
                durations[activity].append(duration)
        
        return durations
    
    def _compare_resource_stats(self) -> pd.DataFrame:
        """Compare basic resource statistics."""
        stats = []
        
        for name, log in [('Original', self.original_log), ('Simulated', self.simulated_log)]:
            if 'resource' not in log.columns:
                continue
                
            n_resources = log['resource'].nunique()
            events_per_resource = log.groupby('resource').size()
            
            stats.append({
                'Log': name,
                'Number of Resources': n_resources,
                'Mean Events per Resource': events_per_resource.mean(),
                'Median Events per Resource': events_per_resource.median(),
                'P90 Events per Resource': events_per_resource.quantile(0.90),
                'Max Events per Resource': events_per_resource.max()
            })
        
        return pd.DataFrame(stats)
    
    def _compare_activity_resource(self, top_n: int = 20) -> pd.DataFrame:
        """Compare activity-resource frequencies."""
        orig_ar = self._extract_activity_resource(self.original_log)
        sim_ar = self._extract_activity_resource(self.simulated_log)
        
        # Get union of top pairs
        orig_top_pairs = [pair for pair, _ in orig_ar.most_common(top_n)]
        sim_top_pairs = [pair for pair, _ in sim_ar.most_common(top_n)]
        all_pairs = list(set(orig_top_pairs + sim_top_pairs))
        
        # Calculate totals
        orig_total = sum(orig_ar.values())
        sim_total = sum(sim_ar.values())
        
        comparison = []
        for activity, resource in all_pairs:
            orig_count = orig_ar.get((activity, resource), 0)
            sim_count = sim_ar.get((activity, resource), 0)
            
            comparison.append({
                'Activity': activity,
                'Resource': resource,
                'Original Count': orig_count,
                'Original Share (%)': 100 * orig_count / orig_total if orig_total > 0 else 0,
                'Simulated Count': sim_count,
                'Simulated Share (%)': 100 * sim_count / sim_total if sim_total > 0 else 0
            })
        
        df = pd.DataFrame(comparison)
        df['Total'] = df['Original Count'] + df['Simulated Count']
        df = df.sort_values('Total', ascending=False).drop('Total', axis=1)
        
        return df
    
    def _extract_activity_resource(self, log: pd.DataFrame) -> Counter:
        """Extract activity-resource pairs as counter."""
        if 'resource' not in log.columns:
            return Counter()
        
        pairs = list(zip(log['activity'], log['resource']))
        return Counter(pairs)
    
    def print_summary(self):
        """Print all comparison results in a formatted way."""
        if not self.results:
            print("No results available. Run compute_all_metrics() first.")
            return
        
        print("\n" + "="*80)
        print("SIMULATION BENCHMARK REPORT")
        print("="*80)
        
        # Basic stats
        print("\n### BASIC STATISTICS ###")
        print(self.results['basic_stats'].to_string(index=False))
        
        # Events per case
        print("\n### EVENTS PER CASE ###")
        print(self.results['events_per_case'].to_string(index=False))
        
        # Throughput time
        print("\n### CASE THROUGHPUT TIME (Cycle Time) ###")
        print(self.results['throughput_time'].to_string(index=False))
        
        # Arrivals
        print("\n### CASE ARRIVALS (Daily) ###")
        print(self.results['arrivals'].to_string(index=False))
        
        # Completions
        print("\n### CASE COMPLETIONS (Daily) ###")
        print(self.results['completions'].to_string(index=False))
        
        # DFG
        print("\n### CONTROL FLOW - Directly-Follows Graph (Top Edges) ###")
        print(self.results['dfg_top'].to_string(index=False))
        
        print("\n### CONTROL FLOW - Detailed Edge Comparison ###")
        print(self.results['dfg_comparison'].to_string(index=False))
        
        # Variants
        print("\n### TRACE VARIANTS (Top) ###")
        print(self.results['variants_top'].to_string(index=False))
        
        print("\n### TRACE VARIANTS - Detailed Comparison ###")
        print(self.results['variants_comparison'].to_string(index=False))
        
        # Start/End activities
        print("\n### START ACTIVITIES ###")
        print(self.results['start_activities'].to_string(index=False))
        
        print("\n### END ACTIVITIES ###")
        print(self.results['end_activities'].to_string(index=False))
        
        # Activity durations
        print("\n### ACTIVITY DURATIONS (Time to Next Event) ###")
        print(self.results['activity_durations'].to_string(index=False))
        
        # Resources (if available)
        if 'resource_stats' in self.results:
            print("\n### RESOURCE STATISTICS ###")
            print(self.results['resource_stats'].to_string(index=False))
            
            print("\n### ACTIVITY-RESOURCE FREQUENCIES ###")
            print(self.results['activity_resource'].to_string(index=False))
        
        print("\n" + "="*80)
        print("END OF REPORT")
        print("="*80 + "\n")
    
    def export_results(self, output_path: str = 'benchmark_results.xlsx'):
        """
        Export all results to an Excel file with multiple sheets.
        
        Args:
            output_path: Path to save the Excel file
        """
        if not self.results:
            print("No results available. Run compute_all_metrics() first.")
            return
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            self.results['basic_stats'].to_excel(writer, sheet_name='Basic Stats', index=False)
            self.results['events_per_case'].to_excel(writer, sheet_name='Events per Case', index=False)
            self.results['throughput_time'].to_excel(writer, sheet_name='Throughput Time', index=False)
            self.results['arrivals'].to_excel(writer, sheet_name='Arrivals', index=False)
            self.results['completions'].to_excel(writer, sheet_name='Completions', index=False)
            self.results['dfg_top'].to_excel(writer, sheet_name='DFG Top', index=False)
            self.results['dfg_comparison'].to_excel(writer, sheet_name='DFG Detailed', index=False)
            self.results['variants_top'].to_excel(writer, sheet_name='Variants Top', index=False)
            self.results['variants_comparison'].to_excel(writer, sheet_name='Variants Detailed', index=False)
            self.results['start_activities'].to_excel(writer, sheet_name='Start Activities', index=False)
            self.results['end_activities'].to_excel(writer, sheet_name='End Activities', index=False)
            self.results['activity_durations'].to_excel(writer, sheet_name='Activity Durations', index=False)
            
            if 'resource_stats' in self.results:
                self.results['resource_stats'].to_excel(writer, sheet_name='Resource Stats', index=False)
                self.results['activity_resource'].to_excel(writer, sheet_name='Activity-Resource', index=False)
        
        print(f"Results exported to {output_path}")


# Example usage
if __name__ == "__main__":
    # Option 1: Direkt .xes Dateien laden
    benchmark = SimulationBenchmark(
        'eventlog/eventlog.xes.gz',  # Original BPIC 2017 Log
        'simulation/output/simulated_log.xes'  # TODO: Replace with path to simulated log
    )
    
    # Option 2: Mit DataFrames (wie vorher)
    # original_df = pd.DataFrame()  # Load BPIC 2017 as DataFrame
    # simulated_df = pd.DataFrame()  # Load simulated log as DataFrame
    # benchmark = SimulationBenchmark(original_df, simulated_df)
    
    # Analyse durchführen
    results = benchmark.compute_all_metrics()
    benchmark.print_summary()
    benchmark.export_results('simulation/output/benchmark_results.xlsx')
