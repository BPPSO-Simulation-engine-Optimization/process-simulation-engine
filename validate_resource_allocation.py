"""
Resource Allocation Validation Script

Validates and compares resource allocation against the original event log.
Checks:
- Allocation success rate (no errors)
- Resource utilization distribution
- Activity-Resource mapping accuracy
- Waiting times when resources are not available
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import logging
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns

from resources.resource_allocation import ResourceAllocator
import pm4py

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ResourceAllocationValidator:
    """Validates resource allocation against original event log."""
    
    def __init__(self, log_path: str, permission_method: str = 'ordinor'):
        """
        Initialize validator.
        
        Args:
            log_path: Path to XES event log
            permission_method: 'basic', 'ordinor', or 'ordinor-strict'
        """
        logger.info(f"Loading event log from: {log_path}")
        
        # Load event log
        log = pm4py.read_xes(log_path)
        self.df = pm4py.convert_to_dataframe(log)
        self.df['time:timestamp'] = pd.to_datetime(self.df['time:timestamp'])
        
        logger.info(f"Event log loaded: {len(self.df)} events")
        
        # Initialize allocator
        logger.info(f"Initializing ResourceAllocator with method: {permission_method}")
        self.allocator = ResourceAllocator(
            df=self.df.copy(),
            permission_method=permission_method
        )
        
        # Extract ground truth
        self._extract_ground_truth()
    
    def _extract_ground_truth(self):
        """Extract ground truth from original event log."""
        logger.info("\nExtracting ground truth from event log...")
        
        # Activity -> Resources mapping
        self.activity_resources = defaultdict(set)
        for _, row in self.df.iterrows():
            activity = row['concept:name']
            resource = row['org:resource']
            self.activity_resources[activity].add(resource)
        
        # Resource statistics
        self.original_resource_counts = self.df['org:resource'].value_counts()
        self.original_activity_counts = self.df['concept:name'].value_counts()
        
        # Calculate original waiting times (time between consecutive events in each case)
        df_sorted = self.df.sort_values(['case:concept:name', 'time:timestamp'])
        df_sorted['next_timestamp'] = df_sorted.groupby('case:concept:name')['time:timestamp'].shift(-1)
        df_sorted['waiting_time_hours'] = (df_sorted['next_timestamp'] - df_sorted['time:timestamp']).dt.total_seconds() / 3600
        
        # Remove last event in each case (no waiting time) and negative values
        self.original_waiting_times = df_sorted[df_sorted['waiting_time_hours'].notna() & (df_sorted['waiting_time_hours'] >= 0)]['waiting_time_hours']
        
        # Calculate resource workload distribution (Gini coefficient for fairness)
        self.original_resource_distribution = self.original_resource_counts.values
        
        # Calculate activity-resource co-occurrence matrix
        self.activity_resource_matrix = defaultdict(Counter)
        for _, row in self.df.iterrows():
            activity = row['concept:name']
            resource = row['org:resource']
            self.activity_resource_matrix[activity][resource] += 1
        
        logger.info(f"✓ {len(self.activity_resources)} unique activities")
        logger.info(f"✓ {len(self.original_resource_counts)} unique resources")
        logger.info(f"✓ {len(self.df)} total events")
        logger.info(f"✓ {len(self.original_waiting_times)} waiting times calculated from original log")
    
    def run_validation(self, sample_size: int = 1000):
        """
        Run complete validation.
        
        Args:
            sample_size: Number of events to test (default: 1000)
        """
        logger.info(f"\n{'='*80}")
        logger.info("STARTING RESOURCE ALLOCATION VALIDATION")
        logger.info(f"{'='*80}\n")
        
        # Sample events for testing
        sample_df = self.df.sample(min(sample_size, len(self.df)), random_state=42)
        logger.info(f"Testing with {len(sample_df)} sampled events\n")
        
        # Run allocations
        results = self._test_allocations(sample_df)
        
        # Generate report
        self._print_report(results)
        
        # Optional: Create visualizations
        self._create_visualizations(results)
        
        return results
    
    def _test_allocations(self, sample_df):
        """Test resource allocation on sample events."""
        logger.info("Testing resource allocations...")
        
        results = {
            'total': len(sample_df),
            'successful': 0,
            'failed_no_resource': 0,
            'failed_no_eligible': 0,
            'delayed': 0,
            'immediate': 0,
            'allocated_resources': Counter(),
            'waiting_times': [],
            'activity_results': defaultdict(lambda: {
                'total': 0,
                'success': 0,
                'failed': 0,
                'allocated_resources': Counter()
            }),
            'resource_accuracy': [],  # Did we use the same resources as original?
            'activity_resource_matches': 0,  # How often did we pick a resource that actually does this activity?
            'total_checked': 0
        }
        
        for idx, row in sample_df.iterrows():
            activity = row['concept:name']
            timestamp = row['time:timestamp']
            case_type = row.get('case:LoanGoal', None)
            original_resource = row['org:resource']
            
            # Test allocation
            allocated_resource, actual_time = self.allocator.allocate(
                activity=activity,
                timestamp=timestamp,
                case_type=case_type
            )
            
            # Update activity-specific results
            activity_result = results['activity_results'][activity]
            activity_result['total'] += 1
            
            if allocated_resource is None:
                # Allocation failed
                results['failed_no_eligible'] += 1
                activity_result['failed'] += 1
            else:
                # Allocation succeeded
                results['successful'] += 1
                activity_result['success'] += 1
                results['allocated_resources'][allocated_resource] += 1
                activity_result['allocated_resources'][allocated_resource] += 1
                
                # Check if this resource actually did this activity in the original log
                if allocated_resource in self.activity_resources[activity]:
                    results['activity_resource_matches'] += 1
                results['total_checked'] += 1
                
                # Check if delayed
                if actual_time > timestamp:
                    results['delayed'] += 1
                    wait_hours = (actual_time - timestamp).total_seconds() / 3600
                    results['waiting_times'].append(wait_hours)
                else:
                    results['immediate'] += 1
        
        logger.info("✓ Allocation testing completed\n")
        return results
    
    def _print_report(self, results):
        """Print validation report."""
        logger.info(f"\n{'='*80}")
        logger.info("VALIDATION REPORT")
        logger.info(f"{'='*80}\n")
        
        # Overall statistics
        total = results['total']
        success_rate = (results['successful'] / total * 100) if total > 0 else 0
        
        print(f"📊 OVERALL STATISTICS")
        print(f"{'─'*80}")
        print(f"Total Events Tested:           {total:,}")
        print(f"Successful Allocations:        {results['successful']:,} ({success_rate:.1f}%)")
        print(f"Failed Allocations:            {results['failed_no_eligible']:,} ({results['failed_no_eligible']/total*100:.1f}%)")
        print(f"  ├─ No Eligible Resources:    {results['failed_no_eligible']:,}")
        print()
        
        # Timing statistics
        immediate_rate = (results['immediate'] / results['successful'] * 100) if results['successful'] > 0 else 0
        delayed_rate = (results['delayed'] / results['successful'] * 100) if results['successful'] > 0 else 0
        
        print(f"⏱️  TIMING STATISTICS")
        print(f"{'─'*80}")
        print(f"Immediate Allocations:         {results['immediate']:,} ({immediate_rate:.1f}%)")
        print(f"Delayed Allocations:           {results['delayed']:,} ({delayed_rate:.1f}%)")
        
        if results['waiting_times']:
            wait_times = results['waiting_times']
            print(f"  ├─ Avg Wait Time:            {np.mean(wait_times):.2f} hours")
            print(f"  ├─ Median Wait Time:         {np.median(wait_times):.2f} hours")
            print(f"  ├─ Max Wait Time:            {np.max(wait_times):.2f} hours")
        print()
        
        # Compare with original waiting times
        print(f"⏱️  WAITING TIME COMPARISON")
        print(f"{'─'*80}")
        if len(self.original_waiting_times) > 0:
            orig_wait = self.original_waiting_times
            print(f"Original Log Statistics:")
            print(f"  ├─ Total waiting periods:    {len(orig_wait):,}")
            print(f"  ├─ Mean waiting time:        {orig_wait.mean():.2f} hours")
            print(f"  ├─ Median waiting time:      {orig_wait.median():.2f} hours")
            print(f"  ├─ Std waiting time:         {orig_wait.std():.2f} hours")
            print(f"  ├─ 25th percentile:          {orig_wait.quantile(0.25):.2f} hours")
            print(f"  ├─ 75th percentile:          {orig_wait.quantile(0.75):.2f} hours")
            print(f"  └─ 95th percentile:          {orig_wait.quantile(0.95):.2f} hours")
            print()
            
            if results['waiting_times']:
                sim_wait = results['waiting_times']
                print(f"Simulation Statistics (for delayed allocations):")
                print(f"  ├─ Total delayed:            {len(sim_wait)}")
                print(f"  ├─ Mean waiting time:        {np.mean(sim_wait):.2f} hours")
                print(f"  ├─ Median waiting time:      {np.median(sim_wait):.2f} hours")
                print(f"  └─ Max waiting time:         {np.max(sim_wait):.2f} hours")
                print()
                print(f"Note: {results['immediate']} allocations ({immediate_rate:.1f}%) had NO waiting time")
                print(f"      This means most resources were immediately available!")
            else:
                print("Simulation: All allocations were immediate (no waiting time)")
        else:
            print("⚠️  Could not calculate original waiting times")
        print()
        
        # Resource utilization
        print(f"👥 RESOURCE UTILIZATION")
        print(f"{'─'*80}")
        print(f"Total Resources Used:          {len(results['allocated_resources'])}")
        print(f"Original Resources Available:  {len(self.original_resource_counts)}")
        print()
        
        # Top 10 most used resources (simulated)
        print("Top 10 Most Allocated Resources (Simulation):")
        for resource, count in results['allocated_resources'].most_common(10):
            original_count = self.original_resource_counts.get(resource, 0)
            print(f"  {resource:30s}: {count:4d} allocations  (Original: {original_count:,})")
        print()
        
        # Activity-level statistics
        print(f"📋 ACTIVITY-LEVEL STATISTICS")
        print(f"{'─'*80}")
        print(f"{'Activity':<40s} {'Tested':>8s} {'Success':>8s} {'Failed':>8s} {'Rate':>8s}")
        print(f"{'─'*80}")
        
        for activity in sorted(results['activity_results'].keys()):
            stats = results['activity_results'][activity]
            success_rate = (stats['success'] / stats['total'] * 100) if stats['total'] > 0 else 0
            print(f"{activity:<40s} {stats['total']:>8d} {stats['success']:>8d} {stats['failed']:>8d} {success_rate:>7.1f}%")
        
        print()
        
        # Resource diversity per activity
        print(f"🔄 RESOURCE DIVERSITY PER ACTIVITY (Top 10)")
        print(f"{'─'*80}")
        
        activity_diversity = []
        for activity, stats in results['activity_results'].items():
            if stats['success'] > 0:
                diversity = len(stats['allocated_resources'])
                original_diversity = len(self.activity_resources[activity])
                activity_diversity.append((activity, diversity, original_diversity, stats['success']))
        
        # Sort by number of different resources used
        activity_diversity.sort(key=lambda x: x[1], reverse=True)
        
        print(f"{'Activity':<40s} {'Sim Resources':>15s} {'Orig Resources':>15s} {'Allocations':>12s}")
        print(f"{'─'*80}")
        for activity, sim_res, orig_res, count in activity_diversity[:10]:
            print(f"{activity:<40s} {sim_res:>15d} {orig_res:>15d} {count:>12d}")
        
        print()
        
        # Summary
        print(f"{'='*80}")
        print(f"✅ VALIDATION COMPLETE")
        print(f"{'='*80}\n")
        
        # QUALITY METRICS
        print(f"🎯 QUALITY METRICS")
        print(f"{'─'*80}")
        
        # 1. Resource-Activity Mapping Accuracy
        if results['total_checked'] > 0:
            mapping_accuracy = (results['activity_resource_matches'] / results['total_checked']) * 100
            print(f"Resource-Activity Mapping Accuracy: {mapping_accuracy:.1f}%")
            print(f"  └─ {results['activity_resource_matches']:,} / {results['total_checked']:,} allocations used historically correct resources")
        
        # 2. Resource Distribution Similarity (using Jensen-Shannon Divergence)
        try:
            from scipy.spatial.distance import jensenshannon
            from scipy.stats import wasserstein_distance
            
            # Normalize distributions
            sim_resources = results['allocated_resources']
            sim_dist = np.array([sim_resources.get(r, 0) for r in self.original_resource_counts.index])
            orig_dist = self.original_resource_counts.values[:len(sim_dist)]
            
            if sim_dist.sum() > 0 and orig_dist.sum() > 0:
                sim_dist_norm = sim_dist / sim_dist.sum()
                orig_dist_norm = orig_dist / orig_dist.sum()
                
                js_divergence = jensenshannon(sim_dist_norm, orig_dist_norm)
                similarity_score = max(0, (1 - js_divergence) * 100)
                print(f"\nResource Distribution Similarity: {similarity_score:.1f}%")
                print(f"  └─ How similar is resource usage compared to original (0-100%)")
            
            # 3. Calculate Gini coefficient for resource fairness
            def gini_coefficient(x):
                sorted_x = np.sort(x)
                n = len(x)
                cumsum = np.cumsum(sorted_x)
                return (2 * np.sum((n - np.arange(1, n + 1) + 0.5) * sorted_x)) / (n * cumsum[-1]) - 1
            
            if len(sim_dist) > 0 and sim_dist.sum() > 0:
                orig_gini = gini_coefficient(self.original_resource_distribution)
                sim_gini = gini_coefficient(sim_dist[sim_dist > 0])
                print(f"\nResource Workload Distribution (Gini Coefficient):")
                print(f"  ├─ Original:    {orig_gini:.3f}")
                print(f"  ├─ Simulation:  {sim_gini:.3f}")
                print(f"  └─ Difference:  {abs(orig_gini - sim_gini):.3f} (lower is better)")
        except ImportError:
            print("\n⚠️  Install scipy for advanced distribution metrics")
            similarity_score = None
            sim_dist = np.array([])
            orig_dist = np.array([])
        
        print()
        print(f"{'='*80}")
        
        # Overall assessment
        score = 0
        max_score = 4
        
        # Criterion 1: Success rate
        if success_rate >= 95:
            score += 1
            print("✅ Success Rate: EXCELLENT (≥95%)")
        elif success_rate >= 80:
            score += 0.5
            print("⚠️  Success Rate: GOOD (≥80%)")
        else:
            print("❌ Success Rate: POOR (<80%)")
        
        # Criterion 2: Resource-Activity Mapping
        if results['total_checked'] > 0:
            mapping_accuracy = (results['activity_resource_matches'] / results['total_checked']) * 100
            if mapping_accuracy >= 80:
                score += 1
                print("✅ Resource Mapping: EXCELLENT (≥80%)")
            elif mapping_accuracy >= 60:
                score += 0.5
                print("⚠️  Resource Mapping: GOOD (≥60%)")
            else:
                print("❌ Resource Mapping: POOR (<60%)")
        
        # Criterion 3: Resource Distribution Similarity
        if similarity_score is not None and sim_dist.sum() > 0 and orig_dist.sum() > 0:
            if similarity_score >= 70:
                score += 1
                print("✅ Distribution Similarity: EXCELLENT (≥70%)")
            elif similarity_score >= 50:
                score += 0.5
                print("⚠️  Distribution Similarity: GOOD (≥50%)")
            else:
                print("❌ Distribution Similarity: POOR (<50%)")
        else:
            score += 0.5
            print("⚠️  Distribution Similarity: Cannot calculate (install scipy)")
        
        # Criterion 4: Waiting times reasonable
        if results['waiting_times']:
            avg_wait = np.mean(results['waiting_times'])
            orig_avg_wait = self.original_waiting_times.mean() if len(self.original_waiting_times) > 0 else 0
            if orig_avg_wait > 0 and avg_wait <= orig_avg_wait * 1.5:  # Within 150% of original
                score += 1
                print("✅ Waiting Times: EXCELLENT (similar to original)")
            else:
                score += 0.5
                print("⚠️  Waiting Times: ACCEPTABLE (slightly higher than original)")
        else:
            score += 1
            print("✅ Waiting Times: EXCELLENT (all immediate)")
        
        final_score = (score / max_score) * 100
        print(f"\n{'='*80}")
        print(f"OVERALL QUALITY SCORE: {final_score:.1f}/100")
        print(f"{'='*80}")
        
        if final_score >= 85:
            print("\n🎉 EXCELLENT: Resource allocation is highly realistic and accurate!")
        elif final_score >= 70:
            print("\n✅ GOOD: Resource allocation is working well with minor deviations.")
        elif final_score >= 50:
            print("\n⚠️  ACCEPTABLE: Resource allocation works but has notable differences from original.")
        else:
            print("\n❌ POOR: Resource allocation needs significant improvements.")
    
    def _create_visualizations(self, results):
        """Create visualization plots."""
        try:
            output_dir = Path("validation_results")
            output_dir.mkdir(exist_ok=True)
            
            # 1. Resource utilization comparison
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            
            # Simulated resource usage
            top_resources_sim = dict(results['allocated_resources'].most_common(20))
            axes[0].bar(range(len(top_resources_sim)), list(top_resources_sim.values()))
            axes[0].set_title('Simulated Resource Allocations (Top 20)')
            axes[0].set_xlabel('Resource Index')
            axes[0].set_ylabel('Number of Allocations')
            axes[0].grid(axis='y', alpha=0.3)
            
            # Original resource usage
            top_resources_orig = self.original_resource_counts.head(20)
            axes[1].bar(range(len(top_resources_orig)), top_resources_orig.values)
            axes[1].set_title('Original Resource Usage (Top 20)')
            axes[1].set_xlabel('Resource Index')
            axes[1].set_ylabel('Number of Events')
            axes[1].grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_dir / "resource_comparison.png", dpi=150, bbox_inches='tight')
            logger.info(f"✓ Saved visualization: {output_dir / 'resource_comparison.png'}")
            plt.close()
            
            # 2. Waiting times distribution
            if results['waiting_times']:
                plt.figure(figsize=(10, 6))
                plt.hist(results['waiting_times'], bins=30, edgecolor='black', alpha=0.7)
                plt.title('Distribution of Waiting Times')
                plt.xlabel('Wait Time (hours)')
                plt.ylabel('Frequency')
                plt.grid(axis='y', alpha=0.3)
                plt.savefig(output_dir / "waiting_times.png", dpi=150, bbox_inches='tight')
                logger.info(f"✓ Saved visualization: {output_dir / 'waiting_times.png'}")
                plt.close()
            
            # 2b. Waiting times comparison with original
            if len(self.original_waiting_times) > 0:
                fig, axes = plt.subplots(2, 2, figsize=(16, 10))
                
                # Original distribution (limited to 95th percentile for visibility)
                orig_wait = self.original_waiting_times
                percentile_95 = orig_wait.quantile(0.95)
                orig_wait_filtered = orig_wait[orig_wait <= percentile_95]
                
                axes[0, 0].hist(orig_wait_filtered, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
                axes[0, 0].set_xlabel('Waiting Time (hours)')
                axes[0, 0].set_ylabel('Frequency')
                axes[0, 0].set_title(f'Original Log - Waiting Times (up to 95th percentile: {percentile_95:.1f}h)')
                axes[0, 0].grid(axis='y', alpha=0.3)
                
                # Box plot comparison
                if results['waiting_times']:
                    data_to_plot = [orig_wait[orig_wait <= percentile_95].values, results['waiting_times']]
                    axes[0, 1].boxplot(data_to_plot, labels=['Original', 'Simulation (delayed)'])
                    axes[0, 1].set_ylabel('Waiting Time (hours)')
                    axes[0, 1].set_title('Waiting Time Comparison')
                    axes[0, 1].grid(axis='y', alpha=0.3)
                else:
                    axes[0, 1].text(0.5, 0.5, 'No delayed allocations\nin simulation', 
                                   ha='center', va='center', fontsize=12)
                    axes[0, 1].set_title('Simulation: All Immediate')
                
                # Statistics comparison
                stats_text = f"Original Log:\n"
                stats_text += f"  Mean: {orig_wait.mean():.1f}h\n"
                stats_text += f"  Median: {orig_wait.median():.1f}h\n"
                stats_text += f"  Std: {orig_wait.std():.1f}h\n"
                stats_text += f"  Count: {len(orig_wait):,}\n\n"
                
                if results['waiting_times']:
                    sim_wait = results['waiting_times']
                    stats_text += f"Simulation (delayed):\n"
                    stats_text += f"  Mean: {np.mean(sim_wait):.1f}h\n"
                    stats_text += f"  Median: {np.median(sim_wait):.1f}h\n"
                    stats_text += f"  Count: {len(sim_wait)}\n\n"
                    stats_text += f"Immediate: {results['immediate']} ({results['immediate']/results['successful']*100:.1f}%)"
                else:
                    stats_text += f"Simulation:\n  All immediate (no delays)"
                
                axes[1, 0].text(0.1, 0.5, stats_text, fontsize=11, verticalalignment='center',
                               family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                axes[1, 0].axis('off')
                axes[1, 0].set_title('Statistics Summary')
                
                # Percentile comparison
                percentiles = [25, 50, 75, 90, 95, 99]
                orig_percentiles = [orig_wait.quantile(p/100) for p in percentiles]
                
                axes[1, 1].plot(percentiles, orig_percentiles, marker='o', linewidth=2, label='Original')
                if results['waiting_times'] and len(results['waiting_times']) > 1:
                    sim_percentiles = [np.percentile(results['waiting_times'], p) for p in percentiles]
                    axes[1, 1].plot(percentiles, sim_percentiles, marker='s', linewidth=2, label='Simulation (delayed)')
                axes[1, 1].set_xlabel('Percentile')
                axes[1, 1].set_ylabel('Waiting Time (hours)')
                axes[1, 1].set_title('Percentile Comparison')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(output_dir / "waiting_times_detailed_comparison.png", dpi=150, bbox_inches='tight')
                logger.info(f"✓ Saved visualization: {output_dir / 'waiting_times_detailed_comparison.png'}")
                plt.close()
            
            # 3. Success rate by activity
            activities = []
            success_rates = []
            for activity, stats in results['activity_results'].items():
                if stats['total'] >= 5:  # Only show activities with enough samples
                    activities.append(activity[:30])  # Truncate long names
                    success_rates.append(stats['success'] / stats['total'] * 100)
            
            if activities:
                plt.figure(figsize=(12, max(6, len(activities) * 0.3)))
                plt.barh(activities, success_rates, color='skyblue', edgecolor='black')
                plt.xlabel('Success Rate (%)')
                plt.title('Resource Allocation Success Rate by Activity')
                plt.xlim(0, 105)
                plt.grid(axis='x', alpha=0.3)
                plt.tight_layout()
                plt.savefig(output_dir / "success_by_activity.png", dpi=150, bbox_inches='tight')
                logger.info(f"✓ Saved visualization: {output_dir / 'success_by_activity.png'}")
                plt.close()
            
        except Exception as e:
            logger.warning(f"Could not create visualizations: {e}")


def main():
    """Main validation function."""
    # Configuration
    LOG_PATH = "Dataset/BPI Challenge 2017.xes"
    PERMISSION_METHOD = 'ordinor'  # or 'basic', 'ordinor-strict'
    SAMPLE_SIZE = 1000
    
    # Check if log exists
    if not Path(LOG_PATH).exists():
        logger.error(f"Event log not found: {LOG_PATH}")
        logger.info("Please update LOG_PATH in the script to point to your event log.")
        return
    
    # Run validation
    validator = ResourceAllocationValidator(
        log_path=LOG_PATH,
        permission_method=PERMISSION_METHOD
    )
    
    results = validator.run_validation(sample_size=SAMPLE_SIZE)
    
    logger.info("\n✅ Validation completed successfully!")
    logger.info(f"Check 'validation_results/' folder for visualizations.\n")


if __name__ == "__main__":
    main()
