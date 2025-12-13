"""
OrdinoR Model Discovery Script.

Usage (from resources directory):
    # FullRecall mode (fast, recommended for simulation):
    python resource_permissions/run_discovery.py --mode full_recall

    # OverallScore mode (slow, precision-optimized):
    python resource_permissions/run_discovery.py --mode overall_score
"""
import argparse
import logging
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from resources.resource_allocation import ResourceAllocator


def run_discovery(mode: str = 'full_recall', n_clusters: int = 10):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    # Paths
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    log_path = os.path.join(project_root, "eventlog.xes.gz")

    # Different cache files for different modes
    if mode == 'full_recall':
        cache_path = os.path.join(os.path.dirname(__file__), "ordinor_fullrecall.pkl")
        permission_method = 'ordinor'
    else:
        cache_path = os.path.join(os.path.dirname(__file__), "ordinor_overallscore.pkl")
        permission_method = 'ordinor-strict'

    if not os.path.exists(log_path):
        logger.error(f"Event log not found at {log_path}")
        return

    logger.info(f"Starting discovery (mode={mode}, clusters={n_clusters})")
    logger.info(f"Log: {log_path}")
    logger.info(f"Cache: {cache_path}")

    try:
        allocator = ResourceAllocator(
            log_path=log_path,
            permission_method=permission_method,
            n_resource_clusters=n_clusters,
            cache_path=cache_path
        )
        logger.info("Discovery completed successfully!")

        # Verify
        logger.info("Verifying model...")
        stats = allocator.permissions.get_coverage_stats()
        logger.info(f"Model Coverage: {stats}")

    except Exception as e:
        logger.error(f"Discovery failed: {e}", exc_info=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Discover OrdinoR organizational model")
    parser.add_argument(
        "--mode",
        choices=['full_recall', 'overall_score'],
        default='full_recall',
        help="Profiling mode: 'full_recall' (fast, for simulation) or 'overall_score' (slow, precision)"
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=10,
        help="Number of resource clusters (default: 10)"
    )
    args = parser.parse_args()

    run_discovery(mode=args.mode, n_clusters=args.n_clusters)
