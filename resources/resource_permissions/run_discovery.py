
import logging
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from resources.resource_allocation import ResourceAllocator

def run_discovery():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    # Paths
    # Assuming run from project root or resources/resource_permissions
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    log_path = os.path.join(project_root, "eventlog.xes.gz")
    cache_path = os.path.join(os.path.dirname(__file__), "ordinor_model.pkl")

    if not os.path.exists(log_path):
        logger.error(f"Event log not found at {log_path}")
        return

    logger.info(f"Starting discovery using log: {log_path}")
    logger.info(f"Model will be saved to: {cache_path}")

    # Initialize allocator to trigger discovery
    # We use a sample for quick verification if the user wants, but for full discovery we should use full log
    # The previous instruction said "run the discovery", implying the full one.
    
    try:
        allocator = ResourceAllocator(
            log_path=log_path,
            permission_method='ordinor',
            n_resource_clusters=10, # As per report
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
    run_discovery()
