"""
Resources package for the Process Simulation Engine.

Provides:
- resource_permissions: Organizational model mining (OrdinoR)
- resource_availabilities: Working hours detection
- resource_allocation: Combined resource assignment

Logging is configured here for all resource-related modules.
"""
from .resource_allocation import ResourceAllocator
import logging
import os

# Configure logging for the resources package
def setup_logging(
    level: str = "INFO",
    log_file: str = None,
    format_string: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
):
    """
    Set up logging for the resources package.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file. If None, logs to console only.
        format_string: Log message format.
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(logging.Formatter(format_string))
    handlers.append(console_handler)
    
    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(logging.Formatter(format_string))
        handlers.append(file_handler)
    
    # Configure loggers for all resource submodules
    for module_name in [
        "resources",
        "resources.resource_permissions",
        "resources.resource_permissions.resource_permissions",
        "resources.resource_permissions.data_preparation",
        "resources.resource_availabilities",
        "resources.resource_allocation",
    ]:
        logger = logging.getLogger(module_name)
        logger.setLevel(log_level)
        logger.handlers = []  # Clear existing handlers
        for handler in handlers:
            logger.addHandler(handler)


# Auto-configure with defaults if LOG_LEVEL env var is set
if os.environ.get("LOG_LEVEL"):
    setup_logging(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        log_file=os.environ.get("LOG_FILE")
    )
