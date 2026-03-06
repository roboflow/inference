from contextlib import contextmanager
from loguru import logger


@contextmanager
def json_logging(filepath: str, module_filter: str = "lidra.metrics.analysis.v3"):
    """Context manager for JSON logging to file.

    Args:
        filepath: Path to JSON log file. If None, no file logging is added.
        module_filter: Only log messages from modules containing this string.
    """
    logger.info(f"Opening JSON log file: {filepath}")

    # Add JSON file handler with serialization
    handler_id = logger.add(
        filepath,
        serialize=True,  # Output as JSON
        filter=lambda record: module_filter in record["name"],  # Only our module
        mode="w",  # Overwrite existing file
    )
    try:
        yield
    finally:
        # Remove the handler when done
        logger.remove(handler_id)
        logger.info(f"Closed JSON log file: {filepath}")
