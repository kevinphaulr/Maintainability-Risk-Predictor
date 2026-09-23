import logging
import sys
from pathlib import Path

# Setup logging configuration
LOG_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d] - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

logger = logging.getLogger("maintainability_risk_predictor")
logger.setLevel(logging.INFO)

# Prevent duplicate handlers
if not logger.handlers:
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

def get_logger(module_name: str) -> logging.Logger:
    """Returns a child logger inheriting the base configuration."""
    return logger.getChild(module_name)
