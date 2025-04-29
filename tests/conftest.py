"""
This file contains fixtures and configurations that are shared across multiple test files
pytest automatically detects and uses fixtures from this file.
"""

import pytest
import logging
import os
from datetime import datetime
from tests.fixtures.fixtures import load_data as load_data_fixture, run_r_ics as run_r_ics_fixture, run_py_ics as run_py_ics_fixture

# Create a logs directory if it doesn't exist
if not os.path.exists('logs'):
    os.makedirs('logs')

# Generate a log file name based on the current date and time
log_filename = datetime.now().strftime('logs/test_results_%Y-%m-%d_%H-%M-%S.log')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Ensure proper cleanup of logging handlers to avoid ValueError
def close_logging_handlers():
    for handler in logger.handlers:
        handler.close()
        logger.removeHandler(handler)

def pytest_runtest_logreport(report):
    if report.when == 'call':
        if report.failed:
            logger.error(f"Test {report.nodeid} failed: {report.longrepr}")
        elif report.passed:
            logger.info(f"Test {report.nodeid} passed")
        elif report.skipped:
            logger.warning(f"Test {report.nodeid} skipped: {report.longrepr}")

@pytest.fixture(scope="module")
def load_data():
    """
    Fixture to load datasets for testing.
    """
    return load_data()

@pytest.fixture(scope="module")
def run_r_ics():
    """
    Fixture to run ICS in R for testing.
    """
    return run_r_ics()

@pytest.fixture(scope="module")
def run_py_ics():
    """
    Fixture to run ICS in Python for testing.
    """
    return run_py_ics()