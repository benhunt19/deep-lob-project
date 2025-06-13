import os

# File Locatin Constants
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Data locations
RAW_DATA_PATH = 'data/raw/'
PROCESSED_DATA_PATH = 'data/processed'
DEMO_DATA_PATH = 'data/demo'
DATA_PROCESS_LOGS = 'data/logs'

# Lobster Constants
LOBSTER_PRICE_SCALE_FACTOR = 10_000