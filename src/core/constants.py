import os

# File Locatin Constants
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Data locations
RAW_DATA_PATH = 'data/raw/'
RAW_ARCIVE_DATA_PATH = 'data/raw/archive'
PROCESSED_DATA_PATH = 'data/processed'
DEMO_DATA_PATH = 'data/demo'
DATA_PROCESS_LOGS = 'data/logs'

# Data Types
SCALED = 'scaled'
UNSCALED = 'unscaled'

# Weights location
WEIGHTS_PATH = 'weights'

# Lobster Constants
LOBSTER_PRICE_SCALE_FACTOR = 10_000

# LOB Simulator Path
LOB_SIMULATOR_PATH = r"C:\Users\benhu\UCL\Term 3\HSBC\cpp-lob-simulator\build\Release"