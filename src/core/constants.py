import os

# File Location Constants
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

# Results location
RESULTS_PATH = 'results'

# Lobster Constants
LOBSTER_PRICE_SCALE_FACTOR = 10_000

# LOB Simulator Path
LOB_SIMULATOR_PATH = r"C:\Users\benhu\UCL\Term 3\HSBC\cpp-lob-simulator\build\Release"

# Model run types
TEST = "TEST"
TRAIN = "TRAIN"
VALIDATE = "VALIDATE"

# Model Label Modes
AUTO = 'auto'

# Global logger name
GLOBAL_LOGGER = "global_logger"

if __name__ == "__main__":
    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("RAW_DATA_PATH:", RAW_DATA_PATH)
    print("PROCESSED_DATA_PATH:", PROCESSED_DATA_PATH)
    print("DATA_PROCESS_LOGS:", DATA_PROCESS_LOGS)
    print("WEIGHTS_PATH:", WEIGHTS_PATH)
    print("LOBSTER_PRICE_SCALE_FACTOR:", LOBSTER_PRICE_SCALE_FACTOR)
    print("LOB_SIMULATOR_PATH:", LOB_SIMULATOR_PATH)
    print("SCALED:", SCALED)
    print("UNSCALED:", UNSCALED)
    print("TEST:", TEST)
    print("TRAIN:", TRAIN)
    print("VALIDATE:", VALIDATE)
    print("AUTO:", AUTO)
    print("GLOBAL_LOGGER:", GLOBAL_LOGGER)