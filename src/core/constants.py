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
NORMALISATION = 'normalisation'

# Orderbook representations
ORDERBOOKS = 'orderbooks'
ORDERFLOWS = 'orderflows'
ORDERVOL = 'ordervol'
ORDERFIXEDVOL = 'orderfixedvol'

# Weights location
WEIGHTS_PATH = 'weights'

# Results location
RESULTS_PATH = 'results'
RESULTS_CSVS = 'results/csv_outputs'

# Lobster Constants
LOBSTER_PRICE_SCALE_FACTOR = 10_000

# LOB Simulator Path
LOB_SIMULATOR_PATH = r"C:\Users\benhu\UCL\Term 3\HSBC\cpp-lob-simulator\build\Release"

# Simulator constants
MEAN_ROW = 0
STD_DEV_ROW = 1

# Process steps
TEST = "TEST"
TRAIN = "TRAIN"
PROCESS_DATA = "PROCESS_DATA"

# Model Label Modes, also used for anything AUTOMATIC
AUTO = 'auto'

# Global logger name
GLOBAL_LOGGER = "global_logger"

# Hydra config path
HYDRA_CONFIG_PATH = "config"

# Label Type, regression or category
REGRESSION = "REGRESSION"
CATEGORICAL = "CATEGORICAL"

# Numpy saved constants
NUMPY_EXTENSION = '.npz'
NUMPY_X_KEY = 'x'