import os
from datetime import datetime
import random
import string
from glob import glob

from src.core.constants import PROJECT_ROOT, WEIGHTS_PATH, PROCESSED_DATA_PATH, SCALED, UNSCALED, ORDERBOOKS, ORDERFLOWS

def weightLocation(model, runName : str = ""):
    """
    Description:
        Single location for defining where model weights are stored
    Parameters:
        Model (Deep Learning model): The model to get the name from
    """
    filePath = f"{PROJECT_ROOT}/{WEIGHTS_PATH}/{model.name}"
    if not os.path.exists(filePath):
        os.makedirs(filePath)
    return f"{filePath}/{model.name}_{runName}.{model.weightsFileFormat}"

def processedDataLocation(ticker : str, scaling : bool, representation: str = ORDERBOOKS):
    """
    Description:
        Single definion of processed data and the location that its stored
    Parameters:
        ticker (str): The name of the stock ticker
        scaling (bool): Is the data scaled or unnscaled
        representation (str): The LOB representation ('orderbooks', 'orderflows',...)
    """
    scaled_unscaled = SCALED if scaling else UNSCALED
    return f"{PROJECT_ROOT}/{PROCESSED_DATA_PATH}/{ticker}/{representation}/{scaled_unscaled}"

def runID(length=8):
    """
    Description:
        Create a run ID for each run on a model within the test / tran farmeworks
    """
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

def nameModelRun(runID : str):
    """
    Description:
        Name generic model run with a consistent name
    Parameters:
        type (str): The type of run (TRAIN, TEST, VALIDATION)
    """
    return f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{runID}"

def getWeightPathFromID(run_id : str) -> list:
    """
    Description:
        Find weights path based on a run ID
    Parameters:
        run_id (str): The ID to find from the weights
    """
    # Use glob to match any file containing the run_id in its name, in any subdirectory of WEIGHTS_PATH
    pattern = f"{PROJECT_ROOT}/{WEIGHTS_PATH}/**/*{run_id}*"
    return glob(pattern)