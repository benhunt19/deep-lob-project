import os
from datetime import datetime
import random
import string

from src.core.constants import PROJECT_ROOT, WEIGHTS_PATH, PROCESSED_DATA_PATH, SCALED, UNSCALED

def weightLocation(model):
    """
    Description:
        Single location for defining where model weights are stored
    Parameters:
        Model (Deep Learning model): The model to get the name from
    """
    filePath = f"{PROJECT_ROOT}/{WEIGHTS_PATH}/{model.name}"
    if not os.path.exists(filePath):
        os.makedirs(filePath)
    return f"{filePath}/{model.name}.{model.weightsFileFormat}"

def processedDataLocation(ticker : str, scaling : bool):
    """
    Description:
        Single definion of processed data and the location that its stored
    Parameters:
        ticker (str): The name of the stock ticker
        scaling (bool): Is the data scaled or unnscaled
    """
    scaled_unscaled = SCALED if scaling else UNSCALED
    return f"{PROJECT_ROOT}/{PROCESSED_DATA_PATH}/{ticker}/{scaled_unscaled}"

def runID(length=8):
    """
    Description:
        Create a run ID for each run on a model within the test / tran farmeworks
    """
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

def nameModelRun(type : str, runID : str):
    """
    Description:
        Name generic model run with a consistent name
    Parameters:
        type (str): The type of run (TRAIN, TEST, VALIDATION)
    """
    return f"{type}_{datetime.now().strftime("%Y%m%d_%H%M%S")}_{runID}"