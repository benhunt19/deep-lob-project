import os
from datetime import datetime
import random
import string
from glob import glob
import subprocess
from omegaconf import OmegaConf, DictConfig
import json
import numpy as np

from src.core.constants import PROJECT_ROOT, WEIGHTS_PATH, RESULTS_PATH, PROCESSED_DATA_PATH, SCALED, UNSCALED, ORDERBOOKS, ORDERFLOWS, NORMALISATION

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

def normalisationDataLocation(ticker : str, scaling : bool, representation: str = ORDERBOOKS):
    """
    Description:
        Get location of normalisation data, this is a subfolder from the process data location
    Parameters:
        ticker (str): The name of the stock ticker
        scaling (bool): Is the data scaled or unnscaled
        representation (str): The LOB representation ('orderbooks', 'orderflows',...)
    """
    return f"{processedDataLocation(ticker=ticker, scaling=scaling, representation=representation)}/{NORMALISATION}"

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

def getResultPathFromID(run_id : str) -> list:
    """
    Description:
        Find results json path based on a run ID
    Parameters:
        run_id (str): The ID to find from the weights
    """
    # Use glob to match any file containing the run_id in its name, in any subdirectory of WEIGHTS_PATH
    pattern = f"{PROJECT_ROOT}/{RESULTS_PATH}/*{run_id}.json"
    return glob(pattern)

def gitAdd(filePath : str):
    """
    Description:
        Stage a file path
    Parameters:
        filePath (str): The path of the file to stage
    """
    try:
        result = subprocess.run(["git", "add", filePath], check=True, capture_output=True, text=True)
        print(f"Staged: {filePath}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to stage file: {e.stderr}")
        
def makeJsonSerializable(obj):
    """
    Description
        Recursively convert objects to be JSON serializable.
        - Converts DictConfigs to dicts
        - Converts numpy types to Python native types
    Parameters:
        Object to convert
    """
    if isinstance(obj, DictConfig):
        obj = OmegaConf.to_container(obj, resolve=True)
    if isinstance(obj, dict):
        return {k: makeJsonSerializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [makeJsonSerializable(i) for i in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj