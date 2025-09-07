import os
from datetime import datetime
import random
import string
from glob import glob
import subprocess
from omegaconf import OmegaConf, DictConfig
import json
import numpy as np
from pathlib import Path
import warnings

from src.core.constants import (
    PROJECT_ROOT,
    WEIGHTS_PATH,
    WEIGHTS_PRD_PATH,
    RESULTS_PATH,
    PROCESSED_DATA_PATH,
    SCALED,
    UNSCALED,
    ORDERBOOKS,
    ORDERFLOWS,
    NORMALISATION,
    ORDERVOL,
    ORDERFIXEDVOL,
    CATEGORICAL,
    REGRESSION,
    ALGO_RESULTS,
    ALGO_SLIPPAGE_RESULTS
)

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

def exportLocation(model, runName : str = ""):
    """
    Description:
        Single location for defining where model weights are stored
    Parameters:
        Model (Deep Learning model): The model to get the name from
    """
    filePath = f"{PROJECT_ROOT}/{WEIGHTS_PATH}/{model.name}"
    if not os.path.exists(filePath):
        os.makedirs(filePath)
    return f"{filePath}/{model.name}_{runName}"

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

def processDataFileNaming(ticker : str, scaling : bool, date: str, representation: str = ORDERBOOKS, extension: str = '.csv'):
    """
    Description:
        Single definition of processed data file handler and its location.
    Parameters:
        ticker (str): The name of the stock ticker.
        scaling (bool): Is the data scaled or unscaled.
        date (str): The date for the data file.
        representation (str): The LOB representation ('orderbooks', 'orderflows',...).
        extension (str): The file extension (default '.csv').
    Returns:
        tuple: (fileName, output_name) where fileName is the name of the file and output_name is the full path.
    """
    file_location = Path(processedDataLocation(ticker, scaling, representation=representation))
    
    # If file_location is not absolute, make it relative to your project root
    if not file_location.is_absolute():
        project_root = Path(__file__).parents[2]
        file_location = project_root / file_location
        
    # Compose the output file name
    fileName = f"{ticker}_{representation}_{date}{extension}"
    output_name = file_location / fileName
    
    # Create directory if it doesn't exist
    file_location.mkdir(parents=True, exist_ok=True)
    return fileName, output_name

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

def getWeightPathFromID(run_id : str, extension : str = None) -> list:
    """
    Description:
        Find weights path based on a run ID
    Parameters:
        run_id (str): The ID to find from the weights
    """
    # Use glob to match any file containing the run_id in its name, in any subdirectory of WEIGHTS_PATH
    if extension is None:
        pattern = f"{PROJECT_ROOT}/{WEIGHTS_PATH}/**/*{run_id}*"
    else:
        pattern = f"{PROJECT_ROOT}/{WEIGHTS_PATH}/**/*{run_id}.{extension}"
    return glob(pattern)

def getResultPathFromID(run_id : str) -> list:
    """
    Description:
        Find results json path based on a run ID
    Parameters:
        run_id (str): The ID to find from the weights
    """
    # Use glob to match any file containing the run_id in its name, in any subdirectory of WEIGHTS_PATH
    pattern = f"{PROJECT_ROOT}/{RESULTS_PATH}/**/*{run_id}*.json"
    return glob(pattern, recursive=True)

def getMetaFromRunID(run_id : str) -> json:
    """
    Description:
        Get JSON meta from a run_id
    """
    weights_results_paths = getResultPathFromID(run_id=run_id)
    print(f'results_path: {weights_results_paths}')
    
    if len(weights_results_paths) == 0:
        warnings.warn(f'Not found meta for existing weights run_id ({weights_results_paths})')
        return {}
    
    if len(weights_results_paths) > 1:
        warnings.warn(f"Multiple metas found for one run_id {run_id}, selecting {weights_results_paths[0]}")
    
    weights_results_path = weights_results_paths[0]
    # Read and return the JSON file
    with open(weights_results_path, 'r') as f:
        j = json.load(f)
        return j['meta']
        

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
    
def resultsLocation(run_id : str, representation : str, ticker : str) -> str:
    """
    Description:
        Single function for defining the results location
    Parameters:
        run_id (str): The id of the run
        representation (str): The representation of the data
        ticker (str): The ticker of the stock run for
    """
    # Save resultsStore as JSON
    date_str = datetime.now().strftime("%Y-%m-%d")
    folder_location = f"{PROJECT_ROOT}/{RESULTS_PATH}/{date_str}/{representation}/{ticker}"
    file_name = f"results_{run_id}.json"
    results_path = f"{folder_location}/{file_name}"
    
    # Create directory if it doesn't exist
    os.makedirs(folder_location, exist_ok=True)
    return results_path

def getPrdWeightsPath(ticker : str, representation : str, lookForwardHorizon : int, labelType : str, extension : str = '.h5'):
    f"""
    Description:
        Get production weights path
    Parameters:
        ticker (str): Stock ticker
        representation (str): Data representation {ORDERBOOKS, ORDERFLOWS, ORDERVOL, ORDERFIXEDVOL}
        lookForwardHorizon (int): Look forward horizon from model 
        labelType (str): The label type the model is {CATEGORICAL, REGRESSION}
    """
    folder = f"{PROJECT_ROOT}/{WEIGHTS_PRD_PATH}/{ticker}/{representation}/{lookForwardHorizon}"
    
    # Check for Categorical or Regression, if extended, change to get name from model class
    file_pattern = f"deepLOB_*{extension}" if labelType == CATEGORICAL else f"deepLOBREG_*{extension}"
    files = glob(f"{folder}/{file_pattern}")
    assert len(files) > 0, f"Production weight not found for specific parameters: {ticker, representation, lookForwardHorizon, labelType, extension}"
    return files[0]

def saveAlgoDictLocation(ticker : str, modelName : str, horizon : int, date : str, signalPercentage : int, slippage : bool = False):
    return f"{PROJECT_ROOT}/{ALGO_SLIPPAGE_RESULTS if slippage else ALGO_RESULTS}/{ticker}/{date}/{horizon}/{signalPercentage}/{modelName}"