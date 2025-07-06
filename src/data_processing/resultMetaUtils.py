import pandas as pd
import numpy as np
import json
from glob import glob

from src.core.constants import RESULTS_PATH, PROJECT_ROOT

# Here we are going to define methods that we can use to get the data out of results

def getResultPaths() -> list:
    """
    Description:
        Get paths of all results from results folder
    """
    return glob(f"{PROJECT_ROOT}/{RESULTS_PATH}/*.json")

def frameFromResultMeta() -> pd.DataFrame:
    """
    Description:
        Add results to a single dataframe
    """
    paths = getResultPaths()
    df = pd.DataFrame()
    for path in paths:
        with open(path, "r") as f:
            data = json.load(f)
            df = pd.concat([df, pd.json_normalize(data)], ignore_index=True)
    return df

if __name__ == "__main__":
    frameFromResultMeta()