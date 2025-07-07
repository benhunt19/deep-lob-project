import pandas as pd
import numpy as np
import json
from glob import glob

from src.core.constants import RESULTS_PATH, PROJECT_ROOT, ORDERBOOKS, ORDERFLOWS

# Local constant, also used in varios places across the project
RUN_ID = 'run_id'

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
        Add results to a single dataframe and return full frame
    """
    paths = getResultPaths()
    df = pd.DataFrame()
    for path in paths:
        with open(path, "r") as f:
            data = json.load(f)
            df = pd.concat([df, pd.json_normalize(data)], ignore_index=True)
    return df

def getFrameFromRun(ticker : str, sortMetric : str = 'accuracy', **kwargs) -> pd.DataFrame:
    """
    Description:
        Gets a filtered dataframe based on any kwargs, 
    Parameters:
        ticker (str): Ticker of stock to choose
        sortMetric (str): Which metric to use as the 'best' and to sort the data by
        kwargs (key=vale): Kwargs to attempt to filter the dataframe by, not case sensitive
    """
    df = frameFromResultMeta()
    # Rename columns to use only the last part after the dot
    df.columns = [col.split(".")[-1] for col in df.columns]
    assert len(df.columns) == len(set(df.columns)), "Column names are not unique after renaming (stripping prefixes, eg. meta.xxx)"
    
    # First chose the stock
    df = df[df['ticker'] == ticker]
    
    assert len(df) > 0, f"No results found for {ticker} before filtering, please check"
    
    # Filter on keyword arguments
    for key, value in kwargs.items():
        if key.lower() not in list(df.columns.map(lambda x : x.lower())):
            print("key not found in meta, please see metaConstants.py")
            continue
        df = df.query(f"{key} == @value")
    
    df = df.sort_values(sortMetric, ascending=False)
    return df

def getBestIDs(ticker : str, sortMetric : str = 'accuracy', **kwargs) -> str:
    """
    Description:
        Get the best run IDs based on results, filtered by keyword arguments
    Parameters:
        ticker (str): Ticker of stock to choose
        sortMetric (str): Which metric to use as the 'best' and to sort the data by
        kwargs (key=vale): Kwargs to attempt to filter the dataframe by, not case sensitive
    """
    df = getFrameFromRun(ticker, sortMetric=sortMetric, **kwargs)
    return list(df[RUN_ID])

if __name__ == "__main__":
    df = getBestIDs('AMZN', representation=ORDERFLOWS, rowLim=1000000, lookForwardHorizon=10)
    print(df)