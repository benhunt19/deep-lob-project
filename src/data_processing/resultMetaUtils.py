import pandas as pd
import numpy as np
import json
from glob import glob
import os
import datetime

from src.core.constants import RESULTS_PATH, PROJECT_ROOT, ORDERBOOKS, ORDERFLOWS
from src.core.generalUtils import getWeightPathFromID, getResultPathFromID, gitAdd

# Local constant, also used in varios places across the project
RUN_ID = 'run_id'
DATETIME_COL = 'datetime'

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

def getFrameFromRun(ticker : str, sortMetric : str = 'accuracy', date : str = None, **kwargs) -> pd.DataFrame:
    """
    Description:
        Gets a filtered dataframe based on any kwargs, 
    Parameters:
        ticker (str): Ticker of stock to choose
        sortMetric (str): Which metric to use as the 'best' and to sort the data by
        date (str): yyyy-mm-dd date to filter the metrics from
        kwargs (key=vale): Kwargs to attempt to filter the dataframe by, not case sensitive
    """
    df = frameFromResultMeta()
    # Rename columns drop meta.
    df.columns = [col.replace("meta.", "") for col in df.columns]
    assert len(df.columns) == len(set(df.columns)), "Column names are not unique after renaming (stripping prefixes, eg. meta.xxx)"
    
    # First choose the stock
    df = df[df['ticker'] == ticker]
    
    assert len(df) > 0, f"No results found for {ticker} before filtering, please check"
    
    # Filter on keyword arguments
    for key, value in kwargs.items():
        if key.lower() not in list(df.columns.map(lambda x : x.lower())):
            print("key not found in meta, please see metaConstants.py")
            continue
        df = df.query(f"{key} == @value")
        
    if date is not None and DATETIME_COL in df.columns:
        df = df.dropna(subset=[DATETIME_COL])
        df = df[df[DATETIME_COL].str.startswith(date)]
        
    sort_cols = [col for col in df.columns if sortMetric in col]
    if sort_cols:
        df = df.sort_values(by=sort_cols[0], ascending=False)
    return df

def getBestIDs(ticker : str, sortMetric : str = 'accuracy', date : str = None, **kwargs) -> str:
    """
    Description:
        Get the best run IDs based on results, filtered by keyword arguments
    Parameters:
        ticker (str): Ticker of stock to choose
        sortMetric (str): Which metric to use as the 'best' and to sort the data by
        date (str): yyyy-mm-dd date to filter the metrics from
        kwargs (key=vale): Kwargs to attempt to filter the dataframe by, not case sensitive
    """
    df = getFrameFromRun(ticker, sortMetric=sortMetric, **kwargs)
    return list(df[RUN_ID])

def stageBestRunWeights():
    """
    Description:
        Stage best run weights from training
    """
    tickers = frameFromResultMeta()['meta.ticker'].unique()
    paths = []
    for ticker in tickers:
        ids = ids = getBestIDs(ticker, representation=ORDERBOOKS, rowLim=1000000, lookForwardHorizon=10)
        if len(ids) > 0:
            paths += getWeightPathFromID(ids[0])
    # Stage
    for path in paths:
        gitAdd(path)

def stageResultsByDate(date : str = None):
    """
    Description:
        Stage results by date, default todays date
    Parameters:
        date (str): yyyy-mm-dd date to filter the metrics from
    """
    if date is None:
        date = datetime.date.today().strftime("%Y-%m-%d")
    
    tickers = frameFromResultMeta()['meta.ticker'].unique()
    ids = []
    for ticker in tickers:
        df = getFrameFromRun(ticker=ticker,date=date)
        ids += list(df[RUN_ID])
    paths = []
    for id in ids:
        paths += getResultPathFromID(id)
        
    for path in paths:
        gitAdd(path)

def deleteRunsFromResults(runIDs : list, dryRun : bool = True):
    f"""
    Description:
        Delete result runs from {RESULTS_PATH} folder
    Parameters:

    """
    for runID in runIDs:
        path = glob(f"{PROJECT_ROOT}/{RESULTS_PATH}/*{runID}.json")[0]
        if os.path.exists(path):
            if not dryRun:
                os.remove(path)
                print(f"{runID} File deleted.")
            else:
                print(f"dryRun: True, {runID} would be deleted.")
        else:
            print("File not found.")

def getLiquidityData():
    # average volume, top 10 levels
    processed_dir = os.path.join(PROJECT_ROOT, 'data', 'processed')
    tickers = [name for name in os.listdir(processed_dir) if os.path.isdir(os.path.join(processed_dir, name))]
    print(tickers)
    time_col = 0
    avg_diffs = {}
    liquidity_stats = {}
    for ticker in tickers:
        ob_dir = os.path.join(processed_dir, ticker, 'orderbooks', 'scaled')
        if os.path.isdir(ob_dir):
            csv_files = [f for f in os.listdir(ob_dir) if f.endswith('.csv')]
            if csv_files:
                first_csv = sorted(csv_files)[0]
                csv_path = os.path.join(ob_dir, first_csv)
                df_ob = pd.read_csv(csv_path)
                # Compute the average difference between consecutive values in column 1 (time)
                col1 = df_ob.iloc[:, time_col]
                avg_diff = col1.diff().abs().mean()
                print(f"Ticker: {ticker}, Average Update difference: {avg_diff}")
                avg_diffs[ticker] = avg_diff
        else:
            print(f"Directory does not exist: {ob_dir}")
    print(avg_diffs)
    return avg_diffs
        

if __name__ == "__main__":
    # stageResultsByDate()
    getLiquidityData()