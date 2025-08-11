import asyncio
from torch import tensor
import torch
import time
import numpy as np
import pandas as pd
import polars as pl
from dataclasses import dataclass, field

from src.core.constants import CATEGORICAL, REGRESSION, AUTO
from src.loaders.dataLoader import CustomDataLoader

@dataclass
class RunGlobals:
    data: object = field(default_factory=list)  # creates a new list on each init
    cooloff: int = 200
    cooloffCounter: int = 0
    horizonCounter: int = 0
    index: int = 0

@dataclass
class SimPrediction:
    """
    Data:
        index (int): Index of event from base
        prediction (int): 0 = down, 1 = neutral
    """
    index: int
    prediction: int
    timeTaken : int
    labelType : str = CATEGORICAL

def reviewPredictions(
    predictions : list[SimPrediction],
    fileName : str,
    lookForwardHorizon : int,
    labelType : str, 
    ) -> pd.DataFrame:
    f"""
    Description:
        Review model predictions
    Parameters:
        predictions list[dict]: List of predictions eg [SimPrediction(index, prediction, timetaken, labelType)]
        fileName (str): File name to review timings from
        lookForwardHorizon (int): Horizon to look forward from
        labelType (str): {REGRESSION} or {CATEGORICAL}
    """
    
    # Get file from fileName
    orderBook = pl.read_csv(fileName, has_header=False).to_numpy()
    timings = orderBook[:, 0]
    print(timings)
    
    results = []
    for prediction in predictions:
        # process each prediction
        # Find the number of updates that occur during the time taken for this prediction
        start_time = timings[prediction.index]
        end_time = start_time + prediction.timeTaken
        # Count how many updates (rows) have a timestamp between start_time (exclusive) and end_time (inclusive)
        updates_during_timetaken = np.sum((timings > start_time) & (timings <= end_time))
        print(f"Prediction at index {prediction.index}: {updates_during_timetaken} updates during timeTaken={prediction.timeTaken}")
        results.append({
            "index": prediction.index,
            "prediction": prediction.prediction,
            "timeTaken": prediction.timeTaken,
            "labelType": prediction.labelType,
            "updatesDuringTimeTaken": updates_during_timetaken
        })
    
    df = pd.DataFrame(results)
    return df