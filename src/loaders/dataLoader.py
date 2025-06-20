import pandas as pd
import polars as pl
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from glob import glob
from tqdm import tqdm
import torch

# import pyspark
# from pyspark.sql import SparkSession

from src.core.constants import PROJECT_ROOT, PROCESSED_DATA_PATH
from src.core.generalUtils import processedDataLocation

# Some data specific constants
DROP_COLUMNS = [0, -1, -2]     # Columns to drop from the processed dataframes

class CustomDataLoader:
    """
    Description:
        A Dataset to be used to test / train / validate models
    """
    def __init__(
        self, 
        ticker : str,                   # Ticker name
        scaled : bool,                  # scaled or unscaled
        horizon : int = 100,            # Horizon length
        maxFiles : int = None,          # Maximum number of files to concatenate into one file
        threshold: float = 10,          # Percentage change over a horizon for it to be positive or negative
        rowLim: int = None              # The row limit number for
    ):
        self.ticker = ticker
        self.scaled = scaled
        self.horizon = horizon
        self.threshold = threshold
        self.rowLim = rowLim
        self.maxFiles = maxFiles
        
        # Require in class
        self.fileLocations = None
        self.x = None
        self.y = None
        

    def getFileLocations(self, dataLocation : str = None) -> list[str]:
        """
        Description:
            Get file locations for ticker
        """
        # print("Getting file locations...")
        if dataLocation is None:
            dataLocation = processedDataLocation(self.ticker, self.scaled)
            
        self.fileLocations = glob(dataLocation + f"/{self.ticker}*.csv")
        
        if self.maxFiles is not None and len(self.fileLocations) > self.maxFiles:
            self.fileLocations = self.fileLocations[0 : self.maxFiles]
        
        return self.fileLocations

    def getDataFromFiles(self, fileLocations :list[str] = None, dropCols : list[int] = DROP_COLUMNS) -> pd.DataFrame:
        """
        Description:
            Get data from CSV files and stack them vertically (append rows)
        """
        all_data = []
        
        if fileLocations is None:
            fileLocations = self.fileLocations
            
        assert fileLocations is not None, "No file Locations provided, run self.getFileLocations()"
        
        # Get DataFrames from the CSVs
        print("Extracting data from files...")
        for csv in tqdm(fileLocations):
            df = pl.read_csv(csv, has_header=False).to_pandas()
            all_data.append(df)

        self.globalFrame = pd.concat(all_data, axis=0, ignore_index=True)
        if dropCols is not None:
            self.globalFrame.drop(self.globalFrame.columns[dropCols], axis=1, inplace=True)
            
        return self.globalFrame

    def dataFrameToFeatures(self) -> np.ndarray:
        """
        Description:
            Turn data into a model 'runnable' dataset, this is the self.x values (features) that are passed into the models
        Parameters:
            rowLim (int): number of rows to return
        """
        # Create the x values for model inputs
        
        self.rowLim = self.rowLim if self.rowLim is not None else len(self.globalFrame) - self.horizon

        arr = self.globalFrame.to_numpy()  # shape (total_rows, features)

        # Create sliding windows of shape (rowLim, horizon, features)
        windows = sliding_window_view(arr, window_shape=(self.horizon), axis=0)
        datasetX = windows[:self.rowLim, :, :]

        # Add the last dimension as requested (e.g., for channel dimension)
        self.x = np.expand_dims(datasetX, axis=-1)
        return self.x
    
    def dataFrameToLabelsRaw(self) -> np.ndarray:
        """
        Description:
            Finds the feature labels from the raw LOB data
        """
        # Create the y labels from the far
        
        # ASKp1, ASKs1, BIDp1, BIDs1, ...
        
        ask1Col = 0; bid1Col = 2
        
        self.rowLim  = self.rowLim if self.rowLim is not None else len(self.globalFrame) - self.horizon
            
        df = self.globalFrame.values  # Convert DataFrame to NumPy array for fast indexing
        horizon = self.horizon
        threshold = self.threshold
        rowLim = min(self.rowLim, len(df) - horizon)

        # Vectorized slicing or df columns
        startAsk = df[0:rowLim, ask1Col]
        endAsk   = df[horizon:horizon + rowLim, ask1Col]
        startBid = df[0:rowLim, bid1Col]
        endBid   = df[horizon:horizon + rowLim, bid1Col]

        startMid = (startAsk + startBid) / 2
        endMid   = (endAsk + endBid) / 2
        percentageDiff = (endMid - startMid) * 100 / startMid

        # Vectorized labeling
        down    = (percentageDiff < -threshold).astype(int)
        up      = (percentageDiff >  threshold).astype(int)
        neutral = ((down == 0) & (up == 0)).astype(int)

        # Stack the one-hot encoded labels
        self.y = np.stack([down, neutral, up], axis=1)
        
        return self.y

    def xyToTensor(self):
        """
        Description:
            Turn the features and labels into a pytorch tensor
        """
        assert self.x is not None, "Please ensure there are features, run self.dataFrameToFeatures()"
        assert self.y is not None, "Please ensure there are labels, run self.dataFrameToLabelsRaw()"
        
        print(f"X Shape before {self.x.shape}")
        print(f"Y Shape before {self.y.shape}")
        
        self.x = torch.tensor(self.x, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32)
        
        print(f"X Shape after {self.x.shape}")
        print(f"Y Shape after {self.y.shape}")
    
    def runFullProcessReturnXY(self, tensor : bool = False) -> tuple[np.ndarray, np.ndarray]:
        """
        Descrition:
            Run full process
        """
        self.getFileLocations()
        self.getDataFromFiles()
        self.dataFrameToFeatures()
        self.dataFrameToLabelsRaw()
        if tensor:
            self.xyToTensor()
        return self.x, self.y
        

if __name__ == "__main__":
    cdl = CustomDataLoader(
        ticker='TSLA',
        scaled=False,
        horizon=100,
        threshold=0.001,
        maxFiles=10,
        rowLim=None
    )
    x, y = cdl.runFullProcessReturnXY(tensor=True)
    print(x.shape)
    print(y.shape)