import pandas as pd
import polars as pl
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from glob import glob
from tqdm import tqdm
import torch
import scipy.stats
from math import floor

# import pyspark
# from pyspark.sql import SparkSession

from src.core.constants import AUTO, ORDERBOOKS, ORDERFLOWS
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
        ticker : str,                       # Ticker name
        scaling : bool,                     # True for scaled, False for unscaled, decides if we use the scaled or unscaled processed data
        horizon : int = 100,                # Horizon length looking backwards, essentially the window
        maxFiles : int = None,              # Maximum number of files to concatenate into one file
        threshold: float = 30,              # Midpoint Change over horizon length, could also be AUTO
        rowLim: int = None,                 # The row limit number for
        trainTestSplit : float = None,      # Split the data between train and test, eg 0.8 for 80% Train, 20% Test
        lookForwardHorizon :int = 10,       # The number of events to look forward after for labelling (prediction horizon)
        representation: str = ORDERBOOKS    # The order book representation, 'orderbooks', 'orderflows'
        
    ):
        self.ticker = ticker
        self.scaling = scaling 
        self.horizon = horizon
        self.threshold = threshold
        self.rowLim = rowLim
        self.maxFiles = maxFiles
        self.trainTestSplit = trainTestSplit
        self.lookForwardHorizon = lookForwardHorizon
        self.representation = representation
        
        assert self.representation in [ORDERBOOKS, ORDERFLOWS], f'representation not valid, please review ({ORDERBOOKS}, {ORDERFLOWS})'
        
        # Required in class
        self.fileLocations = None   # Array of file locations
        self.x = None               # Training data
        self.y = None               # Training Labels
        self.x_test = None          # Test data
        self.y_test = None          # Test labels

    def getFileLocations(self, dataLocation : str = None) -> list[str]:
        """
        Description:
            Get file locations for ticker, this is ORDERBOOK representation
        """
        if dataLocation is None:
            dataLocation = processedDataLocation(self.ticker, self.scaling, representation=ORDERBOOKS)
            
        self.fileLocations = glob(dataLocation + f"/{self.ticker}*.csv")
        
        assert self.fileLocations is not None and len(self.fileLocations) > 0, f"No files found at {dataLocation}"
        
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

    def getOrderFlowsFromFiles(self):
        """
        Description:
            Custom process to handle orderflow representation, usual process required to run first
        """
        locations = processedDataLocation(self.ticker, self.scaling, representation=ORDERFLOWS)
        fileLocations = self.getFileLocations(locations)
        frame = self.getDataFromFiles(fileLocations)
        self.dataFrameToFeatures(alternativeFrame=frame)

    def dataFrameToFeatures(self, alternativeFrame : pd.DataFrame = None) -> np.ndarray:
        """
        Description:
            Turn data into a model 'runnable' dataset, this is the self.x values (features) that are passed into the models
        Parameters:
            alternativeFrame (pd.DataFrame): Process an alternative dataframe to globlFrame
        Returns:
            np.ndarray: shape (batchSize, horizon, features, 1) -> (batchSize, 100, 40, 1)
        """
        
        if alternativeFrame is not None:
            frame = alternativeFrame
        else:
            frame = self.globalFrame
        
        # Set row limit
        self.rowLim = self.rowLim if self.rowLim is not None else len(frame) - self.horizon - self.lookForwardHorizon

        arr = frame.to_numpy()  # shape (total_rows, features)

        # Create sliding windows of shape (rowLim, horizon, features)
        windows = sliding_window_view(arr, window_shape=(self.horizon), axis=0)  # (total_rows-horizon+1, horizon, features)

        # Select only up to rowLim
        datasetX = windows[:self.rowLim, :, :]  # (batchSize, horizon, features)

        # Add the last dimension for channel
        self.x = np.expand_dims(datasetX, axis=-1)  # (batchSize, horizon, features, 1)

        # Transpose horizon and features: (batchSize, horizon, features, 1) -> (batchSize, features, horizon, 1)
        self.x = self.x.transpose(0, 2, 1, 3)  # (batchSize, features, horizon, 1)

        return self.x
    
    def dataFrameToLabelsRaw(self) -> np.ndarray:
        """
        Description:
            Finds the feature labels from the raw LOB data
        """
        # ASKp1, ASKs1, BIDp1, BIDs1, ...
        
        ask1Col = 0; bid1Col = 2
        
        self.rowLim  = self.rowLim if self.rowLim is not None else len(self.globalFrame) - self.horizon - self.lookForwardHorizon
            
        df = self.globalFrame.values  # Convert DataFrame to NumPy array for fast indexing
        horizon = self.horizon
        
        rowLim = min(self.rowLim, len(df) - horizon)

        # For each window, get the last ask/bid in the window as the "start"
        # and the ask/bid at lookForwardHorizon after the window as the "end"
        startAsk = df[self.horizon - 1 : self.horizon - 1 + rowLim, ask1Col]
        endAsk   = df[self.horizon - 1 + self.lookForwardHorizon : self.horizon - 1 + self.lookForwardHorizon + rowLim, ask1Col]
        startBid = df[self.horizon - 1 : self.horizon - 1 + rowLim, bid1Col]
        endBid   = df[self.horizon - 1 + self.lookForwardHorizon : self.horizon - 1 + self.lookForwardHorizon + rowLim, bid1Col]
        
        startMid = (startAsk + startBid) / 2
        endMid   = (endAsk + endBid) / 2
        diff = (endMid - startMid)
        
        assert isinstance(self.threshold, (float, int)) or self.threshold == AUTO, f"Please check threshold is numeric or {AUTO}"
        
        if self.threshold == AUTO:
            # Use the z score to scale the data into thirds
            zscores = scipy.stats.zscore(diff)
            lower, upper = np.percentile(zscores, [33.33, 66.66])
            threshold = (abs(lower) + abs(upper)) / 2  # Symmetric threshold
            print(f"Auto threshold z-score cutoffs: lower={lower:.3f}, upper={upper:.3f}, using threshold={threshold:.3f}")
            down    = (zscores < -threshold).astype(int)
            up      = (zscores >  threshold).astype(int)
            neutral = ((down == 0) & (up == 0)).astype(int)
        else:
            down    = (diff < -self.threshold).astype(int)
            up      = (diff >  self.threshold).astype(int)
            neutral = ((down == 0) & (up == 0)).astype(int)

        # Stack the one-hot encoded labels
        self.y = np.stack([down, neutral, up], axis=1)
        
        return self.y
    
    def splitDataTrainTest(self):
        """
        Description:
            Split the data into test and train if the flag has been enabled
        """
        length = floor(len(self.x) * self.trainTestSplit)
        self.x_test = self.x[length:]
        self.y_test = self.y[length:]
        self.x = self.x[:length]
        self.y = self.y[:length]
        print(f"Train set size: {len(self.x)}, Test set size: {len(self.x_test)}")

    def xyToTensor(self):
        """
        Description:
            Turn the features and labels into a pytorch tensor
        """
        assert self.x is not None, "Please ensure there are test features, run self.dataFrameToFeatures()"
        assert self.y is not None, "Please ensure there are train labels, run self.dataFrameToLabelsRaw()"
        
        if self.trainTestSplit is not None:
            assert self.x_test is not None, "Please ensure there is test data, run self.dataFrameToFeatures()  and self.splitDataTrainTest()"
            assert self.y_test is not None, "Please ensure there are labels, run self.dataFrameToLabelsRaw()  and self.splitDataTrainTest()"
    
        self.x = torch.tensor(self.x, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32)
        
        if self.trainTestSplit is not None:
            self.x_test = torch.tensor(self.x_test, dtype=torch.float32)
            self.y_test = torch.tensor(self.y_test, dtype=torch.float32)
    
    def runFullProcessReturnXY(self, tensor : bool = False) -> tuple[np.ndarray, np.ndarray]:
        """
        Descrition:
            Run full process
        """
        self.getFileLocations()
        self.getDataFromFiles()
        
        if self.representation == ORDERBOOKS:
            self.dataFrameToFeatures()    
        
        self.dataFrameToLabelsRaw()
        
        if self.representation == ORDERFLOWS:
            self.getOrderFlowsFromFiles()
        
        if self.trainTestSplit is not None:
            self.splitDataTrainTest()
        if tensor:
            self.xyToTensor()
        return self.x, self.y
    
    def getTestData(self):
        return self.x_test, self.y_test

if __name__ == "__main__":
    cdl = CustomDataLoader(
        ticker='NFLX',
        scaling=True,
        horizon=100,
        threshold=AUTO,
        maxFiles=2,
        rowLim=None,
        trainTestSplit=0.8,
        lookForwardHorizon=20,
        representation=ORDERFLOWS
    )
    cdl.runFullProcessReturnXY(tensor=True)
    print(cdl.x.shape)
    print(cdl.y.shape)