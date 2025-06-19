import pandas as pd
from glob import glob
from tqdm import tqdm

from src.core.constants import PROJECT_ROOT, PROCESSED_DATA_PATH
from src.core.generalUtils import processedDataLocation

# Some data specific constants
DROP_COLUMNS = [0, -1, -2]     # Columns to drop from the processed dataframes

class CustomDataLoader:
    """
    Description:
        Creates a dataset that a model can consume
    """
    def __init__(
        self, 
        ticker : str,                   # Ticker name
        scaled : bool,                  # scaled or unscaled
        horizon : int = 100,            # Horizon length
        maxFiles : int = 5,             # Maximum number of files to concatenate into one file
    ):
        self.ticker = ticker
        self.horizon = horizon
        self.scaled = scaled
        self.x = None
        self.y = None
        self.fileLocations = None

    def getFileLocations(self, dataLocation : str = None) -> list:
        """
        Description:
            Get file locations for ticker
        """
        if dataLocation is None:
            dataLocation = processedDataLocation(self.ticker, self.scaled)
            
        self.fileLocations = glob(dataLocation + f"/{self.ticker}*.csv")
        print(self.fileLocations)
        return self.fileLocations

    def getDataFromFiles(self, fileLocations :list[str] = None, dropCols : list[int] = DROP_COLUMNS):
        """
        Description:
            Get data from CSV files and stack them vertically (append rows)
        """
        all_data = []
        
        if fileLocations is None:
            fileLocations = self.fileLocations
            
        assert fileLocations is not None, "No file Locations provided, run self.getFileLocations()"
        
        for csv in tqdm(fileLocations):
            df = pd.read_csv(csv, header=None)
            all_data.append(df)

        # Stack row-wise
        self.globalFrame = pd.concat(all_data, axis=0, ignore_index=True)
        if dropCols is not None:
            self.globalFrame.drop(self.globalFrame.columns[dropCols], axis=1, inplace=True)
        print(self.globalFrame)
        return self.globalFrame

    def frameToModelInputs():
        
        
                
        

if __name__ == "__main__":
    cdl = CustomDataLoader(
        ticker='CSCO',
        scaled=False,
        horizon=100
    )
    cdl.getFileLocations()
    cdl.getDataFromFiles()