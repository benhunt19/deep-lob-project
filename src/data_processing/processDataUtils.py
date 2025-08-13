from src.core.constants import PROJECT_ROOT, RAW_DATA_PATH, PROCESSED_DATA_PATH, DATA_PROCESS_LOGS, ORDERBOOKS, ORDERFLOWS, ORDERFIXEDVOL
from src.data_processing.processData import process_data, process_data_per_ticker

class ProcessDataUtils:
    """
    Description:
        Class to handle and process limit order book data
    """
    def __init__(self):
        pass
    
    @staticmethod
    def runDataProcss(
        input_path=f'{PROJECT_ROOT}/{RAW_DATA_PATH}',
        logs_path=f'{PROJECT_ROOT}/{DATA_PROCESS_LOGS}',
        horizons=[100],
        normalization_window=1,
        archive=True,
        scaling=True,
        features=ORDERBOOKS
    ):
        # Just process orderbooks
        if features == ORDERBOOKS:
            process_data_per_ticker(
                input_path=input_path,
                logs_path=logs_path,
                horizons=horizons,
                normalization_window=normalization_window,
                archive=archive,
                scaling=scaling,
                features=ORDERBOOKS
            )
        # Process Orderflows - both flows and books
        elif features == ORDERFLOWS:
            # First process ORDERBOOKS
            process_data_per_ticker(
                input_path=input_path,
                logs_path=logs_path,
                horizons=horizons,
                normalization_window=normalization_window,
                archive=False,
                scaling=scaling,
                features=ORDERBOOKS,
            )
            # Then process the orderflows
            process_data_per_ticker(
                input_path=input_path,
                logs_path=logs_path,
                horizons=horizons,
                normalization_window=normalization_window,
                archive=archive,
                scaling=scaling,
                features=ORDERFLOWS,
            )
        elif features == ORDERFIXEDVOL:
            # First process ORDERBOOKS
            # print("Processing ORDERBOOKS")
            # process_data_per_ticker(
            #     input_path=input_path,
            #     logs_path=logs_path,
            #     horizons=horizons,
            #     normalization_window=normalization_window,
            #     archive=False,
            #     scaling=scaling,
            #     features=ORDERBOOKS,
            # )
            print("PROCESSING ORDERFIXEDVOL")
            # Then process the orderfixedvol
            process_data_per_ticker(
                input_path=input_path,
                logs_path=logs_path,
                horizons=horizons,
                normalization_window=normalization_window,
                archive=archive,
                scaling=scaling,
                features=ORDERFIXEDVOL,
            )
         
        
if __name__ == "__main__":
    util = ProcessDataUtils.runDataProcss(features=ORDERFLOWS)