from src.core.constants import PROJECT_ROOT, RAW_DATA_PATH, PROCESSED_DATA_PATH, DATA_PROCESS_LOGS, ORDERBOOKS, ORDERFLOWS, ORDERVOL, ORDERFIXEDVOL
from src.data_processing.processData import process_data, process_data_per_ticker
import omegaconf
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
        features=ORDERBOOKS,
        rowLim=None
    ):
        # Just process orderbooks
        
        if isinstance(features, omegaconf.listconfig.ListConfig):
            features = list(features)
        
        if isinstance(features, list):
            
            if ORDERFLOWS in features:
                process_data_per_ticker(
                input_path=input_path,
                logs_path=logs_path,
                horizons=horizons,
                normalization_window=normalization_window,
                archive=False,
                scaling=scaling,
                features=ORDERFLOWS,
                rowLim=rowLim
            )
                
            filteredFeatures = list(filter(lambda x : x not in [ORDERBOOKS, ORDERFLOWS], features))
            
            if len(filteredFeatures) > 0:
                process_data_per_ticker(
                    input_path=input_path,
                    logs_path=logs_path,
                    horizons=horizons,
                    normalization_window=normalization_window,
                    archive=archive,
                    scaling=scaling,
                    features=ORDERBOOKS,
                    additional_features=filteredFeatures,
                    rowLim=rowLim
                )
        
        elif features == ORDERBOOKS:
            process_data_per_ticker(
                input_path=input_path,
                logs_path=logs_path,
                horizons=horizons,
                normalization_window=normalization_window,
                archive=archive,
                scaling=scaling,
                features=ORDERBOOKS,
                rowLim=rowLim
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
                rowLim=rowLim
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
                rowLim=rowLim
            )
        
        elif features in (ORDERVOL, ORDERFIXEDVOL):
            # First process ORDERBOOKS, then process the additional features
            print("Processing ORDERBOOKS")
            process_data_per_ticker(
                input_path=input_path,
                logs_path=logs_path,
                horizons=horizons,
                normalization_window=normalization_window,
                archive=False,
                scaling=scaling,
                features=ORDERBOOKS,
                additional_features=[features],
                rowLim=rowLim
            )
        
if __name__ == "__main__":
    util = ProcessDataUtils.runDataProcss(features=ORDERFLOWS)