import logging
from pprint import pprint
import numpy as np
import gc
import json
from glob import glob

from src.core.generalUtils import runID, processedDataLocation
from src.routers.modelRouter import *
from src.core.constants import TEST, TRAIN, VALIDATE, AUTO, GLOBAL_LOGGER, PROJECT_ROOT, RESULTS_PATH, ORDERBOOKS, ORDERFLOWS
from src.loaders.dataLoader import CustomDataLoader
from src.train_test_framework.metaConstants import META_DEFAULTS, REQUIRED_FIELD, DEFAULT_TEST_TRAIN_SPLIT
from datetime import datetime, timedelta

class ModelTrainTestFramework:
    """
    Description:
        Model training framework
    Paraneters:
        metas (list[dict]): Metas to we will use  to drive the framework
        log (bool): Run logging
    """
    def __init__(self, metas : list[dict], log : bool = True):
        self.metas = metas
        self.logger = None
        if log:
            self.initLogger()
        
    def trainModel(self, model, x: np.ndarray, y: np.ndarray, meta : dict, run_id : str = "") -> None:
        """
        Description:
            Create dataset and train model on data
        Parameters:
            model: Model to run 
            meta 
        """
        model.train(
            x=x,
            y=y,
            numEpoch=meta['numEpoch'],
            batchSize=meta['batchSize']
        )
        model.saveWeights(run_id=run_id)
        # Free memory after training
        del x, y
        gc.collect()
            
    def run(self, metas : list[dict] = None) -> None:
        """
        Description:
            Run model train test framework
        Parameters:
            metas: The model metas to run off for each run
        """
        
        if metas is None and self.metas is not None:
            metas = self.metas
        
        # Run for each meta
        for meta in metas:
            # Generate id for run
            run_id = runID()
            
            # Apply defaults to meta if not explicitly stated
            meta = self.applyMetaDefaults(meta=meta)
            model = meta['model'](**meta['modelKwargs'])
            pprint(meta)
            
            # results
            resultsStore = {
                'run_id': run_id,
                'meta': meta
            }
            
            # No need to clone as the model isn't used again as is
            resultsStore['meta']['model'] = str(model.name)
            
            cdl = CustomDataLoader(
                ticker=meta['ticker'],
                scaling=meta['scaling'],
                horizon=100,
                threshold=meta['threshold'],
                maxFiles=meta['maxFiles'],
                rowLim=meta['rowLim'],
                trainTestSplit=meta['trainTestSplit'],
                representation=meta['representation'],
            )
            
            if TRAIN in meta['steps']:
                
                if self.logger is not None:
                    self.logger.info("Started training...")
                
                # startDate = meta['startDate']
                    
                # Find the earliest start date automatically
                # if meta['startDate'] == AUTO:
                #     dataLocation = processedDataLocation(meta['ticker'], meta['scaling'], representation=meta['representation'])
                #     fileLocations = glob(dataLocation + f"/{meta['ticker']}*.csv")
                    
                #     assert len(fileLocations) > 0, "Error attempting to find start date from file names, please check f{dataLocation}"
                
                #     startDate = fileLocations[0].split('_')[-1].strip('.csv')

                # dates = []
                # current_date = datetime.strptime(startDate, "%Y-%m-%d")
                # days_added = 0
                # while days_added < meta['trainDays']:
                #     if current_date.weekday() < 5:  # 0=Monday, ..., 4=Friday
                #         dates.append(current_date.strftime('%Y-%m-%d'))
                #         days_added += 1
                #     current_date += timedelta(days=1)
                
                # print(dates)
                
                # Train model over each day
                # for date in dates:
                try:
                    print("Runnign training")
                    x, y = cdl.runFullProcessReturnXY(tensor=model.requiresTensor)
                    print(x)
                    print(y)
                    self.trainModel(model=model, x=x, y=y, meta=meta, run_id=run_id)
                    # Clear memory
                    cdl.x, cdl.y = None, None, gc.collect()
                except Exception as e:
                    print(e)
            
            if TEST in meta['steps']:
                if TRAIN not in meta['steps']:
                    # Need to run full process as not run in above
                    cdl.runFullProcessReturnXY(tensor=model.requiresTensor)
                x_test, y_test = cdl.getTestData()
                preds = model.predict(x = x_test, y = y_test)
                correct = (preds.argmax(axis=1) == y_test.argmax(axis=1)).sum()
                total = y_test.shape[0]
                accuracy = correct / total 
                print(f"Test Accuracy: {accuracy:.4f}")
                # Free test data memory
                del x_test, y_test, preds
                gc.collect()
                resultsStore['accuracy'] = float(accuracy)

            # Save resultsStore as JSON
            results_path = f"{PROJECT_ROOT}/{RESULTS_PATH}/results_{run_id}.json"
            with open(results_path, "w") as f:
                json.dump(resultsStore, f, indent=4)
            
            # Explicitly delete CustomDataLoader and model to free memory
            del cdl
            del model
            gc.collect()
    
    def initLogger(self):
        self.logger = logging.getLogger(GLOBAL_LOGGER)
        self.logger.setLevel(logging.INFO)
        if not self.logger.hasHandlers():
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s:  %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
    def applyMetaDefaults(self, meta : dict) -> dict:
        """
        Description:
            Get the meta working based off the least amount of meta
        Parameters:
            Meta (dict): The meta that is going to drive the model
        """
        for key, value in META_DEFAULTS.items():
            if key not in meta and value == REQUIRED_FIELD:
                ValueError(f"{key} not in meta, please add")
            elif key not in meta and value != REQUIRED_FIELD:
                meta[key] = value
        # Apply default split if there is testing
        if TEST in meta['steps'] and 'trainTestSplit' not in meta:
            meta['trainTestSplit'] = DEFAULT_TEST_TRAIN_SPLIT
        return meta
        
if __name__ == "__main__":
    metas = [
        {
            'model': DeepLOB_TF,
            'modelKwargs': {
                # 'shape': (100, 40, 1)
            },
            'numEpoch': 3,
            'ticker': 'NFLX',
            'steps' : [TRAIN, TEST],
            'trainTestSplit': 0.8,
            'maxFiles': 4,
            'threshold': AUTO,
            'rowLim': 1_000_000,
            'lookForwardHorizon': 5,
            'representation': ORDERBOOKS,
            # 'startDate': AUTO,
            # 'trainDays': 5
        },
    ]
    
    # results_path = f"{PROJECT_ROOT}/{RESULTS_PATH}/results_{123}.json"
    # with open(results_path, "w") as f:
    #     json.dump(metas, f, indent=4)
    
    mttf = ModelTrainTestFramework(metas, log=True).run()