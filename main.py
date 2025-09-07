import hydra
from omegaconf import DictConfig

from src.train_test_framework.runFramework import runFramework
from src.algo.runAlgoFramework import runAlgoFramework
from src.data_processing.processDataUtils import ProcessDataUtils
from src.core.constants import PROJECT_ROOT, HYDRA_CONFIG_PATH, TRAIN, TEST, TRANSFER_LEARNING, ALGO_TRADING, PROCESS_DATA, ORDERBOOKS, ORDERFLOWS, ORDERVOL, ORDERFIXEDVOL, REGRESSION, CATEGORICAL

@hydra.main(config_path=f"{PROJECT_ROOT}/{HYDRA_CONFIG_PATH}", config_name="config", version_base=None)
def main(config: DictConfig):
    f"""
    This is the main executable, from here we can control the whole train / test process.
    Example usage:
        python main.py ++lookForwardHorizon=[5,10,10,430,100,500] ++ticker='["MSFT"]' ++numEpoch=3
        python main.py ++steps=["{PROCESS_DATA}"] ++representation="{ORDERFLOWS}"
        
        ** The default is stored in {HYDRA_CONFIG_PATH}/config.yaml 
    """
    
    if TRAIN in config.steps or TEST in config.steps:
        f"""
        Run the Train Test Framework, example commands:
            ++lookForwardHorizon=[5,10,10,430,100,500]
            ++ticker='["MSFT"]'
            ++numEpoch=3
            ++representation="{ORDERFLOWS}"
            ++model="deepLOBREG_TF"
            ++rowLim=100000
            ++labelType="{REGRESSION}"
        """

        runFramework(config=config)
        
    if TRANSFER_LEARNING in config.steps:
        f"""
        Run the Train Test Framework for trained models:
            ++lookForwardHorizon=[20]
            ++ticker='["MSFT"]'
            ++representation="{ORDERFLOWS}"
            ++model="deepLOB_TF"
            ++rowLim=100000
            ++labelType="{CATEGORICAL}"
            ++weightsRunID="Ov888KGE"
        """
        runFramework(config=config)
    
    if ALGO_TRADING in config.steps:
        f"""
        Run the Algo Trading Framework with the following parmeters
            ++horizon='[20, 40, 60]'
            ++rowLim=100_000
            ++windowLength=100,
            ++ticker="['AAPL', 'NVDA', 'GOOG']"
            ++date=2025-06-05
            ++signalPercentage=15
            ++plot=False
            ++modelClass= ['LinearRegressionModel']
            ++representation'= 'ORDERFLOWS'
            ++verbose= False
            ++saveResults= False
            ++tradingFees= False
            ++slippage= False
        """
        runAlgoFramework(base=config)


    if PROCESS_DATA in config.steps:
        f"""
        Process raw data from data/raw folder
            ++steps=["PROCESS_DATA"]
            ++representation="{ORDERBOOKS}", "{ORDERFLOWS}, {ORDERFIXEDVOL}, {ORDERVOL}"
            ++scaling=True
            ++rowLim=250000
        """
        ProcessDataUtils.runDataProcss(
            features=config.representation,
            scaling=config.scaling,
            archive=config.archive,
            rowLim=config.rowLim
        )

if __name__ == "__main__":
    main()