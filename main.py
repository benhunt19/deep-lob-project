import hydra
from omegaconf import DictConfig

from src.train_test_framework.runFramework import runFramework
from src.data_processing.processDataUtils import ProcessDataUtils
from src.core.constants import PROJECT_ROOT, HYDRA_CONFIG_PATH, TRAIN, TEST, PROCESS_DATA, ORDERBOOKS, ORDERFLOWS, ORDERFIXEDVOL

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
        """

        runFramework(config=config)

    if PROCESS_DATA in config.steps:
        f"""
        Process raw data from data/raw folder
            ++steps=["PROCESS_DATA"]
            ++representation="{ORDERBOOKS}", "{ORDERFLOWS}, {ORDERFIXEDVOL}"
            ++scaling=True
        """

        ProcessDataUtils.runDataProcss(
            features=config.representation,
            scaling=config.scaling,
            archive=config.archive
        )

if __name__ == "__main__":
    main()