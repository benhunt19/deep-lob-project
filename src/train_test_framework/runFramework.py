# train.py
import hydra
from omegaconf import DictConfig, OmegaConf
import os
from pprint import pprint

from src.core.constants import PROJECT_ROOT, HYDRA_CONFIG_PATH, TRAIN, TEST
from src.train_test_framework.modelTrainTestFramework import ModelTrainTestFramework
from src.train_test_framework.metaMaker import ModelMetaMaker

@hydra.main(config_path=f"{PROJECT_ROOT}/{HYDRA_CONFIG_PATH}", config_name="config", version_base=None)
def runFramework(config: DictConfig):
    # Create list of metas using the meta maker
    metas = ModelMetaMaker.createMeta(config)
    
    # Run train test process
    ModelTrainTestFramework(metas=metas).run()

if __name__ == "__main__":
    runFramework()