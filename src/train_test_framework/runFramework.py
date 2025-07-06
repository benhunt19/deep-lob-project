# train.py
import hydra
from omegaconf import DictConfig, OmegaConf
from src.core.constants import PROJECT_ROOT, HYDRA_CONFIG_PATH
import os

@hydra.main(config_path=f"{PROJECT_ROOT}/{HYDRA_CONFIG_PATH}", config_name="config", version_base=None)
def runFramework(config: DictConfig):
    print(config)

if __name__ == "__main__":
    runFramework()