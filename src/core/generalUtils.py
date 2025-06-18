import os

from src.core.constants import PROJECT_ROOT

def weightLocation(model):
    """
    Description:
        Single location for defining where model weights are stored
    Parameters:
        Model (Deep Learning model): The model to get the name from
    """
    filePath = f"{PROJECT_ROOT}/weights/{model.name}"
    if not os.path.exists(filePath):
            os.makedirs(filePath)
    return f"{filePath}/{model.name}.{model.weightsFileFormat}"