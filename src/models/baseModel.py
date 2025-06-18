from abc import ABC, abstractmethod
import numpy as np
class BaseModel(ABC):
    """
    Description:
        This is the base model that all other models inherit from, essentially the abstract class with contract instructions
    """
    def __init__(self):
        self.name = 'BaseClass'             # Model name
        self.weightsFileFormat = 'h5'       # Extension for saving weights
    
    @abstractmethod
    def train(self, x : np.ndarray, y : np.ndarray):
        """
        Description:
            The method to fit the model to the training data
        Parameters:
            x (ndarray or tensor): The features of the data to train
            y (ndarray or tensor): The values or labels of the data to train
        """
        pass
    
    @abstractmethod
    def predict(self, x : np.ndarray):
        """
        Description:
            Run a forward pass on one or more rows
        Parameters:
            x (ndarray or tensor): The features of the data to test
        """
        pass
    
    @abstractmethod
    def saveWeights(self):
        pass

    # @abstractmethod
    def loadFromWeights(self, weightsPath):
        pass