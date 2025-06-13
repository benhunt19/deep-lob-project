from abc import ABC, abstractmethod
import numpy as np
class BaseModel(ABC):
    """
    Description:
        This is the base model that all other models inherit from, essentially the abstract class with contract instructions
    """
    def __init__(self):
        self.name = 'BaseClass'
    
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
        pass
    
    def saveWeights(self):
        pass