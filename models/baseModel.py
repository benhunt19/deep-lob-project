from abc import ABC, abstractmethod

class BaseModel(ABC):
    """
    Description:
    This is the base model that all other models inherit from
    """
    def __init__(self):
        self.name = 'BaseClass'
    
    @abstractmethod
    def fit():
        """
        Description:
            The method to fit the model to the training data
        """
        pass
    
    def saveWeights():
        pass