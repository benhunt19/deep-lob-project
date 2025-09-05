from abc import ABC, abstractmethod
from enum import Enum

class AlgoTypes(Enum):
    DEEPLOB = 'DEEPLOB'
    PRE_TRAINED = 'PRE_TRAINED'
    FIT_ON_THE_GO = 'FIT_ON_THE_GO'


class BaseAlgoClass(ABC):
    
    AlgoType = AlgoTypes.PRE_TRAINED
    
    def __init__(self):
        super().__init__()
        
    @abstractmethod
    def predict(self):
        pass