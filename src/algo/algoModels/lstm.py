from src.algo.algoModels.baseAlgoModel import BaseAlgoClass, AlgoTypes

from typing import Tuple
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from src.core.generalUtils import weightLocation, nameModelRun, exportLocation
from torch import tensor


class LSTMModel(BaseAlgoClass):
    
    AlgoType = AlgoTypes.PRE_TRAINED
    name = 'LSTMModel'
    
    def __init__(self,  windowLength : int = 100, horizon : int = 20,):
        super().__init__()
        
        self.windowLength = windowLength
        self.horizon = horizon
        
        # Build LSTM model - simplified for speed
        self.model = Sequential([
            LSTM(25, activation='relu', input_shape=(self.lookback, 1), return_sequences=False),
            Dense(10, activation='relu'),
            Dense(1)
        ])
        self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])

        # Early stopping for better generalization - more aggressive for speed
        es = EarlyStopping(monitor='loss', patience=1, restore_best_weights=True, verbose=0)
    
    @property
    def earlyStopping(self):
        return EarlyStopping(
            monitor=self.earlyStoppingMonitor,
            patience=self.patience,
            mode="min" if self.earlyStoppingMonitor == "val_mse" else "max",
            min_delta=0.001,
            restore_best_weights=True
        )
    
    def transformDataToWindows(self, data) -> Tuple:
        """
        Transform time series data into windowed format for LSTM training
        Returns (X, y) where:
        - X: windows of length self.windowLength 
        - y: targets self.horizon steps ahead
        """
        X, y = [], []
        
        # Create windows: lookback windowLength, predict horizon steps ahead
        for i in range(self.windowLength, len(data) - self.horizon):
            # Input window: data[i-windowLength:i]
            window = data[i - self.windowLength : i]
            X.append(window)
            
            # Target: data point horizon steps ahead
            target = data[i + self.horizon]
            y.append(target)
        
        # Convert to numpy arrays and reshape for LSTM
        X = np.array(X)
        y = np.array(y)
        
        # Reshape X to (samples, timesteps, features) for LSTM input
        X = X.reshape(X.shape[0], X.shape[1], 1)
        
        return X, y
    
    def train(self, x : tensor, y: tensor, batchSize : int, numEpoch : int, validation_split = 1/10):

        self.model.fit(
            x=x,
            y=y,
            epochs=numEpoch,
            batch_size=batchSize,
            callbacks=[self.earlyStopping],
            validation_split=validation_split
        )


    def predict(self, x : tensor, y : tensor = None, verbose : int = 0):
        res = self.model.predict(x=x, verbose=verbose)
        return res
        
    def saveWeights(self, run_id : str = "") -> None:
        name = nameModelRun(runID=run_id)
        self.model.save(weightLocation(self, name))
        self.exportWeights(run_id)
    
    def exportWeights(self, run_id : str = "") -> None:
        name = nameModelRun(runID=run_id)
        self.model.export(exportLocation(self, name))
    
    def loadFromWeights(self, weightsPath) -> None:
        self.model.load_weights(weightsPath)