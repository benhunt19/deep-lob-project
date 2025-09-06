from src.algo.algoModels.baseAlgoModel import BaseAlgoClass, AlgoTypes
from typing import Tuple
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from src.core.generalUtils import weightLocation, nameModelRun, exportLocation
from torch import tensor


class LSTMModel(BaseAlgoClass):
    
    AlgoType = AlgoTypes.PRE_TRAINED
    name = 'LSTMModel'
    
    def __init__(self,  windowLength : int = 100, horizon : int = 20, patience : int = 3):
        super().__init__()
        
        self.windowLength = windowLength
        self.horizon = horizon
        self.patience = patience
        self.earlyStoppingMonitor = "val_mse"
        
        # Build LSTM model - simplified for speed
        self.model = Sequential([
            LSTM(128, return_sequences=True, input_shape=(self.windowLength, 1)),
            Dropout(0.2),
            LSTM(128, return_sequences=True),  # Additional layer
            Dropout(0.2),
            LSTM(64, return_sequences=False),
            Dropout(0.3),  # Slightly higher dropout before dense layers
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='tanh')
        ])
        optimizer = adam = Adam(0.0001)
        self.model.compile(
            optimizer=optimizer,
            loss='huber',  # More robust to outliers
            metrics=['mae']
        )

        # Early stopping for better generalization - more aggressive for speed
        es = EarlyStopping(monitor='loss', patience=1, restore_best_weights=True, verbose=0)
    
    @property
    def earlyStopping(self):
        return EarlyStopping(
            monitor=self.earlyStoppingMonitor,
            patience=self.patience,
            mode="min",
            min_delta=0.0025,
            restore_best_weights=True
        )
    
    def transformDataToWindows(self, data) -> Tuple:
        """
        Transform time series data into windowed format for LSTM training
        Returns (X, y) where:
        - X: normalized windows of length self.windowLength 
        - y: directional targets self.horizon steps ahead
        """
        X, y = np.zeros((len(data) - self.windowLength, self.windowLength)), np.zeros(len(data) - self.windowLength)
        
        # Create windows: lookback windowLength, predict horizon steps ahead
        for i in range(0, len(data) - self.windowLength - self.horizon):
                    # Input window: data[i:i+windowLength]
            X[i] = data[i : i + self.windowLength]
            
            # Target: percentage change after horizon steps (directional signal)
            current_price = data[i + self.windowLength - 1]  # Last price in window
            future_price = data[i + self.windowLength + self.horizon - 1]
            target = future_price - current_price  # Percentage return
            y[i] = target
        
        # Normalize the entire dataset consistently
        # Option 1: Normalize each feature across all samples (recommended)
        scaler = StandardScaler()
        X_normalized = scaler.fit_transform(X.reshape(-1, 1)).reshape(X.shape)
        X_normalized = X_normalized.reshape(X_normalized.shape[0], X_normalized.shape[1], 1)
        
        y_normalized = y / y.std()
        
        return X_normalized, y_normalized  # Return scaler for consistent inference
    
    def train(self, x : tensor, y: tensor, batchSize : int = 64, numEpoch : int = 1, validation_split = 0.15):
        
        self.model.fit(
            x=x,
            y=y,
            epochs=numEpoch,
            batch_size=batchSize,
            callbacks=[self.earlyStopping],
            validation_split=validation_split
        )


    def predict(self, x : tensor, y : tensor = None, verbose : int = 0):
        # x_trans, _ = self.transformDataToWindows(x)
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
        

if __name__ == "__main__":
    pass