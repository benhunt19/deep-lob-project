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
        
        self.model.compile(
            optimizer=Adam(learning_rate=0.001, clipnorm=1.0),  # Gradient clipping
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
        X, y = [], []
        
        # Create windows: lookback windowLength, predict horizon steps ahead
        for i in range(self.windowLength, len(data) - self.horizon):
            # Input window: data[i-windowLength:i]
            window = data[i - self.windowLength : i]
            X.append(window)
            
            # Target: percentage change after horizon steps (directional signal)
            current_price = data[i]
            future_price = data[i + self.horizon]
            target = (future_price - current_price) / current_price  # Percentage return
            y.append(target)
        
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        # Use StandardScaler for each window (fit and transform each window independently)
        scaler = StandardScaler()
        X_normalized = np.zeros_like(X)
        
        for i in range(X.shape[0]):
            window = X[i].reshape(-1, 1)  # Reshape for scaler
            X_normalized[i] = scaler.fit_transform(window).flatten()
        
        # Reshape X for LSTM input (samples, timesteps, features)
        X_normalized = X_normalized.reshape(X_normalized.shape[0], X_normalized.shape[1], 1)
        
        # StandardScaler for targets too
        y_scaler = StandardScaler()
        y = y_scaler.fit_transform(np.array(y).reshape(-1, 1)).flatten()
        
        return X_normalized, y
    
    def train(self, x : tensor, y: tensor, batchSize : int = 64, numEpoch : int = 3, validation_split = 1/10):
        
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