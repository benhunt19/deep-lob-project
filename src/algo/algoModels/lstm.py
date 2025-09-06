from src.algo.algoModels.baseAlgoModel import BaseAlgoClass, AlgoTypes

import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from src.core.generalUtils import weightLocation, nameModelRun, exportLocation

from torch import tensor


class LSTMModel(BaseAlgoClass):
    
    AlgoType = AlgoTypes.PRE_TRAINED
    name = 'LSTMModel'
    
    def __init__(self):
        super().__init__()
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
    
