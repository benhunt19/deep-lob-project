import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0" # Suppress warning
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
        
from tensorflow.keras import layers, models
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Flatten, Dense, Dropout, Activation, Input, LSTM, Reshape, Conv2D, MaxPooling2D, LeakyReLU, concatenate)
from tensorflow.keras.optimizers import Adam

from torch import tensor
import gc

from src.models.baseModel import BaseModel
from src.core.generalUtils import weightLocation, nameModelRun

class DeepLOB_TF(BaseModel):
    """
    Description:
        This is the original deepLOB model build with Tensor Flow
        https://github.com/zcakhaa/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books
    
    Parameters:
        shape (tuple): Shape of input data
        number_of_lstm (int): Number of LSTM in LSTM component
    """
    name = 'deepLOB_TF'
    def __init__(self, shape : tuple = (100, 40, 1), number_of_lstm = 64):
        super().__init__()
        self.shape = shape                                                  # Shape of the input data
        self.number_of_lstm = number_of_lstm                                # Number of LSTM
        self.model = self._build_model()                                    # Build the model
        self.name = DeepLOB_TF.name                                         # Model name
        self.weightsFileFormat = 'h5'                                       # File format for saving weights

    def _build_model(self):
        input_lmd = Input(shape=self.shape)
        
        # build the convolutional block
        conv_first1 = Conv2D(32, (1, 2), strides=(1, 2))(input_lmd)
        conv_first1 = LeakyReLU(alpha=0.01)(conv_first1)
        conv_first1 = Conv2D(32, (4, 1), padding='same')(conv_first1)
        conv_first1 = LeakyReLU(alpha=0.01)(conv_first1)
        conv_first1 = Conv2D(32, (4, 1), padding='same')(conv_first1)
        conv_first1 = LeakyReLU(alpha=0.01)(conv_first1)

        conv_first1 = Conv2D(32, (1, 2), strides=(1, 2))(conv_first1)
        conv_first1 = LeakyReLU(alpha=0.01)(conv_first1)
        conv_first1 = Conv2D(32, (4, 1), padding='same')(conv_first1)
        conv_first1 = LeakyReLU(alpha=0.01)(conv_first1)
        conv_first1 = Conv2D(32, (4, 1), padding='same')(conv_first1)
        conv_first1 = LeakyReLU(alpha=0.01)(conv_first1)

        if self.shape[1] == 20:
            conv_first1 = Conv2D(32, (1, 5))(conv_first1) # needed if using orderflows, need to add logic
        else:
            conv_first1 = Conv2D(32, (1, 10))(conv_first1)
        
        conv_first1 = LeakyReLU(alpha=0.01)(conv_first1)
        conv_first1 = Conv2D(32, (4, 1), padding='same')(conv_first1)
        conv_first1 = LeakyReLU(alpha=0.01)(conv_first1)
        conv_first1 = Conv2D(32, (4, 1), padding='same')(conv_first1)
        conv_first1 = LeakyReLU(alpha=0.01)(conv_first1)
        
        # build the inception module
        convsecond_1 = Conv2D(64, (1, 1), padding='same')(conv_first1)
        convsecond_1 = LeakyReLU(alpha=0.01)(convsecond_1)
        convsecond_1 = Conv2D(64, (3, 1), padding='same')(convsecond_1)
        convsecond_1 = LeakyReLU(alpha=0.01)(convsecond_1)

        convsecond_2 = Conv2D(64, (1, 1), padding='same')(conv_first1)
        convsecond_2 = LeakyReLU(alpha=0.01)(convsecond_2)
        convsecond_2 = Conv2D(64, (5, 1), padding='same')(convsecond_2)
        convsecond_2 = LeakyReLU(alpha=0.01)(convsecond_2)

        convsecond_3 = MaxPooling2D((3, 1), strides=(1, 1), padding='same')(conv_first1)
        convsecond_3 = Conv2D(64, (1, 1), padding='same')(convsecond_3)
        convsecond_3 = LeakyReLU(alpha=0.01)(convsecond_3)
        
        convsecond_output = concatenate([convsecond_1, convsecond_2, convsecond_3], axis=3)
        conv_reshape = Reshape((int(convsecond_output.shape[1]), int(convsecond_output.shape[3])))(convsecond_output)
        conv_reshape = Dropout(0.2, noise_shape=(None, 1, int(conv_reshape.shape[2])))(conv_reshape, training=True)

        # build the last LSTM layer
        conv_lstm = LSTM(self.number_of_lstm)(conv_reshape)

        # build the output layer
        out = Dense(3, activation='softmax')(conv_lstm)
        model = Model(inputs=input_lmd, outputs=out)
        adam = Adam(0.0001)
        model.compile(
            optimizer=adam,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def train(self, x : tensor, y: tensor, batchSize : int, numEpoch : int):
        self.model.fit(
            x=x,
            y=y,
            epochs=numEpoch,
            batch_size=batchSize
        )
        del x, y
        gc.collect()

    def predict(self, x : tensor, y : tensor = None):
        res = self.model.predict(x=x, verbose=0)
        del x
        gc.collect()
        return res
    
    def saveWeights(self, run_id : str = "") -> None:
        name = nameModelRun(runID=run_id)
        self.model.save(weightLocation(self, name))
    
    def loadFromWeights(self, weightsPath) -> None:
        self.model.load_weights(weightsPath)
    
if __name__ == "__main__":
    model = DeepLOB_TF()
    model.saveWeights()