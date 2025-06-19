import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Flatten, Dense, Dropout, Activation, Input, LSTM, Reshape, Conv2D, MaxPooling2D, LeakyReLU, concatenate)
from tensorflow.keras.optimizers import Adam

from torch import tensor

from src.models.baseModel import BaseModel
from src.core.generalUtils import weightLocation

class DeepLOB_TF(BaseModel):
    """
    Description:
        This is the original deepLOB model build with Tensor Flow
        https://github.com/zcakhaa/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books
    
    Parameters:
        shape (tuple): Shape of input data
        number_of_lstm (int): Number of LSTM in LSTM component
    """
    def __init__(self, shape : tuple = (100, 40, 1), number_of_lstm = 64):
        self.shape = shape                                                  # Shape of the input data
        self.number_of_lstm = number_of_lstm                                # Number of LSTM
        self.model = self._build_model()                                    # Build the model
        self.name = 'deepLOB_TF'                                            # Model name
        self.weightsFileFormat = 'keras'                                    # File format for saving weights

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
        return self.model.fit(
            x=x,
            y=y,
            epochs=numEpoch,
            batch_size=batchSize
        )

    def predict(self, *args, **kwargs):
        return self.model.predict(verbose=0, *args, **kwargs)
    
    def saveWeights(self) -> None:
        self.model.save(weightLocation(self))
    
    def loadFromWeights(self, weightsPath) -> None:
        self.model.load_weights(weightsPath)

    # Review
    def evaluate(self, *args, **kwargs):
        return self.model.evaluate(*args, **kwargs)
    

if __name__ == "__main__":
    model = DeepLOB_TF()
    model.saveWeights()