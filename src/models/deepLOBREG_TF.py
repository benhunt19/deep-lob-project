import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0" # Suppress warning
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Flatten, Dense, Dropout, Activation, Input, LSTM, Reshape, Conv2D, MaxPooling2D, LeakyReLU, concatenate)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from src.models.deepLOB_TF import DeepLOB_TF

class DeepLOBREG_TF(DeepLOB_TF):
    """
    Description:
        This is an adaptation to the deepLOB model build with Tensor Flow
        https://github.com/zcakhaa/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books
    
    Parameters:
        shape (tuple): Shape of input data
        number_of_lstm (int): Number of LSTM in LSTM component
    """
    name = 'deepLOBREG_TF'
    def __init__(self, shape : tuple = (100, 40, 1), number_of_lstm = 64):
        super().__init__(shape, number_of_lstm)
        self.model = self._build_model()                                    # Build the model
        self.name = DeepLOBREG_TF.name                                         # Model name

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

        book_depth = self.shape[1]
        if book_depth in [20, 40]:
            kernel_width = 5 if book_depth == 20 else 10
        else:
            # For other book depths, use a kernel that covers about 1/4 to 1/3 of the depth
            kernel_width = max(2, book_depth // 4)
        conv_first1 = Conv2D(32, (1, kernel_width))(conv_first1)
        
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

        # build the output layer , SINGEL OUTPUT in (-1 , 1)
        out = Dense(1, activation='tanh')(conv_lstm)
        model = Model(inputs=input_lmd, outputs=out)
        adam = Adam(0.0001)
        model.compile(
            optimizer=adam,
            loss='mse',
            metrics=['mse']
        )
        return model
    
    @property
    def earlyStopping(self):
        return EarlyStopping(monitor="val_mse", patience=self.patience, mode="auto", restore_best_weights=True)
