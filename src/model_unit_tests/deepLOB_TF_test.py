import numpy as np
import tensorflow as tf
import torch as torch

from src.models.deepLOB_TF import DeepLOB_TF
from src.data_processing.processData import prepare_x_y, Dataset, batch_gd
from src.core.constants import DEMO_DATA_PATH, PROJECT_ROOT

if __name__ == "__main__":
    
    # please change the data_path to your local path
    data_path = PROJECT_ROOT + '/' + DEMO_DATA_PATH
    test_lim = 20_000
    dec_data = np.loadtxt(data_path + '/Train_Dst_NoAuction_DecPre_CF_7.txt')[:,:test_lim]
    print(dec_data.shape)
    dec_train = dec_data[:, :int(np.floor(dec_data.shape[1] * 0.6))]
    dec_val = dec_data[:, int(np.floor(dec_data.shape[1] * 0.6)):]

    dec_test1 = np.loadtxt(data_path + '/Test_Dst_NoAuction_DecPre_CF_7.txt')
    dec_test2 = np.loadtxt(data_path + '/Test_Dst_NoAuction_DecPre_CF_8.txt')
    dec_test3 = np.loadtxt(data_path + '/Test_Dst_NoAuction_DecPre_CF_9.txt')
    dec_test = np.hstack((dec_test1, dec_test2, dec_test3))
    # dec_test = dec_test1

    k = 4               # which prediction horizon
    T = 100             # the length of a single input
    n_hiddens = 64
    checkpoint_filepath = './model_tensorflow2/weights.weights.h5'

    trainX_CNN, trainY_CNN = prepare_x_y(dec_train, k, T)
    valX_CNN, valY_CNN = prepare_x_y(dec_val, k, T)
    testX_CNN, testY_CNN = prepare_x_y(dec_test, k, T)
    
    # trainX_CNN = np.squeeze(trainX_CNN)
    
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_loss',
        mode='auto',
        save_best_only=True
    )
    
    n_hiddens = 64
    
    print('trainX_CNN.shape', trainX_CNN.shape)
    print('trainX_CNN.shape', trainX_CNN.shape)
    
    # train
    model = DeepLOB_TF(trainX_CNN.shape[1:], n_hiddens)
    
    # Test to see if this works with torch tensors
    trainX_CNN = torch.from_numpy(trainX_CNN)
    trainY_CNN = torch.from_numpy(trainY_CNN)
    
    model.train(
        x=trainX_CNN,
        y=trainY_CNN,
        epochs=10,
        batch_size=128,
        callbacks=[model_checkpoint_callback]
    )
    
    model.model.load_weights(checkpoint_filepath)
    print("The shape of testX_CNN is ", testX_CNN.shape)
    pred = model.predict(testX_CNN)
    print(pred)