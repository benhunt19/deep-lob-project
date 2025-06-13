import numpy as np
import torch
import torch.nn as nn

from src.data_processing.processData import prepare_x_y, Dataset, batch_gd

if __name__ == "__main__":
    # please change the data_path to your local path
    data_path = '../data/demo'
    test_lim = 100_000
    dec_data = np.loadtxt(data_path + '/Train_Dst_NoAuction_DecPre_CF_7.txt')[:,:test_lim]
    print(dec_data.shape)
    dec_train = dec_data[:, :int(np.floor(dec_data.shape[1] * 0.6))]
    dec_val = dec_data[:, int(np.floor(dec_data.shape[1] * 0.6)):]

    dec_test1 = np.loadtxt(data_path + '/Test_Dst_NoAuction_DecPre_CF_7.txt')
    # dec_test2 = np.loadtxt(data_path + '/Test_Dst_NoAuction_DecPre_CF_8.txt')
    # dec_test3 = np.loadtxt(data_path + '/Test_Dst_NoAuction_DecPre_CF_9.txt')
    # dec_test = np.hstack((dec_test1, dec_test2, dec_test3))
    dec_test = dec_test1

    k = 4               # which prediction horizon
    T = 100             # the length of a single input
    n_hiddens = 64
    checkpoint_filepath = './model_tensorflow2/weights.weights.h5'

    trainX_CNN, trainY_CNN = prepare_x_y(dec_train, k, T)
    valX_CNN, valY_CNN = prepare_x_y(dec_val, k, T)
    testX_CNN, testY_CNN = prepare_x_y(dec_test, k, T)

    batch_size = 16

    dataset_train = Dataset(data=dec_train, k=4, num_classes=3, T=100)
    dataset_val = Dataset(data=dec_val, k=4, num_classes=3, T=100)
    dataset_test = Dataset(data=dec_test, k=4, num_classes=3, T=100)

    train_loader = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=dataset_val, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=False)

    print(dataset_train.x.shape, dataset_train.y.shape)
    
    
    tmp_loader = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=1, shuffle=True)

    for x, y in tmp_loader:
        print(x)
        print(y)
        print(x.shape, y.shape)
        break
    
    from src.models.deepLOB_PT import DeepLOB_PT
    
    model = DeepLOB_PT(y_len = dataset_train.num_classes)
    # model = torch.compile(model)
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = 'cpu'
    print(device)
    # model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    
    train_losses, val_losses = batch_gd(
        model,
        criterion,
        optimizer, 
        train_loader,
        val_loader,
        epochs=4,
        device=device
    )
    
    model = torch.load('best_val_model_pytorch', weights_only=False)

    print("Training finished, testing...")
    
    n_correct = 0.
    n_total = 0.
    for inputs, targets in test_loader:
        # Move to GPU
        inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device, dtype=torch.int64)

        # Forward pass
        outputs = model(inputs)
        
        # Get prediction
        # torch.max returns both max and argmax
        _, predictions = torch.max(outputs, 1)

        # update counts
        n_correct += (predictions == targets).sum().item()
        n_total += targets.shape[0]

    test_acc = n_correct / n_total
    print(f"Test acc: {test_acc:.4f}")