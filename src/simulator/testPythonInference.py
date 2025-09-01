from src.loaders.dataLoader import CustomDataLoader
from src.routers.modelRouter import *
import numpy as np
import tensorflow as tf
import torch
import time
import seaborn as sns
import matplotlib.pyplot as plt


if __name__ == "__main__":
    
    meta = {
        'ticker': 'AAPL',
        'model': 'deepLOB_TF',
        'numEpoch': 5,
        'batchSize': 64,
        'steps': ['TRAIN', 'TEST'],
        'maxFiles': 1,
        'scaling': True,
        'threshold': 'auto',
        'rowLim': 400,
        'trainTestSplit': 0.5,
        'lookForwardHorizon': 20,
        'representation': 'orderbooks',
        'labelType': 'CATEGORICAL',
        'archive': True
    }
    
    
    cdl = CustomDataLoader(
        ticker=meta['ticker'],
        scaling=meta['scaling'],
        horizon=100,
        threshold=meta['threshold'],
        maxFiles=meta['maxFiles'],
        rowLim=meta['rowLim'],
        trainTestSplit=meta['trainTestSplit'],
        lookForwardHorizon=meta["lookForwardHorizon"],
        representation=meta['representation'],
        labelType=meta['labelType'],
    )


    models = [DeepLOB_PT(),  DeepLOB_PT()]
    
    val_store = []
    
    for model in models:
        
        x, y = cdl.runFullProcessReturnXY(tensor=model.requiresTensor)
        x_test, y_test = cdl.getTestData()
        expanded_tensor = torch.tensor(x_test[0][np.newaxis, :], dtype=torch.float32)
        model.predict(x=expanded_tensor, y=None)
        model.predict(x=expanded_tensor, y=None)
        model.predict(x=expanded_tensor, y=None)
        model.predict(x=expanded_tensor, y=None)
        model.predict(x=expanded_tensor, y=None)
        model.predict(x=expanded_tensor, y=None)
        model.predict(x=expanded_tensor, y=None)
        
        vals = np.zeros(200)

        for index, (xt, yt) in enumerate(zip(x_test, y_test)):
            expanded_tensor = torch.tensor(xt[np.newaxis, :], dtype=torch.float32)
            start = time.perf_counter()
            model.predict(x=expanded_tensor, y=yt)
            model.predict(x=expanded_tensor, y=yt)
            elapsed = time.perf_counter() - start
            print(elapsed)
            vals[index] = elapsed
        # preds = model.predict(x = x_test, y = y_test)
        # resultsStore['metrics'], resultsStore['metricsStrength'] = ModelTrainTestFramework.processAllMetrics(preds=preds, actual=y_test, labelType=meta['labelType'])

        av = vals.mean()
        
        # Sort the values
        sorted_vals = np.sort(vals)

        # Calculate cut-off indices for top and bottom 10%
        bottom_cutoff_idx = int(len(vals) * 0.1)
        top_cutoff_idx = int(len(vals) * 0.9)

        # Get the trimmed values (remove top and bottom 10%)
        vals_trimmed = sorted_vals[bottom_cutoff_idx:top_cutoff_idx]
        vals = vals_trimmed * 1000
        # Calculate the new average
        av = vals.mean()
        
        val_store.append((vals, av))
    
    sns.set()
    fig, ax = plt.subplots(1, 1, figsize=(12,6))

    labels = ['PyTorch', 'TensorFlow']
    colors = ['blue', 'red']

    for i, (vals, av) in enumerate(val_store):
        ax.hist(vals, density=True, bins=11, color=colors[i], alpha=0.7, 
                edgecolor='black', label=f'{labels[i]} (avg: {av:.2f}ms)')
        plt.axvline(av, color=colors[i], linestyle='--', linewidth=2, label=f'Average = {av:.2f}')

    ax.set_title("Model Inference Time Comparison, 200 runs")
    ax.set_xlabel("Milliseconds")
    ax.set_ylabel("Density")
    ax.legend()

    plt.tight_layout()
    plt.show()