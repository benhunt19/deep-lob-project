import pandas as pd
import polars as pl
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from typing import Tuple, List
import seaborn as sns
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from numba import jit, njit, prange, cuda
from numpy.lib.stride_tricks import sliding_window_view

from datetime import datetime


from src.core.generalUtils import processDataFileNaming
from src.core.constants import NUMPY_EXTENSION, NUMPY_X_KEY, NUMPY_Y_KEY, ORDERFIXEDVOL, ORDERVOL, ORDERFLOWS

# Functions for creating features from data

# orderflows
def createOrderFlows(orderbook: pd.DataFrame, levels: int, feature_names) -> Tuple[pd.DataFrame, List]:
    """
    Description:
        Process orderflows from the orderboook. (Originally taken from src.data_processing.processData)
    Parameters:
        orderbook (pd.DataFrame): Current orderbook from original representation
        levels (int): How many levels deep is the orderbook 
        feature_names (list): Column names for all the features
    Returns:
        The updated orderbook and featuer_names as a Tuple
    """
    # Compute bid and ask multilevel orderflow.
    ASK_prices = orderbook.loc[:, orderbook.columns.str.contains("ASKp")]
    BID_prices = orderbook.loc[:, orderbook.columns.str.contains("BIDp")]
    ASK_sizes = orderbook.loc[:, orderbook.columns.str.contains("ASKs")]
    BID_sizes = orderbook.loc[:, orderbook.columns.str.contains("BIDs")]

    ASK_price_changes = ASK_prices.diff().dropna().to_numpy()
    BID_price_changes = BID_prices.diff().dropna().to_numpy()
    ASK_size_changes = ASK_sizes.diff().dropna().to_numpy()
    BID_size_changes = BID_sizes.diff().dropna().to_numpy()

    ASK_sizes = ASK_sizes.to_numpy()
    BID_sizes = BID_sizes.to_numpy()

    ASK_OF = (
            (ASK_price_changes > 0.0) * (-ASK_sizes[:-1, :])
            + (ASK_price_changes == 0.0) * ASK_size_changes
            + (ASK_price_changes < 0) * ASK_sizes[1:, :]
    )
    BID_OF = (
            (BID_price_changes < 0.0) * (-BID_sizes[:-1, :])
            + (BID_price_changes == 0.0) * BID_size_changes
            + (BID_price_changes > 0) * BID_sizes[1:, :]
    )

    # Remove all price-volume features and add in orderflow.
    orderbook = orderbook.drop(feature_names, axis=1).iloc[1:, :]

    feature_names_raw = ["ASK_OF", "BID_OF"]
    feature_names = []
    for feature_name in feature_names_raw:
        for i in range(1, levels + 1):
            feature_names += [feature_name + str(i)]
    orderbook[feature_names] = np.concatenate([ASK_OF, BID_OF], axis=1)

    # Re-order columns.
    feature_names_reordered = [[]] * len(feature_names)
    feature_names_reordered[::2] = feature_names[:levels]
    feature_names_reordered[1::2] = feature_names[levels:]
    feature_names = feature_names_reordered

    return orderbook, feature_names

# @njit(parallel=True)
def compute_volumes_cpu(windows, ask_size_idx, bid_size_idx, negativeBidMultiplier, scaling):
    fixed_volumes = []
    for i, window in enumerate(windows):

        # Extract ASK and BID sizes from the snapshot
        ask_sizes = window[:, ask_size_idx]
        bid_sizes = window[:, bid_size_idx]
        
        # Initialize volumes with the right shape based on orderbook_depth
        volumes = np.zeros((window.shape[0], ask_sizes.shape[1] + bid_sizes.shape[1]))
        
        # Process all rows at once using NumPy operations
        bid_volumes = bid_sizes * negativeBidMultiplier  # Apply multiplier to all bid sizes
        
        # Flip the bid volumes along the second axis (columns) and hstack with ask sizes
        volumes = np.hstack((bid_volumes[:, ::-1], ask_sizes))
        
        if scaling:
            non_neg = volumes.ravel()[volumes.ravel() != 0]
            mean = np.mean(np.abs(non_neg))
            volumes = volumes/mean

        fixed_volumes.append(volumes)
    
    return fixed_volumes


# ordervol
def createOrderVolume(
    orderbook: pd.DataFrame,
    ticker: str,
    scaling: bool,
    features: str,
    date: str,
    windowSize=100,
    negativeBids=False,
    rowLim=None
) -> None:
    f"""
    Description:
        Create order volume examples
    Parameters:
        orderbook (pd.DataFrame): Orderbook that has been initially been processed by (processData)
        ticker (str): The ticker string for the required ticker
        scaling (bool): Are we scaing the input data
        features (str): Should be {ORDERVOL}, required for naming
        date (str): Required for naming yyyy-mm-dd
        windowSize (int): How large to make the window (usually 100)
        negativeBids (bool): Are the bids neagative (with the asks positive)
    """
    # Get new dataframe with tick size
    # create sliding windows
    if rowLim is not None:
        orderbook = orderbook.iloc[:rowLim, :]
    print(orderbook)
    
    values = orderbook.values
        
    # Get sliding windows of data, will be 
    windows = sliding_window_view(values, window_shape=(windowSize, values.shape[1]), axis=(0, 1))
    windows = windows[:, 0, :, :]  # shape: (datasize - windowSize + 1, windowSize, 42)
    print(f"Windows shape: {windows.shape}")
    
    fixed_volumes = []
    columns = orderbook.columns
    # Get indexes of relevant columns
    ask_size_idx = np.array([columns.get_loc(col) for col in columns if "ASKs" in col])
    bid_size_idx = np.array([columns.get_loc(col) for col in columns if "BIDs" in col])
    
    ask_price_idx = [columns.get_loc(col) for col in columns if "ASKp" in col]
    bid_price_idx = [columns.get_loc(col) for col in columns if "BIDp" in col]
    
        # Mid price calc
    mid = (values[:, ask_price_idx[0]] + values[:, bid_price_idx[0]]) / 2
    print(f"Mid: {mid}")
    
    negativeBidMultiplier = -1 if negativeBids else 1
    
    fixed_volumes = compute_volumes_cpu(windows, ask_size_idx, bid_size_idx, negativeBidMultiplier, scaling)
    fixed_volumes = np.array(fixed_volumes)
        
    # Print the full fixed_volumes array without truncation
    np.set_printoptions(threshold=np.inf, linewidth=np.inf)
    print(fixed_volumes[0])
    
    print("Saving numpy")
    saveNumpy(array=fixed_volumes, ticker=ticker, scaling=scaling, features=features, date=date, mid=mid )
  

# ---------------------------------------------------
# CPU version (your original)
# ---------------------------------------------------
@njit(parallel=True)
def compute_fixed_volumes_cpu(
        windows,
        ticks,
        ask_idx,
        bid_idx,
        ask_size_idx,
        bid_size_idx,
        bid_sign,
        scaling
    ) -> np.array:
    
    n_windows = windows.shape[0]
    windowSize = windows.shape[1]
    num_ticks = ticks.shape[1]

    fixed_volumes = np.zeros((n_windows, windowSize, num_ticks), dtype=np.float32)
    
    for i in prange(n_windows):
        tick_prices = ticks[i]
        for row_idx in range(windowSize):
            ask_prices = windows[i, row_idx, ask_idx]
            bid_prices = windows[i, row_idx, bid_idx]
            ask_sizes  = windows[i, row_idx, ask_size_idx]
            bid_sizes  = windows[i, row_idx, bid_size_idx]

            for tick_idx in range(num_ticks):
                tick = tick_prices[tick_idx]

                fixed_volumes[i, row_idx, tick_idx] += np.sum(ask_sizes[ask_prices == tick])
                fixed_volumes[i, row_idx, tick_idx] += np.sum(bid_sizes[bid_prices == tick]) * bid_sign

    if scaling:
        mean = np.mean(np.abs(fixed_volumes))
        print(f"Mean: {mean}")
        fixed_volumes /= mean

    return fixed_volumes

# ---------------------------------------------------
# Main wrapper
# ---------------------------------------------------
def createOrderFixedVolume(
    orderbook: pd.DataFrame,
    ticker: str,
    scaling: bool,
    features: str,
    date: str,
    windowSize: int = 100,
    negativeBids: bool = True,
    num_ticks: int = 20,
    rowLim: int = None,
    plot: bool = True,
) -> None:

    if rowLim is not None:
        orderbook = orderbook.iloc[:rowLim, :]

    ticks = processTicks(orderbook=orderbook, num_ticks=num_ticks)
    values = orderbook.values

    # Sliding windows
    windows = sliding_window_view(values, window_shape=(windowSize, values.shape[1]), axis=(0, 1))
    windows = windows[:, 0, :, :]  # (n_windows, windowSize, num_cols)

    columns = orderbook.columns
    ask_price_idx = np.where(columns.str.contains("ASKp"))[0]
    bid_price_idx = np.where(columns.str.contains("BIDp"))[0]
    ask_size_idx  = np.where(columns.str.contains("ASKs"))[0]
    bid_size_idx  = np.where(columns.str.contains("BIDs"))[0]
    
    # Mid price calc
    mid = (values[:, ask_price_idx[0]] + values[:, bid_price_idx[0]]) / 2
    print(f"Mid: {mid}")

    bid_sign = -1 if negativeBids else 1

    print("Running on CPU…")
    fixed_volumes = compute_fixed_volumes_cpu(
        windows.astype(np.float32),
        ticks.astype(np.float32),
        ask_price_idx,
        bid_price_idx,
        ask_size_idx,
        bid_size_idx,
        bid_sign,
        scaling,
    )

    print(f"Fixed volumes shape: {fixed_volumes.shape}")

    if plot:
        plotExamples(fixed_volumes)

    
    saveNumpy(array=fixed_volumes, ticker=ticker, scaling=scaling, features=features, date=date, mid=mid)

     
def processTicks(orderbook: pd.DataFrame, num_ticks=30) -> pd.DataFrame:
    """
    Description:
        Process ticks
    Parameters:
        orderbook (pd.DataFrame): The original dataframe to process into the new format
    """
    # For each row, compute midprice and generate 40 discrete ticks centered at midprice
    ASK_prices = orderbook.loc[:, orderbook.columns.str.contains("ASKp")]
    BID_prices = orderbook.loc[:, orderbook.columns.str.contains("BIDp")]

    # Compute tick size as before
    ask_diffs = ASK_prices.diff().abs().values.flatten()
    bid_diffs = BID_prices.diff().abs().values.flatten()
    ask_diffs = ask_diffs[(ask_diffs > 0) & ~np.isnan(ask_diffs)]
    bid_diffs = bid_diffs[(bid_diffs > 0) & ~np.isnan(bid_diffs)]
    tickSize = int(min(ask_diffs.min(initial=np.inf), bid_diffs.min(initial=np.inf)))
    print(f"The tick size is {tickSize}")

    # Compute midprice for each row
    half_ticks = num_ticks // 2

    best_ask = ASK_prices.min(axis=1)
    best_bid = BID_prices.max(axis=1)

    # For each row, generate price levels:
    # - The first half (including the midpoint if odd) are at or below best_bid
    # - The second half are above best_bid, spaced by tickSize

    price_levels = []
    for bid in best_bid.values:
        # Lower half: up to and including best_bid
        lower_levels = bid - np.arange(half_ticks - 1, -1, -1) * tickSize
        # Upper half: continuing linearly above best_bid
        upper_levels = bid + np.arange(1, num_ticks - half_ticks + 1) * tickSize
        levels = np.concatenate([lower_levels, upper_levels])
        price_levels.append(levels)
    price_levels = np.array(price_levels)

    # Optionally round to int if desired
    price_levels = np.round(price_levels).astype(int)
    print('price_levels shape:', price_levels.shape)
    return price_levels

def saveNumpy(array: np.ndarray, ticker: str, scaling: bool, features: str, date, mid: np.array) -> None:
    """
    Description:
        Save numpy .npz file in standard file saving location for processed data
     Parameters:
        array (np.ndarray): The NumPy array to be saved.
        ticker (str): The ticker symbol associated with the data.
        scaling (bool): Indicates whether scaling has been applied to the data.
        features (str): The feature representation used for the data.
        date (str): String of the date to add to file name
        mid (np.array): Numpy array of mid prices, these mid prices will be ALL, from beginning
    """
    # NUMPY_EXTENSION is '.npz'
    # NUMPY_X_KEY is 'x'
    # NUMPY_Y_KEY is 'mid'
    _, output_name = processDataFileNaming(ticker=ticker, scaling=scaling, representation=features, extension=NUMPY_EXTENSION, date=date)
    np.savez_compressed(output_name, **{NUMPY_X_KEY: array, NUMPY_Y_KEY: mid})
    
def plotExamples(fixed_volumes: np.ndarray, threeD : bool = False):
    """
    Descripton:
        Plot example 2d and 3d surfaces, revealing information about the representation
    Parameters:
        fixed_volumes (np.ndarray): Data to plot from
        threeD (bool): Plot 3d data
    """
    plt.figure(figsize=(18, 6))
    # Plot 9 heatmaps, 100 apart
    num_plots = 6
    plt.figure(figsize=(24, 18))
    for idx in range(num_plots):
        plot_idx = idx * 1000
        if plot_idx < fixed_volumes.shape[0]:
            plt.subplot(2, 3, idx + 1)
            sns.heatmap(fixed_volumes[plot_idx], cmap="RdYlGn", center=0, annot=False)
            # plt.title(f"Heatmap of fixed_volumes[{plot_idx}]")
            plt.xlabel("Tick Index $\kappa$")
            plt.ylabel("Window Row Index")
            # plt.xticks([])
            plt.yticks([])
            plt.gca().set_xticklabels([])
    
    plt.tight_layout(pad=4.0)  # Increase padding between subplots
    plt.subplots_adjust(top=0.80, bottom=0.05, left=0.05, right=0.95)  # Increase gap above top and below bottom
    plt.tight_layout()
    plt.show()
    np.set_printoptions(threshold=1000, linewidth=75)  # Reset to default after printing
    print(fixed_volumes.shape)
    
    # 3D plot of an example from fixed_volumes (surface)
    example_idx = 0  # Change this to plot a different example
    example = fixed_volumes[example_idx]

    if threeD:
        fig = plt.figure(figsize=(16, 9))
        ax = fig.add_subplot(121, projection='3d')

        x = np.arange(example.shape[1])
        y = np.arange(example.shape[0])
        X, Y = np.meshgrid(x, y)
        Z = example

        ax.plot_surface(X, Y, Z, cmap='RdYlGn')
        ax.set_xlabel('Tick Index')
        ax.set_ylabel('Window Row Index')
        ax.set_zlabel('Volume')
        ax.set_title('3D Surface Plot of an example Fixed Volume')

        # 3D bar plot (vertical bars)
        ax2 = fig.add_subplot(122, projection='3d')
        xpos, ypos = X.ravel(), Y.ravel()
        zpos = np.zeros_like(xpos)
        dz = Z.ravel()
        dx = dy = 0.8  # width of the bars

        ax2.bar3d(xpos, ypos, zpos, dx, dy, dz, shade=True, color=plt.cm.RdYlGn((dz - dz.min()) / (np.ptp(dz) + 1e-9)))
        ax2.set_xlabel('Tick Index')
        ax2.set_ylabel('Window Row Index')
        ax2.set_zlabel('Volume')
        ax2.set_title('3D Bar Plot of an example Fixed Volume')

        plt.tight_layout()
        plt.show()
    
if __name__ == "__main__":
    pass