"""
Plot DeepLOB trading decisions and price movements
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.algo.algoTrading import HFTStrategy, ModelBasedPredictor, run_backtest_from_df, getBidMidAsk
from src.algo.algoUtils import Direction

def plot_deeplob_decisions():
    # Load market data
    ticker = 'AAPL'
    date = '2025-06-04'
    
    print("Loading market data...")
    df = getBidMidAsk(ticker=ticker, date=date)
    row_lim = 5000  # Smaller dataset for better visualization
    df = df[:row_lim]
    
    print(f"Loaded {len(df)} data points")
    
    # Create DeepLOB strategy
    deeplob_strategy = HFTStrategy(
        predictor=ModelBasedPredictor(
            model_type='deeplob',
            train_window=200,
            retrain_frequency=50,
            forecast_horizon=20,
            signal_threshold=0.1,  # Lower threshold for more trades
            signal_percentile_threshold=0.70,  # Top 30% of signals
            use_percentile_filter=True,
            ticker='AAPL'
        ),
        slippage_ticks=1,
        min_spread_filter=0.0001,
        transaction_cost_bps=0.1
    )
    
    print("\nRunning DeepLOB backtest...")
    
    # Run backtest
    results = run_backtest_from_df(deeplob_strategy, df)
    
    # Extract data for plotting
    mid_prices = df['mid'].values
    timestamps = np.arange(len(mid_prices))
    
    # Get trades
    trades = deeplob_strategy.trades
    
    # Get prediction history
    prediction_history = deeplob_strategy.predictor.prediction_history if deeplob_strategy.predictor else []
    
    print(f"\nResults:")
    print(f"Total Trades: {results['total_trades']}")
    print(f"Total P&L: ${results['total_pnl']:.2f}")
    print(f"Win Rate: {results['win_rate']:.1%}")
    
    # Create the plot
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
    
    # Plot 1: Price movements with trade entries/exits
    ax1.plot(timestamps, mid_prices, 'b-', linewidth=1, alpha=0.7, label='Mid Price')
    
    # Mark trade entries and exits
    for trade in trades:
        if trade.pnl() is not None:
            entry_tick = trade.timestamp
            exit_tick = trade.exit_timestamp
            
            if trade.direction == Direction.LONG:
                ax1.scatter(entry_tick, mid_prices[entry_tick], color='green', marker='^', s=100, alpha=0.8, label='Long Entry' if entry_tick == trades[0].timestamp else "")
                if exit_tick < len(mid_prices):
                    ax1.scatter(exit_tick, mid_prices[exit_tick], color='red', marker='v', s=100, alpha=0.8, label='Long Exit' if entry_tick == trades[0].timestamp else "")
            else:  # SHORT
                ax1.scatter(entry_tick, mid_prices[entry_tick], color='red', marker='v', s=100, alpha=0.8, label='Short Entry' if entry_tick == trades[0].timestamp else "")
                if exit_tick < len(mid_prices):
                    ax1.scatter(exit_tick, mid_prices[exit_tick], color='green', marker='^', s=100, alpha=0.8, label='Short Exit' if entry_tick == trades[0].timestamp else "")
    
    ax1.set_title(f'DeepLOB Trading Decisions - {ticker} {date}')
    ax1.set_xlabel('Tick')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Signal strength over time
    if prediction_history:
        signal_ticks = [pred['tick'] for pred in prediction_history]
        signal_strengths = [pred['signal_strength'] for pred in prediction_history]
        
        ax2.plot(signal_ticks, signal_strengths, 'purple', linewidth=1, alpha=0.7, label='Signal Strength')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.axhline(y=0.5, color='green', linestyle='--', alpha=0.5, label='Positive Threshold')
        ax2.axhline(y=-0.5, color='red', linestyle='--', alpha=0.5, label='Negative Threshold')
        
        # Mark trade signals
        for trade in trades:
            entry_tick = trade.timestamp
            if entry_tick < len(signal_ticks):
                # Find closest signal
                closest_idx = min(range(len(signal_ticks)), key=lambda i: abs(signal_ticks[i] - entry_tick))
                if abs(signal_ticks[closest_idx] - entry_tick) < 10:  # Within 10 ticks
                    signal_val = signal_strengths[closest_idx]
                    color = 'green' if trade.direction == Direction.LONG else 'red'
                    ax2.scatter(entry_tick, signal_val, color=color, marker='o', s=50, alpha=0.8)
        
        ax2.set_title('DeepLOB Signal Strength')
        ax2.set_xlabel('Tick')
        ax2.set_ylabel('Signal Strength')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No prediction history available', 
                transform=ax2.transAxes, ha='center', va='center')
        ax2.set_title('DeepLOB Signal Strength - No Data')
    
    # Plot 3: Cumulative P&L
    if trades:
        trade_ticks = []
        cumulative_pnl = []
        running_pnl = 0
        
        for trade in trades:
            if trade.pnl() is not None and trade.exit_timestamp is not None:
                trade_ticks.append(trade.exit_timestamp)
                running_pnl += trade.pnl()
                cumulative_pnl.append(running_pnl)
        
        if trade_ticks:
            ax3.plot(trade_ticks, cumulative_pnl, 'orange', linewidth=2, marker='o', markersize=4, label='Cumulative P&L')
            ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            # Color the line green/red based on profit/loss
            for i in range(len(cumulative_pnl)):
                color = 'green' if cumulative_pnl[i] >= 0 else 'red'
                if i > 0:
                    ax3.plot(trade_ticks[i-1:i+1], cumulative_pnl[i-1:i+1], color=color, linewidth=2, alpha=0.7)
            
            ax3.set_title(f'Cumulative P&L (Final: ${running_pnl:.3f})')
            ax3.set_xlabel('Tick')
            ax3.set_ylabel('Cumulative P&L ($)')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'No completed trades', 
                    transform=ax3.transAxes, ha='center', va='center')
            ax3.set_title('Cumulative P&L - No Trades')
    else:
        ax3.text(0.5, 0.5, 'No trades executed', 
                transform=ax3.transAxes, ha='center', va='center')
        ax3.set_title('Cumulative P&L - No Trades')
    
    plt.tight_layout()
    
    # Save the plot
    filename = f'deeplob_analysis_{ticker}_{date.replace("-", "")}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    # Print trade summary
    if trades:
        print(f"\nTrade Summary:")
        print(f"{'Trade':<5} {'Type':<5} {'Entry':<6} {'Exit':<6} {'Duration':<8} {'P&L':<8} {'Entry Price':<12} {'Exit Price':<12}")
        print("-" * 80)
        
        for i, trade in enumerate(trades):
            if trade.pnl() is not None:
                trade_type = 'LONG' if trade.direction == Direction.LONG else 'SHORT'
                duration = trade.exit_timestamp - trade.timestamp if trade.exit_timestamp else 'N/A'
                print(f"{i+1:<5} {trade_type:<5} {trade.timestamp:<6} {trade.exit_timestamp:<6} "
                      f"{duration:<8} {trade.pnl():<8.4f} {trade.entry_price:<12.4f} {trade.exit_price:<12.4f}")
    
    # Print model statistics
    if prediction_history:
        signals = [pred['signal_strength'] for pred in prediction_history]
        print(f"\nSignal Statistics:")
        print(f"Total signals generated: {len(signals)}")
        print(f"Average signal strength: {np.mean(np.abs(signals)):.3f}")
        print(f"Max signal strength: {max(signals):.3f}")
        print(f"Min signal strength: {min(signals):.3f}")
        print(f"Signals > 0.5: {len([s for s in signals if s > 0.5])}")
        print(f"Signals < -0.5: {len([s for s in signals if s < -0.5])}")

if __name__ == "__main__":
    plot_deeplob_decisions()
