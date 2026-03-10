import pandas as pd
from strategy import DynamicExitStrategy, check_entry_condition

def run_backtest(df, strategy_params=None):
    """
    Runs the backtest on the full historical dataset using the shared DynamicExitStrategy.
    df: Intraday DataFrame with datetime index.
    strategy_params: Optional dict (e.g. initial_stop_pct, break_even_r, trail_start_r).
    """
    if strategy_params is None:
        strategy_params = {}
        
    df['date_only'] = df.index.date
    trades = []
    
    for current_date, df_day in df.groupby('date_only'):
        df_day = df_day.sort_index()
        if len(df_day) < 2:
            continue
            
        # Opening High Proxy (first candle high)
        opening_high = df_day.iloc[0]['high']
        
        active_strategy = None
        entry_price = 0.0
        entry_time = None
        
        for idx, row in df_day.iterrows():
            current_time = idx
            
            if not active_strategy:
                # We simulate checking to enter using the close of each candle
                # Realistically, a breakout occurs when High > open_high
                if check_entry_condition(row['high'], opening_high, current_time):
                    # Enter at the slippage price (worst of open and opening breakout)
                    entry_price = max(opening_high, row['open'])
                    entry_time = current_time
                    active_strategy = DynamicExitStrategy(entry_price=entry_price, **strategy_params)
            else:
                # We are in a trade. Simulate exit conditions tick-by-tick
                # We test Low first for conservative stops, then High for trailing advances
                
                # Check low-side (Stop loss / Trailing stop out)
                exit_triggered, exit_reason, execute_price = active_strategy.check_exit_condition(row['low'], current_time)
                if exit_triggered:
                    # If gap down, we exit at open price
                    execute_price = min(execute_price, row['open'])
                    trades.append({
                        'date': current_date, 'entry_time': entry_time, 'entry_price': entry_price,
                        'exit_time': current_time, 'exit_price': execute_price, 'exit_reason': exit_reason,
                        'pnl_pct': (execute_price - entry_price) / entry_price
                    })
                    break
                    
                # Check high-side (Advances trailing stop)
                exit_triggered, exit_reason, execute_price = active_strategy.check_exit_condition(row['high'], current_time)
                if exit_triggered: # Only happens on EOD time boundary here
                    trades.append({
                        'date': current_date, 'entry_time': entry_time, 'entry_price': entry_price,
                        'exit_time': current_time, 'exit_price': row['close'], 'exit_reason': exit_reason,
                        'pnl_pct': (row['close'] - entry_price) / entry_price
                    })
                    break
            
    df_trades = pd.DataFrame(trades)
    
    if df_trades.empty:
        return 0.0, df_trades
        
    total_pnl = df_trades['pnl_pct'].sum()
    return total_pnl, df_trades

def print_backtest_stats(df_trades):
    if df_trades.empty:
        print("No trades executed.")
        return
        
    total_trades = len(df_trades)
    winning_trades = len(df_trades[df_trades['pnl_pct'] > 0])
    losing_trades = len(df_trades[df_trades['pnl_pct'] <= 0])
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    total_pnl = df_trades['pnl_pct'].sum() * 100 # percentage
    
    avg_win = df_trades[df_trades['pnl_pct'] > 0]['pnl_pct'].mean() * 100 if winning_trades > 0 else 0
    avg_loss = df_trades[df_trades['pnl_pct'] <= 0]['pnl_pct'].mean() * 100 if losing_trades > 0 else 0
    
    print("-" * 30)
    print("BACKTEST RESULTS")
    print("-" * 30)
    print(f"Total Trades: {total_trades}")
    print(f"Win Rate:     {win_rate:.2%}")
    print(f"Total PnL:    {total_pnl:.2f}%")
    print(f"Avg Win:      {avg_win:.2f}%")
    print(f"Avg Loss:     {avg_loss:.2f}%")
    print("-" * 30)
