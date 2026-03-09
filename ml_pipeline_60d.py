import os
import pickle
import json
import pandas as pd
import numpy as np
import yfinance as yf
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from backtester import run_backtest

COMM_RATE = 0.00132

def calculate_adr(df):
    """Simple ADR calculation (proven optimal for PCA PC1)"""
    daily = df.resample('D').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
    if len(daily) < 2: return 0
    daily['adr'] = (daily['high'] - daily['low']) / daily['low']
    return daily['adr'].mean()


def main():
    print("Loading universe symbols...")
    with open('russell3000_5min.pkl', 'rb') as f:
        old_data = pickle.load(f)
    symbols = list(old_data.keys())
    
    output_pkl = 'russell3000_60d_5min.pkl'
    data_60d = {}
    
    if os.path.exists(output_pkl):
        print(f"Loading existing {output_pkl}...")
        with open(output_pkl, 'rb') as f:
            data_60d = pickle.load(f)
    else:
        print(f"Downloading 60 days of 5m data for {len(symbols)} symbols from yfinance...")
        batch_size = 100
        for i in tqdm(range(0, len(symbols), batch_size), desc="yfinance bulk downloading"):
            batch = symbols[i:i+batch_size]
            batch_str = " ".join(batch)
            df_batch = yf.download(batch_str, period='60d', interval='5m', group_by='ticker', progress=False, threads=10)
            
            if len(batch) == 1:
                df_batch.columns = [c.lower() for c in df_batch.columns]
                try:
                    df_batch.index = df_batch.index.tz_convert('America/New_York').tz_localize(None)
                except: pass
                data_60d[batch[0]] = df_batch
            else:
                for sym in batch:
                    if sym in df_batch.columns.levels[0]:
                        df_sym = df_batch[sym].dropna(how='all').copy()
                        if not df_sym.empty:
                            df_sym.columns = [c.lower() for c in df_sym.columns]
                            try:
                                df_sym.index = df_sym.index.tz_convert('America/New_York').tz_localize(None)
                            except: pass
                            data_60d[sym] = df_sym
                            
        with open(output_pkl, 'wb') as f:
            pickle.dump(data_60d, f)
            
    print(f"Loaded {len(data_60d)} symbols with data.")
    
    sample_sym = None
    for sym, df_sym in data_60d.items():
        if len(df_sym) > 3000:
            sample_sym = sym
            break
            
    if not sample_sym: 
        print("Could not find a symbol with enough data.")
        return
        
    all_dates = sorted(list(set(data_60d[sample_sym].index.date)))
    print(f"Total trading days fetched: {len(all_dates)} (Min: {all_dates[0]}, Max: {all_dates[-1]})")
    
    n_days = len(all_dates)
    if n_days < 20: 
        print("Not enough days for analysis!")
        return
        
    # PRODUCTION MODE: Use the most recent 40 days as training data
    train_days_len = min(40, n_days)
    train_dates = all_dates[-train_days_len:]  # Most recent 40 days
    
    print(f"\n--- PRODUCTION MODE ---")
    print(f"Training on the most recent {len(train_dates)} days ({train_dates[0]} to {train_dates[-1]})")
    print(f"Next week's live trading will use the Top 10 from this analysis.")
    
    core_params = {
        "entry_start_time": "09:35:00", "entry_end_time": "10:30:00", 
        "take_profit_pct": 0.10, "stop_loss_pct": 0.03,  
        "min_volume_ratio": 1.0, "use_historical_rvol": False
    }
    ROUND_TRIP = COMM_RATE * 2
    
    train_stats = []
    
    print("\n--- PHASE 1: FEATURE EXTRACTION ---")
    for sym, df in tqdm(data_60d.items(), desc="Analyzing stocks"):
        train_df = df.loc[str(train_dates[0]):str(train_dates[-1])]
        if len(train_df) < 100: continue
        
        adr = calculate_adr(train_df)
        _, trades = run_backtest(train_df, core_params)
        
        if len(trades) > 0:
            net_pnl = trades['pnl_pct'].sum() - (len(trades) * ROUND_TRIP)
            wins = trades[(trades['pnl_pct'] - ROUND_TRIP) > 0]
            win_rate = len(wins) / len(trades)
            avg_win = wins['pnl_pct'].mean() if len(wins) > 0 else 0
            
            train_stats.append({
                'Symbol': sym, 'ADR': adr, 'Trades': len(trades),
                'WinRate': win_rate, 'AvgWin': avg_win, 'TotalPnL': net_pnl
            })
            
    df_train = pd.DataFrame(train_stats).fillna(0)
    print(f"Extracted features for {len(df_train)} stocks.")
    
    print("\n--- PHASE 2: PRE-FILTERING & PCA SCORES ---")
    # Apply optimal pre-filter: ADR >= 6%, WinRate >= 45%, Trades >= 5
    filtered_df = df_train[(df_train['ADR'] >= 0.06) & 
                           (df_train['WinRate'] >= 0.45) & 
                           (df_train['Trades'] >= 5)].copy()
                           
    print(f"Stocks passing pre-filter (ADR>=6%, WinRate>=45%, Trades>=5): {len(filtered_df)}")
    
    if len(filtered_df) < 10:
        print("Warning: Less than 10 stocks passed the filter. Using remaining stocks.")
        if len(filtered_df) == 0:
            print("Fallback: Reverting to no filter.")
            filtered_df = df_train.copy()
            
    # Apply PCA (PC1 Ranking)
    features = ['ADR', 'Trades', 'WinRate', 'AvgWin', 'TotalPnL']
    X = filtered_df[features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=min(3, len(filtered_df)-1))
    pca.fit(X_scaled)
    
    # PC1 Score (ADR/AvgWin Heavy)
    scores = X_scaled.dot(pca.components_[0])
    filtered_df['Score'] = scores
    filtered_df.sort_values('Score', ascending=False, inplace=True)
    filtered_df['Rank'] = np.arange(1, len(filtered_df) + 1)
    
    # Select Top 10
    pool_size = 10
    top_10 = filtered_df.head(pool_size)
    
    print("\n==================================================")
    print(f"STALLION TOP {len(top_10)} WATCHLIST FOR NEXT WEEK")
    print("==================================================")
    print(top_10[['Symbol', 'ADR', 'Trades', 'WinRate', 'AvgWin', 'Score']].to_string(index=False))
    print("==================================================")

    # Save the Top 10 to a file for the live trader
    top_symbols = top_10['Symbol'].tolist()
    with open('top_10_watchlist.json', 'w') as f:
        json.dump(top_symbols, f)
    print(f"\nSaved Top 10 watchlist for the week: {top_symbols}")
    print("Ready for Monday trading.")

if __name__ == '__main__':
    main()
