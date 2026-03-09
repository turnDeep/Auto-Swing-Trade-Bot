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
ROUND_TRIP = COMM_RATE * 2

def calculate_adr(df):
    """Simple ADR calculation (proven optimal for PCA PC1)"""
    daily = df.resample('D').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
    if len(daily) < 2: return 0
    daily['adr'] = (daily['high'] - daily['low']) / daily['low']
    return daily['adr'].mean()

def get_stats(df, core_params):
    if len(df) < 5:
        return {'Trades': 0, 'WinRate': 0, 'AvgWin': 0, 'TotalPnL': 0, 'ADR': 0}
    adr = calculate_adr(df)
    _, trades = run_backtest(df, core_params)
    
    if len(trades) > 0:
        net_pnl = trades['pnl_pct'].sum() - (len(trades) * ROUND_TRIP)
        wins = trades[(trades['pnl_pct'] - ROUND_TRIP) > 0]
        win_rate = len(wins) / len(trades)
        avg_win = wins['pnl_pct'].mean() if len(wins) > 0 else 0
        return {'Trades': len(trades), 'WinRate': win_rate, 'AvgWin': avg_win, 'TotalPnL': net_pnl, 'ADR': adr}
    else:
        return {'Trades': 0, 'WinRate': 0, 'AvgWin': 0, 'TotalPnL': 0, 'ADR': adr}

def compute_pca_scores(df_stats):
    if len(df_stats) < 2:
        return np.zeros(len(df_stats))
    features = df_stats[['Trades', 'WinRate', 'AvgWin', 'TotalPnL', 'ADR']].fillna(0)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    pca = PCA(n_components=1)
    scores = pca.fit_transform(scaled)[:, 0]
    
    # Sign check: ensure ADR and WinRate generally correlate positively with the score
    corrs = [df_stats['ADR'].corr(pd.Series(scores)), df_stats['WinRate'].corr(pd.Series(scores))]
    valid_corrs = [c for c in corrs if not np.isnan(c)]
    if sum(valid_corrs) < 0:
        scores = -scores
        
    # Standardize output scores so that weight mixing makes mathematical sense
    scores = (scores - scores.mean()) / (scores.std() + 1e-9)
    return scores
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
    
    print("\n--- PHASE 1: FEATURE EXTRACTION (MULTI-PERIOD) ---")
    train_stats = { '40': {}, '20': {}, '10': {}, '5': {} }
    
    for sym, df in tqdm(data_60d.items(), desc="Analyzing stocks"):
        df_train_40 = df.loc[str(train_dates[0]):str(train_dates[-1])]
        if len(df_train_40) < 100: continue
        dates_in = sorted(list(set(d.date() for d in df_train_40.index)))
        if len(dates_in) == 0: continue
        
        train_stats['40'][sym] = get_stats(df_train_40, core_params)
        
        d20 = dates_in[-20:]
        df_20 = df_train_40.loc[str(d20[0]):str(d20[-1])] if len(d20) > 0 else df_train_40.iloc[0:0]
        train_stats['20'][sym] = get_stats(df_20, core_params)
        
        d10 = dates_in[-10:]
        df_10 = df_train_40.loc[str(d10[0]):str(d10[-1])] if len(d10) > 0 else df_train_40.iloc[0:0]
        train_stats['10'][sym] = get_stats(df_10, core_params)
        
        d5 = dates_in[-5:]
        df_5 = df_train_40.loc[str(d5[0]):str(d5[-1])] if len(d5) > 0 else df_train_40.iloc[0:0]
        train_stats['5'][sym] = get_stats(df_5, core_params)
        
    print("\n--- PHASE 2: PRE-FILTERING & TIME-WEIGHTED PCA ---")
    # Base filter applied to 40-day metrics
    passed_syms = []
    base_stats = []
    for sym, s in train_stats['40'].items():
        if s['ADR'] >= 0.06 and s['WinRate'] >= 0.45 and s['Trades'] >= 5:
            passed_syms.append(sym)
            base_stats.append({
                'Symbol': sym, 'ADR': s['ADR'], 'Trades': s['Trades'],
                'WinRate': s['WinRate'], 'AvgWin': s['AvgWin'], 'TotalPnL': s['TotalPnL']
            })
            
    print(f"Stocks passing base 40d filter (ADR>=6%, WinRate>=45%, Trades>=5): {len(passed_syms)}")
    
    if len(passed_syms) < 10:
        print("Warning: Less than 10 stocks passed the filter. Proceeding anyway.")
        
    df_base = pd.DataFrame(base_stats)
    
    if len(passed_syms) > 0:
        # Time-weighted PCA config matching the optimized result: 5d:40%, 10d:30%, 20d:20%, 40d:10%
        weights = {'5': 0.40, '10': 0.30, '20': 0.20, '40': 0.10}
        final_scores = np.zeros(len(passed_syms))
        
        for period, weight in weights.items():
            period_stats = []
            for sym in passed_syms:
                s = train_stats[period][sym]
                period_stats.append({
                    'Symbol': sym, 'ADR': s['ADR'], 'Trades': s['Trades'],
                    'WinRate': s['WinRate'], 'AvgWin': s['AvgWin'], 'TotalPnL': s['TotalPnL']
                })
            df_period = pd.DataFrame(period_stats)
            scores = compute_pca_scores(df_period)
            final_scores += (scores * weight)
            
        df_base['Score'] = final_scores
        df_base.sort_values('Score', ascending=False, inplace=True)
        df_base['Rank'] = np.arange(1, len(df_base) + 1)
        
    # Select Top 10
    pool_size = 10
    top_10 = df_base.head(pool_size) if not df_base.empty else df_base
    
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
