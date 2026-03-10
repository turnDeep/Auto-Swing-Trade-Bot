import os
import pickle
import json
import pandas as pd
import numpy as np

def calculate_ex_ante_features(df_train):
    """
    Calculates the ex-ante features for a single stock over the training window.
    Features: ADR, ATRPct, ADV, Opening RVOL, AbsGap
    """
    if df_train.empty:
        return None
        
    dates = sorted(list(set(d.date() for d in df_train.index)))
    if len(dates) < 5:
        return None
        
    daily = df_train.resample('D').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna()
    if len(daily) < 5:
        return None
        
    # ADR & ATR%
    daily['range_pct'] = (daily['high'] - daily['low']) / daily['low']
    adr = daily['range_pct'].mean()
    
    daily['true_range'] = np.maximum(daily['high'] - daily['low'], 
                                   np.maximum(abs(daily['high'] - daily['close'].shift(1)), 
                                              abs(daily['low'] - daily['close'].shift(1))))
    daily['atr_pct'] = daily['true_range'] / daily['close']
    atr_pct = daily['atr_pct'].mean()
    
    # ADV
    adv = daily['volume'].mean()
    
    # Opening RVOL & Gap (approximate based on daily aggregation)
    daily['gap_pct'] = abs((daily['open'] - daily['close'].shift(1)) / daily['close'].shift(1))
    avg_gap = daily['gap_pct'].mean()
    
    return {
        'ADR': adr,
        'ATRPct': atr_pct,
        'ADV': adv,
        'AvgGap': avg_gap
    }

def generate_watchlist(data_60d, train_dates, top_n=10):
    """
    Generates a scored watchlist based on Z-Scores of Ex-Ante features.
    """
    stats_list = []
    
    for sym, df in data_60d.items():
        # Slice to training window
        df_train = df.loc[str(train_dates[0]):str(train_dates[-1])]
        features = calculate_ex_ante_features(df_train)
        
        if features and features['ADR'] >= 0.05: # Minimum 5% ADR filter
            features['Symbol'] = sym
            stats_list.append(features)
            
    if not stats_list:
        return []
        
    df_stats = pd.DataFrame(stats_list)
    
    # Z-Score Normalization
    for col in ['ADR', 'ATRPct', 'ADV', 'AvgGap']:
        mean = df_stats[col].mean()
        std = df_stats[col].std()
        if std > 0:
            df_stats[f'{col}_Z'] = (df_stats[col] - mean) / std
        else:
            df_stats[f'{col}_Z'] = 0.0
            
    # Composite Score (Equal weight for Volatility, Liquidity, and Gap presence)
    df_stats['Composite_Score'] = (df_stats['ADR_Z'] * 0.3) + (df_stats['ATRPct_Z'] * 0.3) + \
                                  (df_stats['ADV_Z'] * 0.2) + (df_stats['AvgGap_Z'] * 0.2)
                                  
    df_stats.sort_values('Composite_Score', ascending=False, inplace=True)
    df_stats['Rank'] = np.arange(1, len(df_stats) + 1)
    
    return df_stats.head(top_n)

def main():
    print("Loading universe symbols...")
    data_pkl = 'russell3000_60d_5min.pkl'
    
    if not os.path.exists(data_pkl):
        print(f"Error: {data_pkl} not found. Please run the data collector.")
        return
        
    with open(data_pkl, 'rb') as f:
        data_60d = pickle.load(f)
        
    print(f"Loaded {len(data_60d)} symbols with data.")
    
    # Identify training dates (Most recent 40 days)
    sample_sym = list(data_60d.keys())[0]
    all_dates = sorted(list(set(data_60d[sample_sym].index.date)))
    train_dates = all_dates[-min(40, len(all_dates)):]
    
    print(f"\n--- EX-ANTE WATCHLIST GENERATION ---")
    print(f"Training on the most recent {len(train_dates)} days ({train_dates[0]} to {train_dates[-1]})")
    
    top_10_df = generate_watchlist(data_60d, train_dates, top_n=10)
    
    if top_10_df.empty:
        print("No stocks passed the filters.")
        return
        
    print("\n==================================================")
    print(f"STALLION TOP 10 WATCHLIST FOR NEXT WEEK")
    print("==================================================")
    print(top_10_df[['Symbol', 'ADR', 'ATRPct', 'ADV', 'Composite_Score']].to_string(index=False))
    print("==================================================")
    
    # Save the Top 10 to a file for the live trader
    top_symbols = top_10_df['Symbol'].tolist()
    with open('top_10_watchlist.json', 'w') as f:
        json.dump(top_symbols, f)
        
    print(f"\nSaved Top 10 watchlist for the week: {top_symbols}")
    print("Ready for live trading execution.")

if __name__ == '__main__':
    main()
