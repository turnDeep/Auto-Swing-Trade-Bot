import pandas as pd
from ml_pipeline_60d import calculate_adr
import numpy as np

# Create dummy data with a spike 60 days ago
dates = pd.date_range(end=pd.Timestamp.today(), periods=60, freq='D')
df = pd.DataFrame(index=dates)
df['open'] = 10.0
df['low'] = 10.0
df['high'] = 11.0 # 10% ADR normally
df['close'] = 11.0

# Add a massive spike 50 days ago (10th day)
df.loc[dates[10], 'high'] = 30.0 # 200% ADR

# Add some intra-day records to make it resample nicely
df_intra = df.reindex(pd.date_range(end=pd.Timestamp.today(), periods=60*24, freq='h'), method='pad')

adr_val = calculate_adr(df_intra)
print(f'New EMA style ADR: {adr_val:.4f}')
print(f'Expected ~0.10, definitely much less than simple mean which would be ~0.13')

# Check simple mean
daily = df_intra.resample('D').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
daily['adr'] = (daily['high'] - daily['low']) / daily['low']
simple_mean = daily['adr'].mean()
print(f'Simple mean ADR: {simple_mean:.4f}')
