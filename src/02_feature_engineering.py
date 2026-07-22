# %%
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')

# %% ---------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------
df = pd.read_csv('data/train.csv', parse_dates=['date'])
df = df.sort_values(['store', 'item', 'date']).reset_index(drop=True)

# %% ---------------------------------------------------------
# 2. Calendar features
# ---------------------------------------------------------
df['day_of_week'] = df['date'].dt.dayofweek       # 0=Monday, 6=Sunday
df['month']       = df['date'].dt.month
df['year']        = df['date'].dt.year
df['day_of_year'] = df['date'].dt.dayofyear
df['is_weekend']  = (df['day_of_week'] >= 5).astype(int)

# %% ---------------------------------------------------------
# 3. Lag features — MUST group by store & item to avoid leakage
# ---------------------------------------------------------
grouped = df.groupby(['store', 'item'])['sales']

df['sales_lag_7']  = grouped.shift(7)
df['sales_lag_14'] = grouped.shift(14)
df['sales_lag_28'] = grouped.shift(28)

# %% ---------------------------------------------------------
# 4. Rolling window features — also grouped, and shifted by 1
#    so we don't include the current day's own sales in its own rolling stat
# ---------------------------------------------------------
df['rolling_mean_7']  = grouped.shift(1).rolling(window=7).mean()
df['rolling_mean_30'] = grouped.shift(1).rolling(window=30).mean()
df['rolling_std_7']   = grouped.shift(1).rolling(window=7).std()

# NOTE: the groupby().shift().rolling() chain above works on the flat series,
# but rolling() by itself doesn't respect group boundaries unless applied
# through groupby().rolling(). Safer, explicit version below:

df['rolling_mean_7'] = (
    df.groupby(['store', 'item'])['sales']
      .transform(lambda x: x.shift(1).rolling(window=7).mean())
)
df['rolling_mean_30'] = (
    df.groupby(['store', 'item'])['sales']
      .transform(lambda x: x.shift(1).rolling(window=30).mean())
)
df['rolling_std_7'] = (
    df.groupby(['store', 'item'])['sales']
      .transform(lambda x: x.shift(1).rolling(window=7).std())
)

# %% ---------------------------------------------------------
# 5. Store/item aggregate features (computed from full history — fine since
#    these are just identity-level averages, not time-leaking)
# ---------------------------------------------------------
store_avg = df.groupby('store')['sales'].mean().rename('store_avg_sales')
item_avg  = df.groupby('item')['sales'].mean().rename('item_avg_sales')

df = df.merge(store_avg, on='store', how='left')
df = df.merge(item_avg, on='item', how='left')

# %% ---------------------------------------------------------
# 6. Drop rows with NaN (from lag_28 / rolling_30 needing 30 days history)
# ---------------------------------------------------------
print("Rows before dropping NaN:", len(df))
df = df.dropna().reset_index(drop=True)
print("Rows after dropping NaN:", len(df))

# %% ---------------------------------------------------------
# 7. Save processed dataset
# ---------------------------------------------------------
df.to_csv('data/train_features.csv', index=False)
print("Saved to data/train_features.csv")
print(df.head())
print("\nFinal columns:", list(df.columns))