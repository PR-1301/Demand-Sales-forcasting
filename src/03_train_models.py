# %%
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

# %% ---------------------------------------------------------
# 1. Load engineered features
# ---------------------------------------------------------
df = pd.read_csv('data/train_features.csv', parse_dates=['date'])

# %% ---------------------------------------------------------
# 2. Time-based split
# ---------------------------------------------------------
split_date = '2017-10-01'
train = df[df['date'] < split_date]
val   = df[df['date'] >= split_date]

print("Train rows:", len(train), "| Val rows:", len(val))
print("Train date range:", train['date'].min(), "to", train['date'].max())
print("Val date range:", val['date'].min(), "to", val['date'].max())

# %% ---------------------------------------------------------
# 3. Define X (features) and y (target)
# ---------------------------------------------------------
feature_cols = [
    'store', 'item', 'day_of_week', 'month', 'year', 'day_of_year', 'is_weekend',
    'sales_lag_7', 'sales_lag_14', 'sales_lag_28',
    'rolling_mean_7', 'rolling_mean_30', 'rolling_std_7',
    'store_avg_sales', 'item_avg_sales'
]
target_col = 'sales'

X_train, y_train = train[feature_cols], train[target_col]
X_val, y_val     = val[feature_cols], val[target_col]

# %% ---------------------------------------------------------
# 4. Helper to evaluate any model consistently
# ---------------------------------------------------------
def evaluate(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true.replace(0, np.nan))) * 100
    print(f"{name:20s} | RMSE: {rmse:7.3f} | MAE: {mae:7.3f} | MAPE: {mape:6.2f}%")
    return {'model': name, 'rmse': rmse, 'mae': mae, 'mape': mape}

results = []

# %% ---------------------------------------------------------
# 5. Baseline: naive forecast (predict same as 7 days ago)
# ---------------------------------------------------------
naive_pred = X_val['sales_lag_7']
results.append(evaluate("Naive (lag_7)", y_val, naive_pred))

# %% ---------------------------------------------------------
# 6. Linear Regression
# ---------------------------------------------------------
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_val)
results.append(evaluate("Linear Regression", y_val, lr_pred))

# %% ---------------------------------------------------------
# 7. Random Forest
# ---------------------------------------------------------
rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=12,
    random_state=42,
    n_jobs=-1          # use all CPU cores
)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_val)
results.append(evaluate("Random Forest", y_val, rf_pred))

# %% ---------------------------------------------------------
# 8. XGBoost
# ---------------------------------------------------------
xgb = XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_val)
results.append(evaluate("XGBoost", y_val, xgb_pred))

# %% ---------------------------------------------------------
# 9. Comparison table
# ---------------------------------------------------------
results_df = pd.DataFrame(results).sort_values('rmse')
print("\n=== Final Comparison (sorted by RMSE, lower is better) ===")
print(results_df.to_string(index=False))
results_df.to_csv('outputs/model_comparison.csv', index=False)