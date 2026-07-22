# %%
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor

sns.set_style("darkgrid")

# %% ---------------------------------------------------------
# 1. Reload data and retrain XGBoost (best model) —
#    keeping this script standalone/reproducible
# ---------------------------------------------------------
df = pd.read_csv("data/train_features.csv", parse_dates=["date"])

split_date = "2017-10-01"
train = df[df["date"] < split_date]
val = df[df["date"] >= split_date]

feature_cols = [
    "store",
    "item",
    "day_of_week",
    "month",
    "year",
    "day_of_year",
    "is_weekend",
    "sales_lag_7",
    "sales_lag_14",
    "sales_lag_28",
    "rolling_mean_7",
    "rolling_mean_30",
    "rolling_std_7",
    "store_avg_sales",
    "item_avg_sales",
]
target_col = "sales"

X_train, y_train = train[feature_cols], train[target_col]
X_val, y_val = val[feature_cols], val[target_col]

xgb = XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
)
xgb.fit(X_train, y_train)
val_pred = xgb.predict(X_val)

# %% ---------------------------------------------------------
# 2. Feature importance
# ---------------------------------------------------------
importance = pd.Series(xgb.feature_importances_, index=feature_cols).sort_values(
    ascending=False
)
print("Feature importances:\n", importance)

plt.figure(figsize=(10, 6))
sns.barplot(x=importance.values, y=importance.index, orient="h")
plt.title("XGBoost Feature Importance")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("outputs/feature_importance.png", dpi=150)
plt.show()

# %% ---------------------------------------------------------
# 3. Predicted vs Actual — scatter (overall fit quality)
# ---------------------------------------------------------
plt.figure(figsize=(7, 7))
plt.scatter(y_val, val_pred, alpha=0.15, s=8)
max_val = max(y_val.max(), val_pred.max())
plt.plot(
    [0, max_val], [0, max_val], color="red", linewidth=1.5, label="Perfect prediction"
)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("XGBoost — Predicted vs Actual (Validation Set)")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/pred_vs_actual_scatter.png", dpi=150)
plt.show()

# %% ---------------------------------------------------------
# 4. Predicted vs Actual — time series for one store-item pair
#    (much more intuitive for a report/viva than a scatter plot)
# ---------------------------------------------------------
val = val.copy()
val["prediction"] = val_pred

sample = val[(val["store"] == 1) & (val["item"] == 1)].sort_values("date")

plt.figure(figsize=(14, 5))
plt.plot(sample["date"], sample["sales"], label="Actual", linewidth=1.5)
plt.plot(
    sample["date"],
    sample["prediction"],
    label="Predicted",
    linewidth=1.5,
    linestyle="--",
)
plt.title("Store 1, Item 1 — Actual vs Predicted (Oct–Dec 2017)")
plt.xlabel("Date")
plt.ylabel("Units Sold")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/pred_vs_actual_timeseries.png", dpi=150)
plt.show()

# %% ---------------------------------------------------------
# 5. Residuals (errors) distribution — check if model is biased
# ---------------------------------------------------------
residuals = y_val.values - val_pred

plt.figure(figsize=(10, 5))
sns.histplot(residuals, bins=50, kde=True)
plt.axvline(0, color="red", linestyle="--")
plt.title("Residuals Distribution (Actual - Predicted)")
plt.xlabel("Residual")
plt.tight_layout()
plt.savefig("outputs/residuals_distribution.png", dpi=150)
plt.show()

print("\nMean residual:", residuals.mean())  # close to 0 = no systematic bias
print("Residual std:", residuals.std())
