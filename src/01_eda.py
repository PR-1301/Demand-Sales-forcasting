# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("darkgrid")

# %% ---------------------------------------------------------
# 1. Load and parse dates
# ---------------------------------------------------------
df = pd.read_csv("data/train.csv", parse_dates=["date"])

print("Shape:", df.shape)
print("\nInfo:")
df.info()
print("\nDescribe:")
print(df.describe())
print("\nNull counts:\n", df.isnull().sum())

# %% ---------------------------------------------------------
# 2. Date range and missing-date check
# ---------------------------------------------------------
print("Date range:", df["date"].min(), "to", df["date"].max())

full_range = pd.date_range(start=df["date"].min(), end=df["date"].max(), freq="D")
actual_dates = df["date"].sort_values().unique()
missing_dates = set(full_range) - set(actual_dates)
print("Number of missing calendar dates:", len(missing_dates))

# Also confirm every (store, item) pair has the same number of rows
counts = df.groupby(["store", "item"]).size()
print("\nRows per store-item pair — min:", counts.min(), "max:", counts.max())
print("Unique stores:", df["store"].nunique(), "| Unique items:", df["item"].nunique())

# %% ---------------------------------------------------------
# 3. Total daily sales over 5 years
# ---------------------------------------------------------
daily_total = df.groupby("date")["sales"].sum()

plt.figure(figsize=(14, 5))
plt.plot(daily_total.index, daily_total.values, linewidth=0.8)
plt.title("Total Daily Sales Across All Stores & Items (2013–2017)")
plt.xlabel("Date")
plt.ylabel("Total Units Sold")
plt.tight_layout()
plt.savefig("outputs/total_daily_sales.png", dpi=150)
plt.show()

# %% ---------------------------------------------------------
# 4. One store-item pair — inspect weekly pattern
# ---------------------------------------------------------
sample = df[(df["store"] == 1) & (df["item"] == 1)].sort_values("date")

plt.figure(figsize=(14, 5))
plt.plot(sample["date"], sample["sales"], linewidth=0.8)
plt.title("Daily Sales — Store 1, Item 1")
plt.xlabel("Date")
plt.ylabel("Units Sold")
plt.tight_layout()
plt.savefig("outputs/store1_item1_sales.png", dpi=150)
plt.show()

# Zoom into 2 months to actually see weekly cycle clearly
zoom = sample[(sample["date"] >= "2017-01-01") & (sample["date"] <= "2017-02-28")]
plt.figure(figsize=(14, 5))
plt.plot(zoom["date"], zoom["sales"], marker="o", linewidth=1)
plt.title("Store 1, Item 1 — Jan–Feb 2017 (zoomed, look for weekly cycle)")
plt.xlabel("Date")
plt.ylabel("Units Sold")
plt.tight_layout()
plt.savefig("outputs/store1_item1_zoom.png", dpi=150)
plt.show()

# %% ---------------------------------------------------------
# 5. Sales distribution — skew, zeros, outliers
# ---------------------------------------------------------
plt.figure(figsize=(10, 5))
sns.histplot(df["sales"], bins=50, kde=True)
plt.title("Distribution of Sales Values")
plt.xlabel("Sales (units)")
plt.tight_layout()
plt.savefig("outputs/sales_distribution.png", dpi=150)
plt.show()

print("\nSkewness:", df["sales"].skew())
print(
    "Zero-sales rows:",
    (df["sales"] == 0).sum(),
    f"({(df['sales'] == 0).mean()*100:.2f}%)",
)
print("Sales stats:\n", df["sales"].describe())

# Check for extreme outliers using IQR
Q1, Q3 = df["sales"].quantile([0.25, 0.75])
IQR = Q3 - Q1
upper_bound = Q3 + 1.5 * IQR
outliers = df[df["sales"] > upper_bound]
print(
    f"\nOutliers above {upper_bound:.1f}: {len(outliers)} rows ({len(outliers)/len(df)*100:.2f}%)"
)
