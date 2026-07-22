import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

st.set_page_config(page_title="Demand Forecasting", layout="wide")

# %% ---------------------------------------------------------
# Load model, features, and historical data (cached so it loads once)
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    model = joblib.load('outputs/xgb_model.pkl')
    feature_cols = joblib.load('outputs/feature_cols.pkl')
    return model, feature_cols

@st.cache_data
def load_data():
    df = pd.read_csv('data/train_features.csv', parse_dates=['date'])
    return df

model, feature_cols = load_model()
df = load_data()

# %% ---------------------------------------------------------
# Sidebar — user inputs
# ---------------------------------------------------------
st.sidebar.header("Select Store & Item")
store = st.sidebar.selectbox("Store", sorted(df['store'].unique()))
item = st.sidebar.selectbox("Item", sorted(df['item'].unique()))

st.title("📈 Demand Forecasting Dashboard")
st.markdown(f"**Store {store} | Item {item}** — XGBoost-based sales prediction")

# %% ---------------------------------------------------------
# Filter history for this store-item pair
# ---------------------------------------------------------
history = df[(df['store'] == store) & (df['item'] == item)].sort_values('date')

if history.empty:
    st.warning("No data for this combination.")
    st.stop()

# %% ---------------------------------------------------------
# Predict on the most recent available row (simulates "next day" prediction)
# ---------------------------------------------------------
latest_row = history.iloc[[-1]]
X_latest = latest_row[feature_cols]
prediction = model.predict(X_latest)[0]

col1, col2, col3 = st.columns(3)
col1.metric("Last Actual Sales", f"{latest_row['sales'].values[0]:.0f} units")
col2.metric("Predicted Next-Day Sales", f"{prediction:.0f} units")
col3.metric("Historical Avg (this store-item)", f"{history['sales'].mean():.0f} units")

# %% ---------------------------------------------------------
# Chart: recent actual sales trend
# ---------------------------------------------------------
st.subheader("Recent Sales Trend (last 90 days)")
recent = history.tail(90)

fig = go.Figure()

# Actual Sales
fig.add_trace(
    go.Scatter(
        x=recent["date"],
        y=recent["sales"],
        mode="lines",
        name="Actual Sales",
        line=dict(width=3)
    )
)

# Predicted Point
prediction_date = latest_row["date"].iloc[0] + pd.Timedelta(days=1)

fig.add_trace(
    go.Scatter(
        x=[prediction_date],
        y=[prediction],
        mode="markers",
        name="Prediction",
        marker=dict(size=12, color="red", symbol="diamond")
    )
)

fig.update_layout(
    title="Last 90 Days Sales",
    xaxis_title="Date",
    yaxis_title="Sales",
    hovermode="x unified",
    template="plotly_white",
    height=500
)

st.plotly_chart(fig, use_container_width=True)

# %% ---------------------------------------------------------
# Feature importance panel
# ---------------------------------------------------------
st.subheader("What drives this prediction?")
importance = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)

fig2, ax2 = plt.subplots(figsize=(8, 5))
sns.barplot(x=importance.values[:8], y=importance.index[:8], orient='h', ax=ax2)
ax2.set_title("Top 8 Feature Importances (Model-wide)")
st.pyplot(fig2)

# %% ---------------------------------------------------------
# Raw data table (optional, collapsible)
# ---------------------------------------------------------
with st.expander("View raw historical data for this store-item"):
    st.dataframe(history[['date', 'sales'] + feature_cols].tail(30))