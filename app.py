import pandas as pd
import numpy as np
import streamlit as st

# 1. Page Configuration
st.set_page_config(
    page_title="Streamlit Boilerplate",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 2. Sidebar Navigation and Inputs
st.sidebar.header("Configuration")
user_name = st.sidebar.text_input("Enter your name:", "Developer")
data_points = st.sidebar.slider("Number of data points:", 10, 100, 50)

# 3. Main Header
st.title(f"Welcome back, {user_name}! 👋")
st.write("This is a basic boilerplate template for your Streamlit application.")

# 4. Interactive Components & Logic
st.header("Interactive Data Analysis")

# Generate mock data based on sidebar input
chart_data = pd.DataFrame(
    np.random.randn(data_points, 3), columns=["Metric A", "Metric B", "Metric C"]
)

# Create layout columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("Data Summary")
    st.dataframe(chart_data.head())  # Interactive table

with col2:
    st.subheader("Visual Trend")
    st.line_chart(chart_data)  # Interactive chart

# 5. Status Indicators
st.success("App loaded successfully!")
