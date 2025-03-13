import streamlit as st
import pandas as pd

import subprocess
import sys

try:
    import joblib
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "joblib"])
    import joblib

    
import joblib
import matplotlib.pyplot as plt
import pickle

# Load the model
model = joblib.load('arima_model.pkl')

# Load data
df = pd.read_csv('dataset/Electric_Production.csv')
df = df.rename(columns={"IPG2211A2N":"temp"})

# Streamlit app
st.title('ARIMA Time Series Forecasting')

# Plot actual data
st.line_chart(df['temp'])

# Forecasting section
steps = st.number_input("Forecast Steps", min_value=1, max_value=200, value=30)

if st.button("Generate Forecast"):
    forecast = model.predict(start=len(df), end=len(df) + steps - 1)
    
    # Plot forecast
    fig, ax = plt.subplots()
    ax.plot(df['temp'], label="Actual", color="blue")
    ax.plot(forecast, label="Predicted", color="red")
    ax.legend()
    st.pyplot(fig)
