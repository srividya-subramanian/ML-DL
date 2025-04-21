# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 15:48:43 2025
ARIMA is great for linear trends, Prophet handles seasonality well, and LSTM captures nonlinearities.

@author: srivi
"""

## pip install yfinance, statsmodels, prophet

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Download NIFTY 50 historical data
nifty = yf.download("^NSEI", start="2000-01-01", end="2024-12-31")
nifty = nifty[['Close']].dropna()
nifty.columns = ['NIFTY50']

nifty.plot(title="NIFTY 50 Index")
plt.show()

from statsmodels.tsa.arima.model import ARIMA

# Fit ARIMA model (order can be tuned)
model = ARIMA(nifty, order=(5, 1, 0))
model_fit = model.fit()
forecast_arima = model_fit.forecast(steps=30)

# Plot
plt.figure(figsize=(10, 4))
plt.plot(nifty, label="Historical")
plt.plot(forecast_arima.index, forecast_arima, label="ARIMA Forecast", color='red')
plt.title("NIFTY 50 Forecast (ARIMA)")
plt.legend()
plt.show()


from prophet import Prophet

# Format for Prophet
df_prophet = nifty.reset_index()
df_prophet.columns = ['ds', 'y']

model_prophet = Prophet(daily_seasonality=True)
model_prophet.fit(df_prophet)

future = model_prophet.make_future_dataframe(periods=30)
forecast = model_prophet.predict(future)

model_prophet.plot(forecast)
plt.title("NIFTY 50 Forecast (Facebook Prophet)")
plt.show()


import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Normalize data
scaler = MinMaxScaler()
nifty_scaled = scaler.fit_transform(nifty)

# Create sequences
def create_sequences(data, seq_len):
    x, y = [], []
    for i in range(seq_len, len(data)):
        x.append(data[i-seq_len:i])
        y.append(data[i])
    return np.array(x), np.array(y)

seq_len = 60
x, y = create_sequences(nifty_scaled, seq_len)

# Train/Test split
train_size = int(0.8 * len(x))
x_train, x_test = x[:train_size], x[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Build LSTM Model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(seq_len, 1)))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=10, batch_size=32)

# Predict
predictions = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predictions)
actual_prices = scaler.inverse_transform(y_test)

# Plot
plt.plot(actual_prices, label='Actual')
plt.plot(predicted_prices, label='LSTM Predicted')
plt.title("NIFTY 50 Forecast (LSTM)")
plt.legend()
plt.show()























