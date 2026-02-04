### Gold prediction

import numpy as np
import pandas as pd
import yfinance as yf

df = yf.download("GLD", period="5y")
# df = df[['Close', 'Volume']]
#
# df['Return'] = df['Close'].pct_change()
# df['MA_10'] = df['Close'].rolling(10).mean()
# df['MA_30'] = df['Close'].rolling(30).mean()
# df['Volatility'] = df['Return'].rolling(10).std()
#
# # Target: tomorrow up or down
# df['Target'] = (df['Return'].shift(-1) > 0).astype(int)
#
# df = df.dropna()


# Add reasonable financial features: These might contain weak signal.

df['Return_1d'] = df['Close'].pct_change()
df['Return_5d'] = df['Close'].pct_change(5)
df['MA_10'] = df['Close'].rolling(10).mean()
df['MA_30'] = df['Close'].rolling(30).mean()
df['MA_ratio'] = df['MA_10'] / df['MA_30']

df['Volatility_10'] = df['Return_1d'].rolling(10).std()
df['Volatility_30'] = df['Return_1d'].rolling(30).std()

df['Volume_Change'] = df['Volume'].pct_change()



# Add noisy features
for lag in range(2, 20):   # many lags = mostly junk
    df[f'Return_lag_{lag}'] = df['Return_1d'].shift(lag)

df['Vol_MA_5'] = df['Volume'].rolling(5).mean()
df['Vol_MA_20'] = df['Volume'].rolling(20).mean()


# Prediction target: tomorrow return
df['Target'] = df['Return_1d'].shift(-1)  # predict
df = df.dropna()


# Prepare dataset for ML

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

features = df.drop(columns=['Target'])
X = features.values
y = df['Target'].values

split = int(len(df) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)







