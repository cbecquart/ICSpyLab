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
df['Return_20d'] = df['Close'].pct_change(20)


df['MA_10'] = df['Close'].rolling(10).mean()
df['MA_30'] = df['Close'].rolling(30).mean()
df['MA_ratio'] = df['MA_10'] / df['MA_30']

df['Volatility_10'] = df['Return_1d'].rolling(10).std()
df['Volatility_30'] = df['Return_1d'].rolling(30).std()

df['Volume_Change_1d'] = df['Volume'].pct_change()
df['Volume_Change_5d'] = df['Volume'].pct_change(5)
df['Volume_Change_20d'] = df['Volume'].pct_change(20)


# Season features
df['Month'] = df.index.month
df['DayOfYear'] = df.index.dayofyear
df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
df['DOY_sin'] = np.sin(2 * np.pi * df['DayOfYear'] / 365)
df['DOY_cos'] = np.cos(2 * np.pi * df['DayOfYear'] / 365)


# Trend features
close = df['Close'].squeeze()
for w in [3, 5, 7, 12, 18, 25, 40, 60]:
    ma = close.rolling(w).mean()
    df[f'MA_{w}'] = ma
    df[f'Dist_from_MA_{w}'] = close / ma - 1


# Add noisy features
for lag in range(2, 21):   # many lags = mostly junk
    df[f'Return_lag_{lag}'] = df['Return_1d'].shift(lag)

df['Vol_MA_5'] = df['Volume'].rolling(5).mean()
df['Vol_MA_20'] = df['Volume'].rolling(20).mean()
df['Vol_Ratio'] = df['Vol_MA_5'] / df['Vol_MA_20']

df['Return_30d_ago'] = df['Return_1d'].shift(30)
df['Return_60d_ago'] = df['Return_1d'].shift(60)


# Prediction target: tomorrow return
df['Target'] = df['Return_1d'].pct_change(5).shift(-5)
# df['Target'] = df['Close'].pct_change(5).shift(-5)


# Additional cleaning
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()


# Prepare dataset for ML

features = df.drop(columns=['Target'])
X = features.values
y = df['Target'].values

split = int(len(df) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
y_test_dir = (y_test > 0).astype(int)
# y_train = np.clip(y_train, -1, 1)


# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# Regression pipeline

from sklearn import linear_model
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, f1_score

reg_pipe = Pipeline([
    # ("scaler", StandardScaler()),
    ("reg", linear_model.LinearRegression())
])
reg_pipe.fit(X_train, y_train)
y_pred_lr = reg_pipe.predict(X_test)
y_pred_lr_dir = (y_pred_lr > 0).astype(int)

direction_accuracy = (y_pred_lr_dir == y_test_dir).mean()
print("LinearRegression")
print("Direction Accuracy:", round(direction_accuracy*100,2), "%")
print("F1 score:", round(f1_score(y_test_dir, y_pred_lr_dir),2))
print("RMSE:", round(mean_absolute_error(y_test, y_pred_lr),2))
print("Test R²:", reg_pipe.score(X_test, y_test))


# Penalized regression pipeline (Ridge)

ridge_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("reg", linear_model.Ridge())
])
ridge_pipe.fit(X_train, y_train)
y_pred_ridge = ridge_pipe.predict(X_test)
y_pred_ridge_dir = (y_pred_ridge > 0).astype(int)

print("\nRidge")
direction_accuracy = (y_pred_ridge_dir == y_test_dir).mean()
print("Direction Accuracy:", round(direction_accuracy*100,2), "%")
print("F1 score:", round(f1_score(y_test_dir, y_pred_ridge_dir),2))
print("RMSE:", round(mean_absolute_error(y_test, y_pred_ridge),2))
print("Test R²:", ridge_pipe.score(X_test, y_test))


# Penalized regression pipeline (Lasso)

lasso_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("reg", linear_model.Lasso())
])
lasso_pipe.fit(X_train, y_train)
y_pred_lasso = lasso_pipe.predict(X_test)
y_pred_lasso_dir = (y_pred_lasso > 0).astype(int)

print("\nLasso")
direction_accuracy = (y_pred_lasso_dir == y_test_dir).mean()
print("Direction Accuracy:", round(direction_accuracy*100,2), "%")
print("F1 score:", round(f1_score(y_test_dir, y_pred_lasso_dir),2))
print("RMSE:", round(mean_absolute_error(y_test, y_pred_lasso),2))
print("Test R²:", lasso_pipe.score(X_test, y_test))


# ICS pipeline
from icspylab import ICS, cov, mcd, tcov, tcov2, tM, normal_crit
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit

pipe = Pipeline([
    ("ics", ICS(algorithm="whiten", S1=tcov, S2=cov, method_select=normal_crit, select_args={"test": "normal"})),
    # ("scaler", StandardScaler()),
    ("reg", linear_model.LinearRegression())
])

pipe.fit(X_train, y_train)
y_pred_ics = pipe.predict(X_test)
y_pred_ics_dir = (y_pred_ics > 0).astype(int)

print("\nICS + LinearRegression")
direction_accuracy_grid = (y_pred_ics_dir == y_test_dir).mean()
print("Direction Accuracy:", round(direction_accuracy_grid*100,2), "%")
print("F1 score:", round(f1_score(y_test_dir, y_pred_ics_dir),2))
print("RMSE:", round(mean_absolute_error(y_test, y_pred_ics),2))
print("Test R²:", pipe.score(X_test, y_test))


import plotly.graph_objects as go
import numpy as np

# Build a common x-axis
t_train = np.arange(len(y_train))
t_test  = np.arange(len(y_train), len(y_train) + len(y_test))

fig = go.Figure()

# ===== Train target =====
fig.add_trace(go.Scatter(
    x=t_train,
    y=y_train,
    mode="lines",
    name="y_train",
    line=dict(color="black"),
    opacity=0.6
))

# ===== Test target =====
fig.add_trace(go.Scatter(
    x=t_test,
    y=y_test,
    mode="lines",
    name="y_test (true)",
    line=dict(color="blue", width=2)
))

# ===== Predictions =====
fig.add_trace(go.Scatter(
    x=t_test,
    y=y_pred_lr,
    mode="lines",
    name="LinearRegression",
    line=dict(dash="dash")
))

fig.add_trace(go.Scatter(
    x=t_test,
    y=y_pred_ridge,
    mode="lines",
    name="Ridge",
    line=dict(dash="dash")
))

fig.add_trace(go.Scatter(
    x=t_test,
    y=y_pred_lasso,
    mode="lines",
    name="Lasso",
    line=dict(dash="dash")
))

fig.add_trace(go.Scatter(
    x=t_test,
    y=y_pred_ics,
    mode="lines",
    name="ICS + Ridge",
    line=dict(dash="dash")
))

# ===== Train / Test split =====
fig.add_vline(
    x=len(y_train),
    line_width=1,
    line_dash="dot",
    line_color="gray",
    annotation_text="Train / Test split",
    annotation_position="top"
)

# ===== Layout =====
fig.update_layout(
    title="Target vs Predictions (Train / Test)",
    xaxis_title="Time index",
    yaxis_title="Target value",
    template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

fig.show()

