import sys
import os

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.regression.rolling import RollingOLS

from prophet import Prophet

import tensorflow

def generate_ar_ma_dataframe(time_series, ar_order, diff_order, ma_order, seasonal):

    # Create a DataFrame with the original time series
    df = pd.DataFrame({'target': time_series})

    # Differencing (diff) Component
    if diff_order > 0:
       df[f'diff({diff_order})'] = time_series.diff(diff_order)

    # Autoregressive (AR) Component
    for i in range(1, ar_order + 1):
        df[f'AR({i})'] = df['target'].shift(i)

    # Moving Average (MA) Component
    df[f'MA({ma_order})'] = df['target'].rolling(window=ma_order).mean()


    # Seasonal Variable
    if seasonal != 0:
      if seasonal < df.index.dayofyear.max():
        df[f'Season'] = df['target'].shift(seasonal)
      else:
        df[f'Season'] = df['target'].shift(seasonal)

    df = df.dropna()

    return df


from sklearn.model_selection import train_test_split

def split_train_test(data,var_to_drop, train_size=0.9, random_state=None):

    # Split into training and test sets
    train, test = train_test_split(data, test_size=1 - train_size, shuffle=False, random_state=random_state)

    # Extract target variable
    y_train = train['target']
    y_test = test['target']

    X_train = train.drop(var_to_drop, axis=1)
    X_test = test.drop(var_to_drop, axis=1)

    return X_train, X_test, y_train, y_test, train.index, test.index


def evaluation_measure(model_name, y_train, y_train_pred, y_test, y_test_pred):
    # Calculate additional metrics for training data
    n_train = len(y_train)
    k_train = 1  # Assuming simple model with only intercept
    residuals_train = y_train - y_train_pred

    # Calculate AIC and BIC for training data
    aic_train = n_train * np.log(np.sum(residuals_train**2) / n_train) + 2 * k_train
    bic_train = n_train * np.log(np.sum(residuals_train**2) / n_train) + k_train * np.log(n_train)

    # Calculate MAE for training data
    mae_train = mean_absolute_error(y_train, y_train_pred)
    rmse_train = np.sqrt(((y_train-y_train_pred) ** 2).mean())

    # Calculate additional metrics for testing data
    n_test = len(y_test)
    k_test = 1
    residuals_test = y_test - y_test_pred

    # Calculate AIC and BIC for testing data
    aic_test = n_test * np.log(np.sum(residuals_test**2) / n_test) + 2 * k_test
    bic_test = n_test * np.log(np.sum(residuals_test**2) / n_test) + k_test * np.log(n_test)
    rmse_test = np.sqrt((( y_test - y_test_pred) ** 2).mean())

    results = {
        'Model': model_name,
        'AIC_train': aic_train,
        'BIC_train': bic_train,
        'AIC_test': aic_test,
        'BIC_test': bic_test,
        'R-squared_train': r2_score(y_train, y_train_pred),
        'R-squared_test': r2_score(y_test, y_test_pred),
        'MAE_train': mae_train,
        'MAE_test': mean_absolute_error(y_test, y_test_pred),
        'RMSE_train': rmse_train,
        'RMSE_test': rmse_test
    }

    return results


def evaluate_model(model, model_name, X_train, y_train, X_test, y_test):
    y_train_pred = model.predict(X_train).round(0)
    y_test_pred = model.predict(X_test).round(0)

    return evaluation_measure(model_name, y_train, y_train_pred, y_test, y_test_pred)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

def build_rnn_model(input_shape):
    model = Sequential()
    model.add(SimpleRNN(units=64, activation='relu', input_shape=input_shape))
    model.add(Dense(units=1, activation='linear'))

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

from tensorflow.keras.layers import LSTM, Dense

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(30, activation='relu', input_shape=input_shape))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def evaluation_all(model_name, target, prediction):
    n_train = len(target)
    k_train = 1
    residuals_train = target - prediction
    aic_train = n_train * np.log(np.sum(residuals_train**2) / n_train) + 2 * k_train
    bic_train = n_train * np.log(np.sum(residuals_train**2) / n_train) + k_train * np.log(n_train)
    mae_train = mean_absolute_error(target, prediction)
    rmse_train = np.sqrt(((prediction - target) ** 2).mean())

    results = {
        'Model': model_name,
        'AIC_train': aic_train,
        'BIC_train': bic_train,
        'R-squared_train': r2_score(target, prediction),
        'MAE_train': mae_train,
        'RMSE_train': rmse_train
    }

    return results

def forecast_new(latest, model, ar_order, diff_order, ma_order, month_ahaed = 1):
  # Make sure that the latest value of last series is not missing or 0
  i = 0
  while latest.iloc[-1]['target'] == 0:
    latest = latest.drop(latest.index[-1])
    i = i+1
  month_ahaed = i+month_ahaed

  # Forecasting new series
  for _ in range(month_ahaed):
    new_date = latest.index.max() + pd.DateOffset(months=1)
    new_row = pd.Series({'target': latest.loc[latest.index.max(), 'target']}, name=new_date)
    latest = pd.concat([latest, new_row.to_frame().T])

    last_data = generate_ar_ma_dataframe(latest['target'], ar_order, diff_order, ma_order,0)

    last_data = last_data.drop(columns='target')
    pred = model.predict(last_data)

    latest_index = latest.index.max()
    latest.loc[latest_index, 'target'] = pred[len(pred)-1]

  latest.iloc[0:len(latest)-month_ahaed, latest.columns.get_loc('target')] = np.nan
  return latest