import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_squared_error
import os


def load_and_prepare_data(filepath):
    data = pd.read_csv(filepath, delimiter=';')
    data = data.rename(columns={'tahun': 'year', 'bulan': 'month'})
    data['date'] = pd.to_datetime(data[['year', 'month']].assign(day=1))
    data.set_index('date', inplace=True)
    data = data.drop(columns=['year', 'month'])
    return data


def add_features(df, prediction):
    df = df.reset_index().rename(columns={'date': 'ds', 'jumlah_penumpang': 'y'})
    df = pd.concat([df, prediction])
    df['lockdown'] = 0
    lockdown_dates = ['2019-12-01', '2020-01-01', '2020-02-01', '2020-03-01', 
                      '2020-04-01', '2020-05-01', '2020-06-01', '2021-06-01', 
                      '2021-07-01', '2023-02-01', '2024-04-01']
    df.loc[df['ds'].isin(pd.to_datetime(lockdown_dates)), 'lockdown'] = -1
    df['peak_holiday'] = 0
    df.loc[(df['ds'].dt.month == 6) | (df['ds'].dt.month == 7), 'peak_holiday'] = 1
    df['before_2020'] = (df['ds'] < '2020-01-01').astype(int)
    df['after_2020'] = (df['ds'] >= '2020-01-01').astype(int)
    return df


def create_and_fit_model(df):
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode='multiplicative',
        changepoint_prior_scale=25
    )
    model.add_seasonality(name='monthly_before_2020', period=30.5, fourier_order=12, condition_name='before_2020')
    model.add_seasonality(name='monthly_after_2020', period=30.5, fourier_order=12, condition_name='after_2020')
    model.add_regressor('lockdown')
    model.add_regressor('peak_holiday')
    model.fit(df)
    return model


def make_predictions(model, df, periods):
    future = model.make_future_dataframe(periods=periods, freq='MS')
    future['lockdown'] = 0
    future['peak_holiday'] = 0
    future['before_2020'] = (future['ds'] < '2020-01-01').astype(int)
    future['after_2020'] = (future['ds'] >= '2020-01-01').astype(int)
    future.loc[(future['ds'].dt.month == 6) | (future['ds'].dt.month == 7), 'peak_holiday'] = 1
    forecast = model.predict(future)
    forecast[['yhat', 'yhat_lower', 'yhat_upper']] = forecast[['yhat', 'yhat_lower', 'yhat_upper']].round(0)
    return forecast


def evaluate_model(df, forecast):
    df_forecast = forecast.set_index('ds')[['yhat']].join(df.set_index('ds'))
    rmse = np.sqrt(mean_squared_error(df['y'], df_forecast['yhat'][:len(df['y'])]))
    return rmse, df_forecast


def plot_results(df_forecast, forecast, rmse):
    plt.figure(figsize=(10, 6))
    plt.plot(df_forecast.index, df_forecast['y'], label='True Value', color='blue')
    plt.plot(df_forecast.index, df_forecast['yhat'], label='Predicted Value', color='red')
    plt.fill_between(df_forecast.index, forecast['yhat_lower'], forecast['yhat_upper'], color='pink', alpha=0.3)
    plt.axvspan(pd.to_datetime('2023-01-01'), pd.to_datetime('2023-12-01'), color='yellow', alpha=0.3)
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('Jumlah Penumpang')
    plt.title(f'Prediksi Jumlah Penumpang Menggunakan Prophet\nRMSE: {rmse:.2f}')
    plt.show()


def main_prophet_v1(filepath, periods, prediction):
    data = load_and_prepare_data(filepath)
    df = add_features(data, prediction)
    model = create_and_fit_model(df)
    forecast = make_predictions(model, df, periods)
    rmse, df_forecast = evaluate_model(df, forecast)
    print(f'RMSE: {rmse}')
    plot_results(df_forecast, forecast, rmse)
    return df_forecast