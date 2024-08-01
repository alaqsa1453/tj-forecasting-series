import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_squared_error
import os

def load_and_prepare_data(filepath):
    """Load and preprocess data."""
    data = pd.read_csv(filepath, delimiter=';')
    data = data.rename(columns={'tahun': 'year', 'bulan': 'month'})
    data['date'] = pd.to_datetime(data[['year', 'month']].assign(day=1))
    data.set_index('date', inplace=True)
    data = data.drop(columns=['year', 'month'])
    return data

dir_path = 'data'
for filename in os.listdir(dir_path):
  if filename.endswith('.csv'):
    file_path = os.path.join(dir_path, filename)

    df_name = filename[:-4]
    globals()[df_name] = pd.read_csv(file_path, delimiter=';')

def add_features(df, prediction):
    """Add features to the dataframe."""
    df = df.reset_index().rename(columns={'date': 'ds', 'jumlah_penumpang': 'y'})
    df = pd.concat([df, prediction])
    df['lockdown'] = 0
    lockdown_dates = ['2021-06-01', '2021-07-01', '2023-02-01']
    df.loc[df['ds'].isin(pd.to_datetime(lockdown_dates)), 'lockdown'] = -1
    df['peak_holiday'] = 0
    df.loc[(df['ds'].dt.month == 6) | (df['ds'].dt.month == 7), 'peak_holiday'] = 1
    df['before_2020'] = (df['ds'] < '2020-01-01').astype(int)
    df['after_2020'] = (df['ds'] >= '2020-01-01').astype(int)
    return df

def create_and_fit_model(df, period_name):
    """Create and fit a Prophet model for a given period."""
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode='multiplicative',
        changepoint_prior_scale=0.25
    )
    model.add_regressor('lockdown')
    model.add_regressor('peak_holiday')
    model.add_seasonality(name=f'monthly_{period_name}', period=30.5, fourier_order=12)
    model.fit(df)
    return model

def make_predictions(model, periods):
    """Generate predictions using the fitted model."""
    future = model.make_future_dataframe(periods=periods, freq='MS')
    future['lockdown'] = 0
    future['peak_holiday'] = 0
    future.loc[(future['ds'].dt.month == 6) | (future['ds'].dt.month == 7), 'peak_holiday'] = 1
    forecast = model.predict(future)
    return forecast

def evaluate_model(df, forecast):
    """Evaluate model performance using RMSE."""
    df_forecast = forecast.set_index('ds')[['yhat']].join(df.set_index('ds'))
    rmse = np.sqrt(mean_squared_error(df['y'].iloc[-6:], df_forecast['yhat'][:len(df['y'].iloc[-6:])]))
    return rmse, df_forecast

def plot_results(df_forecast_combined, rmse):
    """Plot the results of the forecasts."""
    plt.figure(figsize=(10, 6))
    plt.plot(df_forecast_combined.index, df_forecast_combined['y'], label='True Value', color='blue')
    plt.plot(df_forecast_combined.index, df_forecast_combined['yhat'], label='Predicted Value', color='red')
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('Jumlah Penumpang')
    plt.title(f'Prediksi Jumlah Penumpang Menggunakan Prophet\nRMSE: {rmse:.2f}')
    plt.show()

def main_prophet_v2(filepath, periods, prediction):
    """Main function to execute the forecasting pipeline."""
    data = load_and_prepare_data(filepath)
    df = add_features(data, prediction)
    
    # Segment data
    df_before_2020 = df[df['ds'] < '2020-01-01']
    df_after_2020 = df[df['ds'] >= '2020-01-01']
    
    # Create and fit models
    model_before_2020 = create_and_fit_model(df_before_2020, 'before_2020')
    model_after_2020 = create_and_fit_model(df_after_2020, 'after_2020')
    
    # Make predictions
    forecast_before_2020 = make_predictions(model_before_2020, periods=0)
    forecast_after_2020 = make_predictions(model_after_2020, periods=periods)
    
    # Evaluate models
    rmse_before_2020, df_forecast_before_2020 = evaluate_model(df_before_2020, forecast_before_2020)
    rmse_after_2020, df_forecast_after_2020 = evaluate_model(df_after_2020, forecast_after_2020)
    
    # Combine forecasts
    df_forecast_combined = pd.concat([df_forecast_before_2020, df_forecast_after_2020])
    aux_.index = df_forecast_combined.iloc[-6:,].index
    df_forecast_combined.iloc[-6:,0] = aux_['pred_1']

    # Print overall RMSE
    overall_rmse = np.sqrt(mean_squared_error(df['y'], df_forecast_combined['yhat'][:len(df['y'])]))
    print(f'Overall RMSE: {overall_rmse}')
    
    # Plot results
    plot_results(df_forecast_combined, overall_rmse)

    return df_forecast_combined