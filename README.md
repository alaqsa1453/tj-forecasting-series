# tj-forecast

## Forecasting Transjakarta Passengers Through Time Series Model

### Data Explanation
The data is located in the `data` folder.

#### Files
- `training_jumlah_penumpang_tj.csv` - Dataset used to train the model.
- `testing_jumlah_penumpang_tj.csv` - Monthly series data to be predicted (test the model).
- `sample_submission.csv` - Sample submission file in the correct format.
- `jumlah_armada_tj.csv` - Auxiliary data used to train the model.
- `jumlah_penumpang_lrt.csv` - Auxiliary data used to train the model.
- `jumlah_perjalanan_lrt.csv` - Auxiliary data used to train the model.
- `jumlah_penumpang_mrt.csv` - Auxiliary data used to train the model.
- `jumlah_perjalanan_lrt.csv` - Auxiliary data used to train the model.

#### Variable Descriptions
- `bulan` - Month of the data.
- `tahun` - Year of the data.
- `jumlah_penumpang` - Total passengers who tapped in on Transjakarta, MRT, and LRT.
- `jumlah_armada_tj` - Total fleet operating under PT. Transportasi Jakarta, subsidized by the local government.
- `jumlah_perjalanan` - Number of trips made by the transportation modes to carry passengers.

### File and Folder Explanation
- **Folder `data`**: Contains the datasets needed for the analysis process.
- **Folder `script`**: Contains compiled Python functions for analysis processing.
- **Folder `output`**: Stores analysis results, including model predictions and forecasting.
- **File `config.json`**: Contains the Python version and libraries used.
- **File `main.ipynb`**: Full source file to run the model and overall analysis.

### Required Libraries

To run this project, you will need the following Python libraries:

```python
import sys
import os
import warnings
warnings.filterwarnings("ignore")

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
import prophet
import sklearn
import statsmodels

import tensorflow
```

You can install these libraries using pip with the following command:

```
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels prophet tensorflow
```

## Usage Instructions
1. **Prepare the Environment**: Ensure you have the required Python version and libraries as specified in `config.json`.
2. **Run the Notebook**: Open and run all cells in `main.ipynb` to perform the complete analysis and forecasting process.
3. **Save the Results**: The analysis and prediction results will be saved in the `output` folder.

### Project Performance Metrics

**Average Execution Time**

The average total execution time for the notebook is approximately **114.49 seconds**.

**Average Storage Size**

The average total storage size of the notebook is approximately **4.12 MB**.



# Methods

## 1. Machine Learning Model

We employed various machine learning models to forecast time series data. The models used include Random Forest, Support Vector Regression (SVR), Neural Network, Recurrent Neural Network (RNN), and Long Short-Term Memory (LSTM). The process involves extracting time series features and utilizing them for modeling.

### Steps:
- **Modeling Techniques**: Implemented Random Forest, SVR, Neural Network, RNN, and LSTM.
- **Feature Extraction**: Extracted time series features such as AutoRegressive (AR), Moving Average (MA), differences, and seasonal components.
- **Model Training**: Utilized the extracted time series features for training the models.

### Detailed Explanation:
1. **Random Forest**: Used for its robustness in handling non-linear relationships in time series data.
2. **SVR**: Employed for its ability to model complex patterns with flexibility.
3. **Neural Network, RNN, and LSTM**: Applied for their proficiency in capturing temporal dependencies and sequential patterns in the data.

![Machine Learning Model Diagram](path_to_machine_learning_model_image.png)

## 2. Prophet Model

Prophet is an open-source forecasting tool developed by Facebook. It is designed to handle time series data with strong seasonal effects and multiple seasonality. In our project, we separated seasonal patterns before and after the COVID-19 pandemic using a changepoint on '2020-01-01'. Additionally, we included dummy variables for COVID-19 lockdown periods, school holidays, and major holidays like Idul Fitri.

### Steps:
- **Seasonal Separation**: Differentiated the seasonal patterns before and after COVID-19 with a changepoint at '2020-01-01'.
- **Dummy Variables**: Added dummy variables for COVID-19 lockdown periods, school holidays, and major holidays.

### Detailed Explanation:
Prophet allows for the incorporation of domain knowledge through dummy variables and changepoints, which helps in improving the accuracy of the forecasts.

![Prophet Model Diagram](path_to_prophet_model_image.png)

## 3. Combine Model

We combined the results from the machine learning models and the Prophet model for enhanced forecasting accuracy. This method involves regressing the time series data with available auxiliary variables using linear regression. The combined data is then used to run the Prophet model and make future predictions.

### Steps:
1. **Linear Regression**: Regressed the time series data with available auxiliary variables for the period 2023 (monthly) to predict from '2024-01-01' to '2024-01-05'.
2. **Combine Predictions**: Added the regression model's predictions to the training data.
3. **Prophet Model**: Ran the Prophet model and obtained predictions up to '2024-06-01'.
4. **Ridge Regression**: Combined the training data with auxiliary MRT data (monthly passenger numbers from January 2022 to June 2024) and applied ridge regression.
5. **Forecast Results**: Detailed the forecast results in a table format.

### Forecast Results Table:
| Date       | Forecast Method                       |
|------------|---------------------------------------|
| 2024-01-01 | Linear regression + Prophet           |
| 2024-02-01 | Linear regression + Prophet           |
| 2024-03-01 | Linear regression + Prophet           |
| 2024-04-01 | Linear regression + Prophet           |
| 2024-05-01 | Linear regression + Prophet           |
| 2024-06-01 | Prophet + Ridge regression            |

### Detailed Explanation:
By combining the strengths of machine learning models and the Prophet model, we can leverage their respective advantages to enhance the accuracy and robustness of our forecasts.

![Combine Model Diagram](path_to_combine_model_image.png)

