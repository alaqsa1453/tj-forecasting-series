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

## Usage Instructions
1. **Prepare the Environment**: Ensure you have the required Python version and libraries as specified in `config.json`.
2. **Run the Notebook**: Open and run all cells in `main.ipynb` to perform the complete analysis and forecasting process.
3. **Save the Results**: The analysis and prediction results will be saved in the `output` folder.

### Project Performance Metrics

**Average Execution Time**

The average total execution time for the notebook is approximately **114.49 seconds**.

**Average Storage Size**

The average total storage size of the notebook is approximately **4.12 MB**.
