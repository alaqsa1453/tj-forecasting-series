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

### Usage Instructions
1. **Prepare the Environment**: Ensure you have the required Python version and libraries as specified in `config.json`.
2. **Run the Notebook**: Open and run all cells in `main.ipynb` to perform the complete analysis and forecasting process.
3. **Save the Results**: The analysis and prediction results will be saved in the `output` folder.

