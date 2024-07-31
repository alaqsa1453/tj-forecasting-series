import os
import pandas as pd

def load_dataset(dir_path, data_list):
    # Load all CSV files in the directory
    for filename in os.listdir(dir_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(dir_path, filename)
            df_name = filename[:-4]
            globals()[df_name] = pd.read_csv(file_path, delimiter=';')
    
    # Rename columns and merge dataframes
    merged_df = None
    for df_name in data_list:
        df = globals().get(df_name)
        if df is not None:
            cols = df.columns.tolist()
            for col in cols:
                if col not in ['bulan', 'tahun']:
                    df.rename(columns={col: df_name}, inplace=True)
            if merged_df is None:
                merged_df = df
            else:
                merged_df = pd.merge(merged_df, df, on=['bulan', 'tahun'], how='outer')
    
    return merged_df