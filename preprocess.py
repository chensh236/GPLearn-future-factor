import pandas as pd
import numpy as np
import pickle

target_inv = 'PTA'
data_df = pd.read_csv(f'dataset/{target_inv}.csv')
data_df.date = pd.to_datetime(data_df.date, format='%Y-%m-%d %H:%M:%S')
data_df.index = data_df.date
data_df.drop(['StockID'], axis=1, inplace=True)
data_df['1_min_return'] = data_df.open.pct_change(1).shift(-1)
data_df = data_df.dropna()
data_df.to_hdf(f'dataset/{target_inv}.h5', key='data', mode='w', complevel=9)