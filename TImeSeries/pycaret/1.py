import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime
import statsmodels.api as sm
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
import imageio
import os
from statsmodels.graphics.tsaplots import plot_acf

import pandas as pd

url = 'https://drive.google.com/file/d/1g3UG_SWLEqn4rMuYCpTHqPlF0vnIDRDB/view?usp=sharing'
path = 'https://drive.google.com/uc?export=download&id=' + url.split('/')[-2]

df2 = pd.read_csv(path)
df2_ds = df2[['date', 'sale_dollars']]
df2_ds = df2_ds.sort_index(axis=0)


def create_features(df):
    """
    Creates time series features from datetime index
    """
    df['date'] = df.index
    df['dayofweek'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['dayofyear'] = df['date'].dt.dayofyear
    df['dayofmonth'] = df['date'].dt.day
    df['weekofyear'] = df['date'].dt.weekofyear
    df['flag'] = pd.Series(np.where(df['date'] >= np.datetime64('2020-03-03'), 1, 0),
                           index=df.index)  # flag for COVID-19
    # df['rolling_mean_7'] = df['sale_dollars'].shift(7).rolling(window=7).mean()
    # df['lag_7'] = df['sale_dollars'].shift(7)
    # df['lag_15']=df['sale_dollars'].shift(15)
    # df['lag_last_year']=df['sale_dollars'].shift(52).rolling(window=15).mean()

    X = df[['dayofweek', 'quarter', 'month', 'year',
            'dayofyear', 'dayofmonth', 'weekofyear', 'flag', 'sale_dollars']]
    X.index = df.index
    return X


def split_data(data, split_date):
    return data[data.index <= split_date].copy(), \
           data[data.index > split_date].copy()


aggregated = df2_ds.groupby('date', as_index=True).sum()
aggregated.index = pd.to_datetime(aggregated.index)
aggregated = create_features(aggregated)
train, test = split_data(aggregated, '2020-06-15')  # splitting the data for training before 15th June

plt.figure(figsize=(20, 10))
plt.xlabel('date')
plt.ylabel('sales')
plt.plot(train.index, train['sale_dollars'], label='train')
plt.plot(test.index, test['sale_dollars'], label='test')
plt.legend()
# plt.show()

from pycaret.regression import *

reg = setup(data = train,
             target = 'sale_dollars',
             numeric_imputation = 'mean',
             categorical_features = ['dayofweek','quarter','month','year','dayofyear','dayofmonth','weekofyear',
                                     'flag']  ,
            transformation = True, transform_target = True,
                  combine_rare_levels = True, rare_level_threshold = 0.1,
                  remove_multicollinearity = True, multicollinearity_threshold = 0.95,
             silent = True)
