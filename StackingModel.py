# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: 'Python 3.7.7 64-bit (''tensorflow'': conda)'
#     language: python
#     name: python37764bittensorflowconda80fc7f00de1049449f5177d6f3ffe3e9
# ---

# #

import pandas as pd
import numpy as np

# interaction terms of loading and their feature names
from sklearn.preprocessing import PolynomialFeatures
from itertools import combinations
import functools
import operator

from scipy.stats import ks_2samp
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# IPython
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', None)

PATH = r"G:\kaggleData\TReNDS"
fnc = pd.read_csv(f'{PATH}/fnc.csv').sort_values(by=['Id'])
loading = pd.read_csv(f'{PATH}/loading.csv').sort_values(by=['Id'])
train_scores = pd.read_csv(f'{PATH}/train_scores.csv').sort_values(by=['Id'])
reveal_id_site2 = pd.read_csv(f'{PATH}/reveal_ID_site2.csv')

# get attributes
fnc_features = fnc.columns.tolist()
fnc_features.remove('Id')
loading_features = loading.columns.tolist()
loading_features.remove('Id')
target_features = train_scores.columns.tolist()
target_features.remove('Id')

# dimension of data frames
dfs = [fnc, loading, train_scores, reveal_id_site2]
nms = ['fnc', 'loading', 'train_scores', 'reveal_id_site2']
for (df, nm) in zip(dfs, nms):
    df.name = nm
    print(f'{nm:16s}: shape {df.shape}')

#
print(f'fnc_features:     len {len(fnc_features)}')
print(f'loading_features: len {len(loading_features)}')
print(f'target_features:  len {len(target_features)}')


# iteraction terms' names
def convertTuple(tup):
    str = functools.reduce(operator.add, tup)
    return str

polynomial_features = list(map(convertTuple, combinations(loading_features, r=2)))

# original loading features appended with interaction terms
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
temp = poly.fit_transform(loading.drop(['Id'], axis=1).values)
loading2_features = loading_features + polynomial_features
loading2 = pd.concat([loading[['Id']], pd.DataFrame(temp, columns=loading2_features)], axis=1)

#
full_features = fnc_features + loading2_features
train_scores['is_train'] = True
full_data = pd.merge(fnc, loading2, on='Id').merge(train_scores, how='left', on='Id')
train_df = full_data.loc[full_data['is_train']==True, :]
test_df = full_data.loc[~(full_data['is_train']==True), :]
train_df.equals(test_df)








