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

# Run successfully on Kaggle and Colab

# # Environment

# !pip install pytorch-tabnet category_encoders

# +
#===========================================================
# Library
#===========================================================
import os
import gc
from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter
from contextlib import contextmanager
import time

import numpy as np
import pandas as pd
import scipy as sp
import random

import matplotlib.pyplot as plt
import seaborn as sns

from IPython.display import display

from functools import partial

from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold
from sklearn import preprocessing
import category_encoders as ce
from sklearn.metrics import mean_squared_error

import torch
from sklearn.preprocessing import LabelEncoder

import pandas as pd
import numpy as np
np.random.seed(0)

from pytorch_tabnet.tab_model import TabNetRegressor ##Import Tabnet 

from pathlib import Path

import lightgbm as lgb

import warnings
warnings.filterwarnings("ignore")
# -

os.listdir('../input/trends-assessment-prediction/') # kaggle data directory


# +
#===========================================================
# Utils
#===========================================================

def get_logger(filename='log'): # [QuickStart for logging in Python](shorturl.at/hnsyQ)
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=f"{filename}.log")
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger

logger = get_logger()


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    logger.info(f'[{name}] done in {time.time() - t0:.0f} s')


def seed_everything(seed=777):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    
def load_df(path, df_name, debug=False):
    if path.split('.')[-1]=='csv':
        df = pd.read_csv(path)
        if debug:
            df = pd.read_csv(path, nrows=1000)
    elif path.split('.')[-1]=='pkl':
        df = pd.read_pickle(path)
    if logger==None:
        print(f"{df_name} shape / {df.shape} ")
    else:
        logger.info(f"{df_name} shape / {df.shape} ")
    return df


# +
#===========================================================
# Config
#===========================================================
OUTPUT_DICT = ''

ID = 'Id'
TARGET_COLS = ['age', 'domain1_var1', 'domain1_var2', 'domain2_var1', 'domain2_var2']
SEED = 42
seed_everything(seed=SEED)

N_FOLD = 5
# -

# # Data

train = pd.read_csv('../input/trends-assessment-prediction/train_scores.csv', dtype={'Id':str})\
            .dropna().reset_index(drop=True) # to make things easy
reveal_ID = pd.read_csv('../input/trends-assessment-prediction/reveal_ID_site2.csv', dtype={'Id':str})
ICN_numbers = pd.read_csv('../input/trends-assessment-prediction/ICN_numbers.csv')
loading = pd.read_csv('../input/trends-assessment-prediction/loading.csv', dtype={'Id':str})
fnc = pd.read_csv('../input/trends-assessment-prediction/fnc.csv', dtype={'Id':str})
sample_submission = pd.read_csv('../input/trends-assessment-prediction/sample_submission.csv', dtype={'Id':str})

# +
# at first glance of each data
dfs = [train, reveal_ID, ICN_numbers, loading, fnc, sample_submission]
names = ['train', 'reveal_ID', 'ICN_numbers', 'loading', 'fnc', 'sample_submission']
for i, (f, n) in enumerate(zip(dfs, names)):
    print(f'{n}')
    display(f.head())

sample_submission['ID_num'] = sample_submission[ID].apply(lambda x: int(x.split('_')[0]))
test = pd.DataFrame({ID: sample_submission['ID_num'].unique().astype(str)})
del sample_submission['ID_num']; gc.collect()
test.head()

fnc_features =list(fnc.columns[1:])

# merge
train = train.merge(loading, on=ID, how='left')
train = train.merge(fnc, on=ID, how='left')
train.head()

# merge
test = test.merge(loading, on=ID, how='left')
test = test.merge(fnc, on=ID, how='left')
test.head()

FNC_SCALE = 1/500 # LIPIN: this step seems to be stupid.                 

train.loc[:, fnc_features] *= FNC_SCALE
test.loc[:, fnc_features] *= FNC_SCALE

folds = train.loc[:,[ID]+TARGET_COLS].copy()
Fold = KFold(n_splits=N_FOLD, shuffle=True, random_state=SEED)
for n, (train_index, val_index) in enumerate(Fold.split(folds, folds[TARGET_COLS])):
    folds.loc[val_index, 'fold'] = int(n)
folds['fold'] = folds['fold'].astype(int)
folds.head()


# +
#===========================================================
# model
#===========================================================
def run_single_tabnet(clf,train_df, test_df, folds, features, target, fold_num=0, categorical=[]):
    
    trn_idx = folds[folds.fold != fold_num].index
    val_idx = folds[folds.fold == fold_num].index
    logger.info(f'len(trn_idx) : {len(trn_idx)}')
    logger.info(f'len(val_idx) : {len(val_idx)}')
    X_train= train_df.iloc[trn_idx][features].values ###Converted this into Numpy array because TabNet will give error otherwise .
    y_train=target.iloc[trn_idx].values.reshape(-1, 1)
    X_valid = train_df.iloc[val_idx][features].values
    y_valid= target.iloc[val_idx].values.reshape(-1, 1)

    oof = np.zeros(len(train_df))
    predictions = np.zeros(len(test_df))
    

    clf.fit(
                X_train=X_train, y_train=y_train, ##Train features and train targets
                X_valid=X_valid, y_valid=y_valid, ##Valid features and valid targets
                weights =0,#0 for no balancing,1 for automated balancing,dict for custom weights per class
                max_epochs=1000,##Maximum number of epochs during training , Default 1000. I used 10
                patience=70, ##Number of consecutive non improving epoch before early stopping , Default 50
                batch_size=1024, ##Training batch size
                virtual_batch_size=128 )##Batch size for Ghost Batch Normalization (virtual_batch_size < batch_size)

    oof[val_idx] = np.squeeze(clf.predict(train_df.iloc[val_idx][features].values))

    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = features
    fold_importance_df["importance"] = clf.feature_importances_
    fold_importance_df["fold"] = fold_num

    predictions += np.squeeze(clf.predict(test_df[features].values))
    
    # RMSE
    logger.info("fold{} RMSE score: {:<8.5f}".format(fold_num, np.sqrt(mean_squared_error(target[val_idx], oof[val_idx]))))
    
    return oof, predictions, fold_importance_df


def run_kfold_tabnet(clf,train, test, folds, features, target, n_fold=5, categorical=[]):
    
    logger.info(f"================================= {n_fold}fold TabNet =================================")
    
    oof = np.zeros(len(train))
    predictions = np.zeros(len(test))
    feature_importance_df = pd.DataFrame()

    for fold_ in range(n_fold):
        print("Fold {}".format(fold_))
        _oof, _predictions, fold_importance_df = run_single_tabnet(clf,train,
                                                                     test,
                                                                     folds,
                                                                     features,
                                                                     target,
                                                                     fold_num=fold_,
                                                                     categorical=categorical)
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        oof += _oof
        predictions += _predictions / n_fold

    # RMSE
    logger.info("CV RMSE score: {:<8.5f}".format(np.sqrt(mean_squared_error(target, oof))))

    logger.info(f"=========================================================================================")
    
    return feature_importance_df, predictions, oof

    
def show_feature_importance(feature_importance_df, name):
    cols = (feature_importance_df[["Feature", "importance"]]
            .groupby("Feature")
            .mean()
            .sort_values(by="importance", ascending=False)[:50].index)
    best_features = feature_importance_df.loc[feature_importance_df.Feature.isin(cols)]

    plt.figure(figsize=(8, 16))
    sns.barplot(x="importance", y="Feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('Features importance (averaged/folds)')
    plt.tight_layout()
    plt.savefig(OUTPUT_DICT+f'feature_importance_{name}.png')


# +
prediction_dict = {}
oof_dict = {}

for TARGET in TARGET_COLS: ## I think this model will work for multiple targets altogether , let me try that later
    
    logger.info(f'### TABNET for {TARGET} ###')

    target = train[TARGET]
    test[TARGET] = np.nan

    # features
    cat_features = []
    num_features = [c for c in test.columns if (test.dtypes[c] != 'object') & (c not in cat_features)]
    features = num_features + cat_features
    drop_features = [ID] + TARGET_COLS
    features = [c for c in features if c not in drop_features]

    if cat_features:
        ce_oe = ce.OrdinalEncoder(cols=cat_features, handle_unknown='impute')
        ce_oe.fit(train)
        train = ce_oe.transform(train)
        test = ce_oe.transform(test)
        
    cat_idxs = [ i for i, f in enumerate(features) if f in cat_features]

    cat_dims = [ categorical_dims[f] for i, f in enumerate(features) if f in cat_features]    
        
    clf = TabNetRegressor(
                        n_d = 16,##Width of the decision prediction layer. Bigger values gives more capacity to the model with the risk of overfitting. Values typically range from 8 to 64.
                        n_a = 16,##Width of the attention embedding for each mask. According to the paper n_d=n_a is usually a good choice. (default=8)
                        n_steps = 3,##Number of steps in the architecture (usually between 3 and 10)
                        gamma =1.3,##This is the coefficient for feature reusage in the masks. A value close to 1 will make mask selection least correlated between layers. Values range from 1.0 to 2.0.
                        cat_idxs=cat_idxs, ##List of categorical features indices.
                        cat_dims=cat_dims,
                        cat_emb_dim =1, ##List of embeddings size for each categorical features. (default =1)
                        n_independent =2,##Number of independent Gated Linear Units layers at each step. Usual values range from 1 to 5.
                        n_shared =2,##Number of shared Gated Linear Units at each step Usual values range from 1 to 5
                        epsilon  = 1e-15,##Should be left untouched.
                        seed  =0,##Random seed for reproducibility
                        momentum = 0.02, ##Momentum for batch normalization, typically ranges from 0.01 to 0.4 (default=0.02)
                        lr = 0.01, ##Initial learning rate used for training. As mentionned in the original paper, a large initial learning of 0.02 with decay is a good option.
                        clip_value =None,
                        lambda_sparse =1e-3,##This is the extra sparsity loss coefficient as proposed in the original paper. The bigger this coefficient is, the sparser your model will be in terms of feature selection. Depending on the difficulty of your problem, reducing this value could help.
                        optimizer_fn =torch.optim.Adam, ## Optimizer
                        scheduler_fn = None, #torch.optim.lr_scheduler.ReduceLROnPlateau, ## LR scheduler 
                        scheduler_params = None,#{"mode":'min', "factor":0.1, "patience":10, "verbose":"False"}, ## LR scheduler parameters dictionary
                        verbose =1,
                        device_name = 'auto' ## Auto or 'gpu' ## I have no GPU
                        
                        )    

    feature_importance_df, predictions, oof = run_kfold_tabnet(clf,train, test, folds, features, target, 
                                                                 n_fold=N_FOLD, categorical=cat_features)
    
    prediction_dict[TARGET] = predictions
    oof_dict[TARGET] = oof
    
    show_feature_importance(feature_importance_df, TARGET)


# +
# https://www.kaggle.com/akurmukov/trends-starter-rf-0-168-lb-metric

def lb_metric(y_true, y_pred):
    '''Computes lb metric, both y_true and y_pred should be DataFrames of shape n x 5'''
    y_true = y_true[['age', 'domain1_var1', 'domain1_var2','domain2_var1','domain2_var2']]
    y_pred = y_pred[['age', 'domain1_var1', 'domain1_var2','domain2_var1','domain2_var2']]
    weights = np.array([.3, .175, .175, .175, .175])
    return np.sum(weights * np.abs(y_pred.values - y_true.values).sum(axis=0) / y_true.values.sum(axis=0))

oof_df = pd.DataFrame()

for TARGET in TARGET_COLS:
    oof_df[TARGET] = oof_dict[TARGET]


score = lb_metric(train, oof_df)
logger.info(f'Local Score: {score}')
# -

# # Submission

# +
sample_submission.head()

pred_df = pd.DataFrame()

for TARGET in TARGET_COLS:
    tmp = pd.DataFrame()
    tmp[ID] = [f'{c}_{TARGET}' for c in test[ID].values]
    tmp['Predicted'] = prediction_dict[TARGET]
    pred_df = pd.concat([pred_df, tmp])

print(pred_df.shape)
print(sample_submission.shape)

pred_df.head()

submission = sample_submission.drop(columns='Predicted').merge(pred_df, on=ID, how='left')
print(submission.shape)
submission.to_csv('submission.csv', index=False)
submission.head()
