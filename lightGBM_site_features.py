##

# Credits to the article "TReNDS Neuroimaging - Data Analysis & MATLAB Files" by Gunes Evitan
#
# data:
#     attributes comes from two files: fnc, loading
# target:
#     five target features to predict
# issue:
#     training set consists of examples collectd from site 1 only
#     test set is composed of examples collected from both sites 1 and 2.
#     the partial list of site 2 ID is released.
#
# what Gunes Evitan did:
# 1. use lightgbm to geneate the site feature base on a subset of fnc features and 
#    all of the loading features using the Kolmogorov-Sminorov 2 sample test to find
#    the essential features that are distributed differently across site 1 and 2.
# 
# 2020. June 22 by Li-Pin

##

# lightgbm: generate the site feature.
# ks_2samp: identify features with difference in distribution across sites,
#   where ks refers to Kolmogorov Sminorov test.
import pandas as pd
import numpy as np
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

##
# todo:
# 1. FNC should provide good features for predicting age
# 2. [seq2seq with RNN for multiple output regression](https://rb.gy/k9wo43)
# 3. sns.distplot
# 4. main stream: RAPIDS svm, LASSO, 3D CNN, TabNet.
# 5. Missing target features, site prediction (and its imbalance issue)

PATH = r"C:\Users\libin\Desktop\TReNDS"
fnc = pd.read_csv(f'{PATH}/fnc.csv')
loading = pd.read_csv(f'{PATH}/loading.csv')
train_scores = pd.read_csv(f'{PATH}/train_scores.csv')
site2_ids = pd.read_csv(f'{PATH}/reveal_ID_site2.csv').values.flatten()

#
fnc_features, loading_features = list(fnc.columns[1:]), list(loading.columns[1:])
train_scores['is_train'] = 1
target_features = train_scores.drop(['Id', 'is_train'], axis=1).columns
df = fnc.merge(loading, on='Id')

#
print(f'Static FNC Correlation Shape = {fnc.shape}')
print(f'sMRI SBM Loadings Shape = {loading.shape}')
print(f'Train Scores Shape = {train_scores.shape}')
print(f'Train & Test Set Shape = {df.shape}')

#
df = df.merge(train_scores, how='left', on='Id')
df.loc[df['is_train'].isnull(), 'is_train'] = 0

df['is_train'] = np.dtype('int64').type(df['is_train'])
df['Id'] = np.dtype('int64').type(df['Id'])

#

df['site'] = 0 # denote unknown
df.loc[df['is_train'] == 1, 'site'] = 1
df.loc[df['Id'].isin(site2_ids), 'site'] = 2
df['site'] = np.dtype('int64').type(df['site'])


"""
# works
ks_threshold = 0.125
ks_pvalue_threshold = 0.005 # (alpha=0.05, 450) (alpha=0.005, 219)
# shifted_features = list()
# shifted_features2 = list()
res = list()
for i, fnc_feature in enumerate(fnc_features):
    ks_stat, ks_pvalue = get_distribution_difference(fnc_feature)[1]
    # if ks_stat > ks_threshold:
    #     shifted_features.append(fnc_feature)
    # if ks_pvalue < ks_pvalue_threshold:
        # shifted_features2.append(fnc_feature)
    res.append([fnc_feature, ks_stat, ks_pvalue])
    print(i, fnc_feature)
"""
def get_distribution_difference(feature):
    site1_values = df[df['site'] == 1][feature].values
    site2_values = df[df['site'] == 2][feature].values
    return feature, ks_2samp(site1_values, site2_values)

ks_threshold = 0.125 # (0.1 = 84) (0.125 = 22) (0.135 = 10) (0.145 = 4)
shifted_features = list()
for i, fnc_feature in enumerate(fnc_features):
    print(i, fnc_feature)
    f, (stat, pval) = get_distribution_difference(fnc_feature)
    if stat > 0.125:
        shifted_features.append(fnc_feature)

# shifted_features = [fnc_feature for fnc_feature in fnc_features if get_distribution_difference(fnc_feature)[1][0] > ks_threshold]
display(f'The size of shifted_features: {len(shifted_features)}')
##
site_predictors = shifted_features + loading_features

X_train = df.loc[df['site'] > 0, site_predictors]
y_train = df.loc[df['site'] > 0, 'site']
X_test = df.loc[df['site'] == 0, site_predictors]
df['site_predicted'] = 0

# K: the number of folds
# oof_scores: to store out-of-fold scores for K iteration
# feature_importance: m X n dataframe, store m feature importances for n folds
K = 2
SEED = 1337
skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=SEED)
oof_scores = []
feature_importance = pd.DataFrame(np.zeros((X_train.shape[1], K)),
                                  columns=[f'Fold_{i}_Importance' for i in range(1, K + 1)],
                                  index=X_train.columns)
site_model_parameters = {
    'num_iterations': 500,
    'early_stopping_round': 50,
    'num_leaves': 2 ** 5,
    'learning_rate': 0.05,
    'bagging_fraction': 0.9,
    'bagging_freq': 1,
    'feature_fraction': 0.9,
    'feature_fraction_bynode': 0.9,
    'lambda_l1': 0,
    'lambda_l2': 0,
    'max_depth': -1,
    'objective': 'regression',
    'seed': SEED,
    'feature_fraction_seed': SEED,
    'bagging_seed': SEED,
    'drop_seed': SEED,
    'data_random_seed': SEED,
    'boosting_type': 'gbdt',
    'verbose': 1,
    'metric': 'rmse',
    'n_jobs': -1,
}

# [lightgbm.train API doc](shorturl.at/aims8)
# [Understand lightgbm parameters like num_iteration](shorturl.at/rsNOV)
# [lightgbm.feature_importance API doc](shorturl.at/arIZ3)
print('Running LightGBM Site Classifier Model\n' + ('-' * 38) + '\n')

for fold, (trn_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):

    trn_data = lgb.Dataset(X_train.iloc[trn_idx, :], label=y_train.iloc[trn_idx])
    val_data = lgb.Dataset(X_train.iloc[val_idx, :], label=y_train.iloc[val_idx])

    site_model = lgb.train(site_model_parameters, trn_data, valid_sets=[trn_data, val_data], verbose_eval=50)
    feature_importance.iloc[:, fold - 1] = site_model.feature_importance(importance_type='gain')

    site_oof_predictions = site_model.predict(X_train.iloc[val_idx, :], num_iteration=site_model.best_iteration)
    df.loc[X_train.iloc[val_idx, :].index, 'site_predicted'] = site_oof_predictions

    site_test_predictions = site_model.predict(X_test, num_iteration=site_model.best_iteration)
    df.loc[X_test.index, 'site_predicted'] += site_test_predictions / K

    oof_score = f1_score(y_train.iloc[val_idx], np.clip(np.round(site_oof_predictions), 1, 2))
    oof_scores.append(oof_score)
    print(f'\nFold {fold} - F1 Score {oof_score:.6}\n')

    df['site_predicted'] = df['site_predicted'].astype(np.float32)
    site_f1_score = f1_score(df.loc[df['site'] > 0, 'site'],
                            np.clip(np.round(df.loc[df['site'] > 0, 'site_predicted']), 1, 2))

    print('-' * 38)
    print(f'LightGBM Site Classifier Model Mean F1 Score {np.mean(oof_scores):.6} [STD:{np.std(oof_scores):.6}]')
    print(f'LightGBM Site Classifier Model OOF F1 Score {site_f1_score:.6}') #<

    plt.figure(figsize=(20, 20))
    feature_importance['Mean_Importance'] = feature_importance.sum(axis=1) / K
    feature_importance.sort_values(by='Mean_Importance', inplace=True, ascending=False)
    sns.barplot(x='Mean_Importance', y=feature_importance.index, data=feature_importance)

    plt.xlabel('')
    plt.tick_params(axis='x', labelsize=18)
    plt.tick_params(axis='y', labelsize=18)
    plt.title('LightGBM Site Classifier Model Feature Importance (Gain)', size=20, pad=20)

    plt.show()
##
df = df.astype('float32')

## pd.read_pickle()
# df.to_pickle('trends_tabular_data.pkl')
print(f'TReNDS Tabular Data Shape = {df.shape}')
print(f'TReNDS Tabular Data Memory Usage = {df.memory_usage().sum() / 1024 ** 2:.2f} MB')

features = [f for f in df.columns.tolist() if not f in target_features]
train = df.loc[df.is_train == 1, :]
test = df.loc[~(df.is_train==1), :]

# Neural netwrks are fairly resistant to noise - that's one of its big advantages.
train.loc[:,target_features].isnull().sum()
display(train.loc[train.domain1_var1.isnull(), target_features])
display(train.loc[train.domain2_var1.isnull(), target_features])

# check boxplot to see the distribution
train.boxplot(column=target_features.tolist())
plt.show()
train.loc[:,target_features] = train.loc[:,target_features].fillna(value=-100.0) # write out the 'value' parameter.
train.loc[train.domain1_var1.isnull(),target_features] # check to see nothing missing now.

##
train.to_pickle('trends_train_data.pkl')
test.to_pickle('trends_test_data.pkl')

with open('trends_variables.pkl', 'wb') as f:
    pickle.dump([target_features, fnc_features, loading_features, site_predictors, features], f)

train = pd.read_pickle('trends_train_data.pkl')
test = pd.read_pickle('trends_test_data.pkl')
##
with open('trends_variables.pkl','rb') as f:
    target_features, fnc_features, loading_features, site_predictors, features = pickle.load(f)













