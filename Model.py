import numpy as np # linear algebra
from multiprocessing import Pool
import pandas as pd
#import seaborn as sns
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
from sklearn import metrics
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp
from sklearn.model_selection import train_test_split
from MLFeatureSelection import sequence_selection, importance_selection, coherence_selection,tools
from sklearn import metrics
from sklearn.metrics import roc_auc_score, roc_curve

df_test = pd.read_csv('test.csv')
df_test.drop(['ID_code'], axis=1, inplace=True)
df_test = df_test.values
unique_samples = []
unique_count = np.zeros_like(df_test)
for feature in range(df_test.shape[1]):
    _, index_, count_ = np.unique(df_test[:, feature], return_counts=True, return_index=True)
    unique_count[index_[count_ == 1], feature] += 1

# Samples which have unique values are real the others are fake
real_samples_indexes = np.argwhere(np.sum(unique_count, axis=1) > 0)[:, 0]
synthetic_samples_indexes = np.argwhere(np.sum(unique_count, axis=1) == 0)[:, 0]

tr = pd.read_csv('train.csv')
te = pd.read_csv('test.csv')

tr = pd.concat([tr, te.loc[real_samples_indexes]]).reset_index(drop = True)

def gaus(x,a,x0,sigma):
    return a*exp(-(x-x0)**2/(2*sigma**2))

def add_count(features):
    new = tr[features].value_counts().reset_index()
    new.columns = [features,'{}_count'.format(features)]
    popt,pcov = curve_fit(gaus,new[features],new['{}_count'.format(features)],p0=[1,tr[features].mean(),tr[features].std()])
    new['{}_count_norm'.format(features)] = gaus(new[features],*popt)
    new3 = tr.ix[tr.target == 1,features].value_counts().reset_index()
    new3.columns = [features,'{}_count'.format(features)]
    new3['{}_count'.format(features)] /= new3['{}_count'.format(features)].sum()
    popt3,pcov3 = curve_fit(gaus,new3[features],new3['{}_count'.format(features)],p0=[1,tr.ix[tr.target == 1,features].mean(),tr.ix[tr.target == 1,features].std()])
    new['{}_ratio'.format(features)] = new['{}_count'.format(features)] - new['{}_count_norm'.format(features)]
    new['{}_target_diff_1'.format(features)] = new['{}_count'.format(features)] / new['{}_count'.format(features)].sum() - gaus(new[features],*popt3)
    df = tr[[features]].merge(new[[features,'{}_ratio'.format(features), '{}_count'.format(features), '{}_target_diff_1'.format(features)]], on = features, how = 'left')
    df['{}_sum_0'.format(features)] = ((df[features] - df[features].mean()) * df['{}_count'.format(features)].map(lambda x: int(x > 1))).astype(np.float32)
    thres = df['{}_ratio'.format(features)].mean()
    df['{}_sum_1'.format(features)] = ((df[features] - df[features].mean()) * df['{}_ratio'.format(features)].map(lambda x: int(x > thres))).astype(np.float32)
    thres = df['{}_target_diff_1'.format(features)].mean()
    df['{}_sum_2'.format(features)] = ((df[features] - df[features].mean()) * df['{}_target_diff_1'.format(features)].map(lambda x: int(x > thres))).astype(np.float32)
    return df[['{}_ratio'.format(features), '{}_count'.format(features), '{}_target_diff_1'.format(features), '{}_sum_0'.format(features), '{}_sum_1'.format(features),'{}_sum_2'.format(features)]]

with Pool(16) as p:
    fea = p.map(add_count, ['var_{}'.format(i) for i in range(200)])

tr = pd.concat([tr,pd.concat(fea, axis=1)], axis=1)

X_train, X_test, y_train, y_test = train_test_split(tr[~pd.isnull(tr.target)], tr[~pd.isnull(tr.target)].target, test_size=0.2, random_state=42)

param = {
        'num_leaves': 12,
        'max_bin': 256,
        'min_data_in_leaf': 11,
        'learning_rate': 0.01,
        'min_sum_hessian_in_leaf': 0.00245,
        'bagging_fraction': 1, 
        'bagging_freq': 5, 
        'feature_fraction': 0.008,
        'lambda_l1': 4.7,
        'lambda_l2': 1,
        'min_gain_to_split': 0.65,
        'max_depth': -1,
        'save_binary': False,
        'seed': 1337,
        'feature_fraction_seed': 1337,
        'bagging_seed': 1337,
        'drop_seed': 1337,
        'data_random_seed': 1337,
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'verbose': 0,
        'metric': 'auc',
        'is_unbalance': False,
        'boost_from_average': False,
    }


predictors =  [i for i in  X_train.columns.tolist()[2:] if ('target_diff_1' not in i)]
xg_train = lgb.Dataset(X_train[predictors].values,
                      label=y_train)
xg_valid = lgb.Dataset(X_test[predictors].values,
                       label=y_test)

#clf = lgb.train(param, xg_train, 50000, valid_sets = [xg_train, xg_valid], verbose_eval=1000, early_stopping_rounds = 1000)

test = te[['ID_code']]
test['target'] = 0
xg_train = lgb.Dataset(tr[~pd.isnull(tr.target)][predictors].values,
                      label=tr[~pd.isnull(tr.target)].target)

xg_pred = lgb.Dataset(tr[pd.isnull(tr.target)][predictors].values)

clf = lgb.train(param, xg_train, 13162, verbose_eval=100)
test.loc[real_samples_indexes,'target'] = clf.predict(xg_pred.data, num_iteration=13162)
#test[['ID_code','target']].to_csv('submission_no_scale_combine_delete_fake.csv', index = None)
test[['ID_code','target']].to_csv('submission.csv', index = None)

