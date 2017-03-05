# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
from __future__ import division

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.svm import OneClassSVM

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# load data
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

# remove constant columns
remove = []
for col in train.columns:
    if train[col].std() == 0:
        remove.append(col)

train.drop(remove, axis=1, inplace=True)
test.drop(remove, axis=1, inplace=True)

# remove duplicated columns
remove = []
c = train.columns
for i in range(len(c)-1):
    v = train[c[i]].values
    for j in range(i+1,len(c)):
        if np.array_equal(v,train[c[j]].values):
            remove.append(c[j])

train.drop(remove, axis=1, inplace=True)
test.drop(remove, axis=1, inplace=True)

# Replace -999999 in var3 column with most common value 2 
train = train.replace(-999999,2)
test = test.replace(-999999,2)

# Add feature that counts the number of zeros in a row
train['n0'] = (train.iloc[:,:-1]==0).sum(axis=1)
test['n0'] = (test.iloc[:,:-1]==0).sum(axis=1)

# var38mc == 1 when var38 has the most common value and 0 otherwise
# logvar38 is log transformed feature when var38mc is 0, zero otherwise
def splitinto2(df_):
	df_['var38mc'] = np.isclose(df_.var38, 117310.979016)
	df_['logvar38'] = df_.loc[~df_['var38mc'], 'var38'].map(np.log)
	df_.loc[df_['var38mc'], 'logvar38'] = 0
	# properly decide to make it 0 or something else?
splitinto2(train)
splitinto2(test)


y_train = train['TARGET'].values
X_train = train.drop(['ID','TARGET'], axis=1).values

id_test = test['ID']
X_test = test.drop(['ID'], axis=1).values

# length of dataset
len_train = len(X_train)
len_test  = len(X_test)

# classifier
clf = xgb.XGBClassifier(missing=np.nan, max_depth=5, n_estimators=350, learning_rate=0.03, nthread=4, subsample=0.95, colsample_bytree=0.85, seed=4242)

X_fit, X_eval, y_fit, y_eval= train_test_split(X_train, y_train, test_size=0.3)

# fitting
clf.fit(X_train, y_train, early_stopping_rounds=20, eval_metric="auc", eval_set=[(X_eval, y_eval)])

# print(clf.feature_importances_)
# plt.bar(range(len(clf.feature_importances_)), clf.feature_importances_)
# plt.show()

print('Overall AUC:', roc_auc_score(y_train, clf.predict_proba(X_train)[:,1]))

# predicting
y_pred= clf.predict_proba(X_test)[:,1]

submission = pd.DataFrame({"ID":id_test, "TARGET":y_pred})
submission.to_csv("../submission/submission.csv", index=False)

print('Completed!')