#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 17:47:42 2018

@author: stian
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import datetime
import math
import gc
from sklearn.linear_model import LogisticRegression
from sklearn import metrics, cross_validation
data_path="~/Desktop/DS5220/project/"

train=pd.read_csv( data_path+'train.csv', dtype={'ip':'int32','app':'uint8','device':'int8',
                  'os':'uint8','channel':'int16','is_attributed':'bool_'}
                    ,parse_dates=['click_time'], usecols=[0,1,2,3,4,5,7])#shrink according to data distribution

test=pd.read_csv( data_path+'test.csv.zip', dtype={'ip':'int32','app':'uint8','device':'int8',
                  'os':'uint8','channel':'int16'}
                    ,parse_dates=['click_time'])
test['click_time']=test['click_time'].dt.hour.astype(np.int8).astype('category') #turn into hour
train['click_time']=train['click_time'].dt.hour.astype(np.int8).astype('category')

for col in ['ip','app','device','os','channel']: #turn in category
    test[col]=test[col].astype('category')
    train[col] = train[col].astype('category')

# Dummy encoding
# 有毒，一跑就崩
# It seems like models in library fix the dummy encoding problem automatically
#dummy_app = pd.get_dummies(train['app'])
#dummy_

# Logistic Regression
#LR_model = LogisticRegression(class_weight='balanced')# Deal with the problem of imbalanced data
small_sample = train.sample(n = int(len(train.ix[:,0])/100))
predicted = cross_validation.cross_val_predict(LogisticRegression(), small_sample.ix[:,0:6],small_sample.ix[:,6], cv=10)
print(metrics.accuracy_score(small_sample['is_attributed'], predicted))
print(metrics.classification_report(small_sample['is_attributed'], predicted)) 
#LR_model = LR_model.fit(train.ix[:,0:6],train.ix[:,6])
# The following is lgbm part
"""
ids=test['click_id'].values
train_final = lgb.Dataset( train.drop(['is_attributed'],axis=1),
                        train['is_attributed'].values)
print("training")
params = {
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.3,
    'num_leaves': 75,
}
gc.collect()
model = lgb.train(params,train_set=train_final,valid_sets=train_final, verbose_eval=25)
score=model.predict(test.drop(['click_id'],axis=1))

subm = pd.DataFrame()
subm['click_id'] = ids
subm['is_attributed'] = score
subm.to_csv(data_path + 'submission_lgbm.csv.gz', compression = 'gzip', index=False, float_format = '%.5f')
"""