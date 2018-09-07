#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 17:47:42 2018

@author: stian
"""

import pandas as pd
import lightgbm as lgb
import math
from sklearn.linear_model import LogisticRegression
from sklearn import metrics, cross_validation
from sklearn.preprocessing import LabelEncoder
import mat4py
import numpy as np
from sklearn.decomposition import PCA

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score  
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report



data_path="~/Desktop/DS5220/project/"

train_raw=pd.read_csv( data_path+'train.csv', dtype={'ip':'int32','app':'uint8','device':'int8',
                  'os':'uint8','channel':'int16','is_attributed':'bool_'}
                    ,parse_dates=['click_time'], usecols=[0,1,2,3,4,5,7])#shrink according to data distribution

test_raw=pd.read_csv( data_path+'test.csv.zip', dtype={'ip':'int32','app':'uint8','device':'int8',
                  'os':'uint8','channel':'int16'}
                    ,parse_dates=['click_time'])
test_raw['click_time']=test_raw['click_time'].dt.hour.astype(np.int8) #turn into hour
train_raw['click_time']=train_raw['click_time'].dt.hour.astype(np.int8)


#for col in ['ip','app','device','os','channel']: #turn in category
#    test[col]=test[col].astype('category')
#    train[col] = train[col].astype('category')
    




# Logistic Regression
#LR_model = LogisticRegression(class_weight='balanced')# Deal with the problem of imbalanced data
small_sample = train_raw.sample(n = int(len(train_raw.ix[:,0])/1000))
train_result = small_sample['is_attributed']
#for i in range(len(small_sample)):
#    if int(str(small_sample[i:i+1]['click_time']).split()[1]) in range(9, 18):
#        small_sample[i:i+1]['working_hour'] = 1
#    else:
#        small_sample[i:i+1]['working_hour'] = 0
#    print(i)
    
dummy_app = pd.get_dummies(small_sample['app'],prefix = 'app')
dummy_ip = pd.get_dummies(small_sample['ip'],prefix = 'ip')
dummy_device = pd.get_dummies(small_sample['device'],prefix = 'device')
dummy_os = pd.get_dummies(small_sample['os'],prefix = 'os')
dummy_channel = pd.get_dummies(small_sample['channel'],prefix = 'channel')
dummy_time = pd.get_dummies(small_sample['click_time'],prefix = 'click_time')
# Join dummy tables
ggg = dummy_ip.join(dummy_os)
ggg = ggg.join(dummy_device)
ggg = ggg.join(dummy_app)
ggg = ggg.join(dummy_channel)
ggg = ggg.join(dummy_time)


# Turn them into categorical feature


# Next, join them, do PCA or Lasso or other reduce dimension methods.

#small_sample = small_sample.apply(LabelEncoder().fit_transform)
#predicted = cross_validation.cross_val_predict(LogisticRegression(class_weight='balanced'), ggg,train_result, cv=10)
#print(metrics.accuracy_score(small_sample['is_attributed'], predicted))
#print(metrics.classification_report(small_sample['is_attributed'], predicted)) 


# Select k best features
#k_best_feature_ggg = SelectKBest(chi2, k=400).fit_transform(ggg, train_result)



# Split the dataset into train and test. Cross validation
X_train, X_test, y_train, y_test = train_test_split(ggg, train_result,train_size = 0.2, test_size=0.1, random_state=42)
#pca = PCA(n_components=1000)
#pca.fit(ggg)
#添加l1 penalty后， 效果有0.002的提升
LR_model = LogisticRegression(penalty = 'l2',class_weight = 'balanced')
LR_model.fit(X_train,y_train)
#predict = LR_model.predict(ggg)
#print(np.linalg.norm(predict, train_result))
#print(metrics.accuracy_score(small_sample['is_attributed'], predict))


score=accuracy_score(y_test, LR_model.predict(X_test))
roc=roc_auc_score(y_test, LR_model.predict(X_test))
cr=classification_report(y_test, LR_model.predict(X_test))
print(cr)


# Select same number False class as True class


small_sample_false = small_sample.loc[small_sample['is_attributed'] == False]
small_sample_true = small_sample.loc[small_sample['is_attributed'] == True]
small_sample_false = small_sample_false.sample(n = 470)
small_sample_balanced = pd.concat([small_sample_false, small_sample_true])
train_result_balanced = small_sample_balanced['is_attributed']

dummy_app_balanced = pd.get_dummies(small_sample_balanced['app'],prefix = 'app')
dummy_ip_balanced = pd.get_dummies(small_sample_balanced['ip'],prefix = 'ip')
dummy_device_balanced = pd.get_dummies(small_sample_balanced['device'],prefix = 'device')
dummy_os_balanced = pd.get_dummies(small_sample_balanced['os'],prefix = 'os')
dummy_channel_balanced = pd.get_dummies(small_sample_balanced['channel'],prefix = 'channel')
dummy_time_balanced = pd.get_dummies(small_sample_balanced['click_time'],prefix = 'click_time')
# Join dummy tables
ggg_balanced = dummy_ip_balanced.join(dummy_os_balanced)
ggg_balanced = ggg_balanced.join(dummy_device_balanced)
ggg_balanced = ggg_balanced.join(dummy_app_balanced)
ggg_balanced = ggg_balanced.join(dummy_channel_balanced)
ggg_balanced = ggg_balanced.join(dummy_time_balanced)


# Without using pca
X_train_balanced, X_test_balanced, y_train_balanced, y_test_balanced = train_test_split(ggg_balanced, train_result_balanced,test_size=0.3, random_state=32)
LR_model_balanced = LogisticRegression(penalty = 'l2',C = 1, tol = 0.01)
LR_model_balanced.fit(X_train_balanced,y_train_balanced)



score_balanced=accuracy_score(y_test_balanced, LR_model_balanced.predict(X_test_balanced))
roc_balanced=roc_auc_score(y_test_balanced, LR_model_balanced.predict(X_test_balanced))
cr_balanced=classification_report(y_test_balanced, LR_model_balanced.predict(X_test_balanced))
print(cr_balanced)
print(score_balanced)
print(roc_balanced)

# witout using pca cv result




# Using pca
ggg_np_balanced = ggg_balanced.values
ggg_np_balanced = np.nan_to_num(ggg_np_balanced)

pca = PCA(n_components = 0.95, svd_solver = 'full')
pca.fit(ggg_np_balanced)
pca_result_balanced = pca.transform(ggg_np_balanced)

X_train_balanced_pca, X_test_balanced_pca, y_train_balanced_pca, y_test_balanced_pca = train_test_split(pca_result_balanced, train_result_balanced,test_size=0.3, random_state=52)
LR_model_balanced_pca = LogisticRegression(penalty = 'l2',C = 1, tol = 0.001)
LR_model_balanced_pca.fit(X_train_balanced_pca,y_train_balanced_pca)

score_balanced_pca=accuracy_score(y_test_balanced_pca, LR_model_balanced_pca.predict(X_test_balanced_pca))
roc_balanced_pca=roc_auc_score(y_test_balanced_pca, LR_model_balanced_pca.predict(X_test_balanced_pca))
cr_balanced_pca=classification_report(y_test_balanced_pca, LR_model_balanced_pca.predict(X_test_balanced_pca))
print(cr_balanced_pca)
print(score_balanced_pca)
print(roc_balanced_pca)
































