#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 10:26:15 2018

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
from sklearn.svm import SVC

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score  
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from mklaren.mkl.alignf import Alignf
from mklaren.kernel.kernel import linear_kernel, poly_kernel, rbf_kernel,sigmoid_kernel
from mklaren.kernel.kinterface import Kinterface
from sklearn.model_selection import GridSearchCV
from sklearn.metrics.pairwise import laplacian_kernel





data_path="~/Desktop/DS5220/project/"

train_raw=pd.read_csv( data_path+'train.csv', dtype={'ip':'int32','app':'uint8','device':'int8',
                  'os':'uint8','channel':'int16','is_attributed':'bool_'}
                    ,parse_dates=['click_time'], usecols=[0,1,2,3,4,5,7])#shrink according to data distribution

test_raw=pd.read_csv( data_path+'test.csv.zip', dtype={'ip':'int32','app':'uint8','device':'int8',
                  'os':'uint8','channel':'int16'}
                    ,parse_dates=['click_time'])
test_raw['click_time']=test_raw['click_time'].dt.hour.astype(np.int8) #turn into hour
train_raw['click_time']=train_raw['click_time'].dt.hour.astype(np.int8)


small_sample = train_raw.sample(n = int(len(train_raw.ix[:,0])/1000))
train_result = small_sample['is_attributed']

#dummy_app = pd.get_dummies(small_sample['app'],prefix = 'app')
#dummy_ip = pd.get_dummies(small_sample['ip'],prefix = 'ip')
#dummy_device = pd.get_dummies(small_sample['device'],prefix = 'device')
#dummy_os = pd.get_dummies(small_sample['os'],prefix = 'os')
#dummy_channel = pd.get_dummies(small_sample['channel'],prefix = 'channel')
#dummy_time = pd.get_dummies(small_sample['click_time'],prefix = 'click_time')
## Join dummy tables
#ggg = dummy_ip.join(dummy_os)
#ggg = ggg.join(dummy_device)
#ggg = ggg.join(dummy_app)
#ggg = ggg.join(dummy_channel)
#ggg = ggg.join(dummy_time)

# Fix the problem of imbalanced data
small_sample_false = small_sample.loc[small_sample['is_attributed'] == False]
small_sample_true = small_sample.loc[small_sample['is_attributed'] == True]
small_sample_false = small_sample_false.sample(n = 434)#number of True class
small_sample_balanced = pd.concat([small_sample_false, small_sample_true])
train_result_balanced = small_sample_balanced['is_attributed']
list_result = list(train_result_balanced)
for i in range(len(train_result_balanced)):
    if list_result[i] == True:
        list_result[i] = 1
    else:
        list_result[i] = -1
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




X_train, X_test, y_train, y_test = train_test_split(ggg_balanced, list_result,test_size=0.3, random_state=42)


# Normal SVM
svm_model =SVC()
svm_model.fit(X_train, y_train)
predict_svm = svm_model.predict(X_test)



#predict = LR_model.predict(ggg)
#print(np.linalg.norm(predict, train_result))
#print(metrics.accuracy_score(small_sample['is_attributed'], predict))


score=accuracy_score(y_test, svm_model.predict(X_test))
roc=roc_auc_score(y_test, svm_model.predict(X_test))
cr=classification_report(y_test, svm_model.predict(X_test))
print(cr)
print(score)
print(roc)




# Score
score_balanced=accuracy_score(y_test, svm_model.predict(X_test))
roc_balanced=roc_auc_score(y_test, svm_model.predict(X_test))
cr_balanced=classification_report(y_test, svm_model.predict(X_test))
print(cr_balanced)
print(score_balanced)
print(roc_balanced)



# After doing PCA test
pca = PCA(n_components = 0.9, svd_solver = 'full')
pca.fit(ggg_balanced)
pca_result = pca.transform(ggg_balanced)

X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(pca_result, list_result,test_size=0.3, random_state=42)


# Let's try ICA now





# Normal SVM
svm_model_pca =SVC()
svm_model_pca.fit(X_train_pca, y_train_pca)
predict_svm_pca = svm_model_pca.predict(X_test_pca)
# Score
score_balanced_pca=accuracy_score(y_test_pca, svm_model_pca.predict(X_test_pca))
roc_balanced_pca=roc_auc_score(y_test_pca, svm_model_pca.predict(X_test_pca))
cr_balanced_pca=classification_report(y_test_pca, svm_model_pca.predict(X_test_pca))
print(cr_balanced_pca)
print(score_balanced_pca)
print(roc_balanced_pca)

# linear kernel
svm_model_pca_linear =SVC(kernel = 'linear')
svm_model_pca_linear.fit(X_train_pca, y_train_pca)
predict_svm_pca_linear = svm_model_pca_linear.predict(X_test_pca)
# Score
score_balanced_pca_linear=accuracy_score(y_test_pca, svm_model_pca_linear.predict(X_test_pca))
roc_balanced_pca_linear=roc_auc_score(y_test_pca, svm_model_pca_linear.predict(X_test_pca))
cr_balanced_pca_linear=classification_report(y_test_pca, svm_model_pca_linear.predict(X_test_pca))
print(cr_balanced_pca_linear)
print(score_balanced_pca_linear)
print(roc_balanced_pca_linear)



# poly kernel
svm_model_pca_poly =SVC(kernel = 'poly',degree = 8, coef0 = 1)
svm_model_pca_poly.fit(X_train_pca, y_train_pca)
predict_svm_pca_poly = svm_model_pca_poly.predict(X_test_pca)
# Score
score_balanced_pca_poly=accuracy_score(y_test_pca, svm_model_pca_poly.predict(X_test_pca))
roc_balanced_pca_poly=roc_auc_score(y_test_pca, svm_model_pca_poly.predict(X_test_pca))
cr_balanced_pca_poly=classification_report(y_test_pca, svm_model_pca_poly.predict(X_test_pca))
print(cr_balanced_pca_poly)
print(score_balanced_pca_poly)
print(roc_balanced_pca_poly)

# rbf kernel
svm_model_pca_rbf =SVC(kernel = 'rbf',gamma = 0.01)
svm_model_pca_rbf.fit(X_train_pca, y_train_pca)
predict_svm_pca_rbf = svm_model_pca_rbf.predict(X_test_pca)
# Score
score_balanced_pca_rbf=accuracy_score(y_test_pca, svm_model_pca_rbf.predict(X_test_pca))
roc_balanced_pca_rbf=roc_auc_score(y_test_pca, svm_model_pca_rbf.predict(X_test_pca))
cr_balanced_pca_rbf=classification_report(y_test_pca, svm_model_pca_rbf.predict(X_test_pca))
print(cr_balanced_pca_rbf)
print(score_balanced_pca_rbf)
print(roc_balanced_pca_rbf)

# sigmoid kernel
svm_model_pca_sigmoid =SVC(kernel = 'sigmoid',coef0 = 0)
svm_model_pca_sigmoid.fit(X_train_pca, y_train_pca)
predict_svm_pca_sigmoid = svm_model_pca_sigmoid.predict(X_test_pca)
# Score
score_balanced_pca_sigmoid=accuracy_score(y_test_pca, svm_model_pca_sigmoid.predict(X_test_pca))
roc_balanced_pca_sigmoid=roc_auc_score(y_test_pca, svm_model_pca_sigmoid.predict(X_test_pca))
cr_balanced_pca_sigmoid=classification_report(y_test_pca, svm_model_pca_sigmoid.predict(X_test_pca))
print(cr_balanced_pca_sigmoid)
print(score_balanced_pca_sigmoid)
print(roc_balanced_pca_sigmoid)


# chi-square kernel, can't run, negative values exist
#K = chi2_kernel(X_train_pca, gamma=.5)
#svm_model_pca_chi2 =SVC(kernel = 'chi2_kernel')
#svm_model_pca_chi2.fit(X_train_pca, y_train_pca)
#predict_svm_pca_chi2 = svm_model_pca_chi2.predict(X_test_pca)
## Score
#score_balanced_pca_chi2=accuracy_score(y_test_pca, svm_model_pca_chi2.predict(X_test_pca))
#roc_balanced_pca_chi2=roc_auc_score(y_test_pca, svm_model_pca_chi2.predict(X_test_pca))
#cr_balanced_pca_chi2=classification_report(y_test_pca, svm_model_pca_chi2.predict(X_test_pca))
#print(cr_balanced_pca_chi2)
#print(score_balanced_pca_chi2)
#print(roc_balanced_pca_chi2)

# laplacian kernel
K = laplacian_kernel(X_train_pca)
svm_model_pca_laplacian = SVC(kernel='precomputed').fit(K, X_test_pca)
predict_svm_pca_laplacian = svm_model_pca_laplacian.predict(X_test_pca)
# Score
score_balanced_pca_laplacian=accuracy_score(y_train_pca, svm_model_pca_laplacian.predict(X_train_pca))
roc_balanced_pca_laplacian=roc_auc_score(y_train_pca, svm_model_pca_laplacian.predict(X_train_pca))
cr_balanced_pca_laplacian=classification_report(y_train_pca, svm_model_pca_laplacian.predict(X_train_pca))
print(cr_balanced_pca_laplacian)
print(score_balanced_pca_laplacian)
print(roc_balanced_pca_laplacian)

#Try 

# linear combination of different kernels:

K_exp  = Kinterface(data=X_train_pca, kernel=rbf_kernel,  kernel_args={"sigma": 30}) # RBF kernel 
K_poly = Kinterface(data=X_train_pca, kernel=poly_kernel, kernel_args={ })      # polynomial kernel with degree=3
K_lin  = Kinterface(data=X_train_pca, kernel=linear_kernel)                          # linear kernel
K_sig = Kinterface(data=X_train_pca, kernel=sigmoid_kernel)   

model = Alignf(typ="linear")
model.fit([K_exp, K_lin, K_poly], np.array(y_train_pca))

mu = model.mu
combined_kernel = lambda x, y: \
    mu[0] * K_exp(x, y) + mu[1] * K_lin(x, y) + mu[2] * K_poly(x, y)

svm_model_pca_multi =SVC(kernel = combined_kernel)
svm_model_pca_multi.fit(X_train_pca, y_train_pca)
predict_svm_pca_multi = svm_model_pca_multi.predict(X_test_pca)
# Score
score_balanced_pca_multi=accuracy_score(y_test_pca, svm_model_pca_multi.predict(X_test_pca))
roc_balanced_pca_multi=roc_auc_score(y_test_pca, svm_model_pca_multi.predict(X_test_pca))
cr_balanced_pca_multi=classification_report(y_test_pca, svm_model_pca_multi.predict(X_test_pca))
print(cr_balanced_pca_multi)
print(score_balanced_pca_multi)
print(roc_balanced_pca_multi)

#try convex combination
model_convex = Alignf(typ="convex")
model_convex.fit([K_exp, K_lin, K_poly], np.array(y_train_pca))

mu_convex = model_convex.mu
combined_kernel_convex = lambda x, y: \
    mu_convex[0] * K_exp(x, y) + mu_convex[1] * K_lin(x, y) + mu_convex[2] * K_poly(x, y)

svm_model_pca_multi_convex =SVC(kernel = combined_kernel_convex)
svm_model_pca_multi_convex.fit(X_train_pca, y_train_pca)
predict_svm_pca_multi_convex = svm_model_pca_multi_convex.predict(X_test_pca)
# Score
score_balanced_pca_multi_convex=accuracy_score(y_test_pca, svm_model_pca_multi_convex.predict(X_test_pca))
roc_balanced_pca_multi_convex=roc_auc_score(y_test_pca, svm_model_pca_multi_convex.predict(X_test_pca))
cr_balanced_pca_multi_convex=classification_report(y_test_pca, svm_model_pca_multi_convex.predict(X_test_pca))
print(cr_balanced_pca_multi_convex)
print(score_balanced_pca_multi_convex)
print(roc_balanced_pca_multi_convex)


















###
small_sample_raw = train_raw.sample(n = int(len(train_raw.ix[:,0])/10000))
train_result_raw = small_sample['is_attributed']

dummy_app_raw = pd.get_dummies(small_sample['app'],prefix = 'app')
dummy_ip_raw = pd.get_dummies(small_sample['ip'],prefix = 'ip')
dummy_device_raw = pd.get_dummies(small_sample['device'],prefix = 'device')
dummy_os_raw = pd.get_dummies(small_sample['os'],prefix = 'os')
dummy_channel_raw = pd.get_dummies(small_sample['channel'],prefix = 'channel')
dummy_time_raw = pd.get_dummies(small_sample['click_time'],prefix = 'click_time')
# Join dummy tables
ggg_raw = dummy_ip_raw.join(dummy_os_raw)
ggg_raw = ggg_raw.join(dummy_device_raw)
ggg_raw = ggg_raw.join(dummy_app_raw)
ggg_raw = ggg_raw.join(dummy_channel_raw)
ggg_raw = ggg_raw.join(dummy_time_raw)

ggg_raw.to_csv("sample_data", sep='\t')













