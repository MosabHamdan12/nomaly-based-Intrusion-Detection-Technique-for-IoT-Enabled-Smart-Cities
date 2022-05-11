# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 00:40:20 2022

@author: lenovo
"""

import pandas as pd
from matplotlib import pyplot as plt 
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

train = pd.read_csv("UNSW_NB15_training-set.csv")
test = pd.read_csv("UNSW_NB15_testing-set.csv")
df = pd.concat([train, test], ignore_index=True)
df['proto']=le.fit_transform(df['proto'])
df['state']=le.fit_transform(df['state'])
df['service']=le.fit_transform(df['service'])
df.drop('state', axis=1, inplace=True)
df.drop('sload', axis=1, inplace=True)
df.drop('dload', axis=1, inplace=True)
df.drop('swin', axis=1, inplace=True)
df.drop('dwin', axis=1, inplace=True)
df.drop('stcpb', axis=1, inplace=True)
df.drop('dtcpb', axis=1, inplace=True)
df.drop('trans_depth', axis=1, inplace=True)
df.drop('rate', axis=1, inplace=True)
df.drop('djit', axis=1, inplace=True)
df.drop('sinpkt', axis=1, inplace=True)
df.drop('tcprtt', axis=1, inplace=True)
df.drop('synack', axis=1, inplace=True)
df.drop('ackdat', axis=1, inplace=True)
df.drop('is_sm_ips_ports', axis=1, inplace=True)
df.drop('ct_flw_http_mthd', axis=1, inplace=True)
df.drop('is_ftp_login', axis=1, inplace=True)
df.drop('ct_ftp_cmd', axis=1, inplace=True)
df.drop('ct_src_ltm', axis=1, inplace=True)
df.drop('id', axis=1, inplace=True)
#df.drop('dur', axis=1, inplace=True)
#df.drop('dmean', axis=1, inplace=True)
#df.drop('spkts', axis=1, inplace=True)
#df.drop('sjit', axis=1, inplace=True)
#df.drop('sloss', axis=1, inplace=True)
#df.drop('dttl', axis=1, inplace=True)
#df.drop('smean', axis=1, inplace=True)
print(df.shape)
#df.to_csv(r'D:\New folder\mosab\IG-Dataset1.csv', index=False)

X = df.drop(['attack_cat','label'], axis=1) # droped label
Y = df.loc[:,['label']]

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)
'''
from sklearn.ensemble import RandomForestClassifier
RFM_model= RandomForestClassifier(criterion =  'entropy',n_estimators=100,max_depth=10,random_state=33)
RFM_model.fit(X, Y)

from sklearn.svm import SVC
SVM_model=SVC(kernel = 'rbf', random_state = 0)
SVM_model.fit(X, Y)

from sklearn.tree import DecisionTreeClassifier
DT_model=DecisionTreeClassifier(criterion='entropy',max_depth=10,random_state=33)
DT_model.fit(X, Y)
'''
from sklearn.ensemble import GradientBoostingClassifier
GBC_model=GradientBoostingClassifier(n_estimators=100,max_depth=10,random_state=33)
GBC_model.fit(X, Y)
y_pred = GBC_model.predict(X_test)

from sklearn import metrics
#print ("Accuracy = ", metrics.accuracy_score(y_test, prediction_xgb))
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

print ("SVM Train Accuracy score   is :  ", metrics.accuracy_score(y_test, y_pred))
#print('RandomForestClassifierModel Train F1_Score is : \n' ,f1_score(y_test, y_prediction, average='micro'))
print('SVM classification report  is : \n' , classification_report(y_test, y_pred ))

#Confusion Matrix - verify accuracy of each class
from sklearn.metrics import confusion_matrix
import numpy as np
cm = confusion_matrix(y_test, y_pred)
print(cm)
import seaborn as sns
sns.heatmap(cm/np.sum(cm), annot=True,fmt='.2%')
