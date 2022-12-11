# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 00:58:40 2022

@author: lenovo
"""

import pandas as pd
from matplotlib import pyplot as plt 
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

train = pd.read_csv(r"C:\Users\Osama\Desktop\new-Dataset-36-Normal-attacks.csv")
## test = pd.read_csv("UNSW_NB15_testing-set.csv")
df = pd.concat([train], ignore_index=True)
df['proto']=le.fit_transform(df['proto'])
df['state']=le.fit_transform(df['state'])
df['service']=le.fit_transform(df['service'])

X = df.drop(['attack_cat','label'], axis=1) # droped label
Y = df.loc[:,['label']]

'''
#Define the dependent variable that needs to be predicted (labels)
y = df["Label"].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
X = labelencoder.fit_transform(X) # M=1 and B=0
#################################################################
#Define x and normalize values

#Define the independent variables. Let's also drop Gender, so we can normalize other data
#X = df.drop(labels = ["Label", "ID"], axis=1) 
'''
import numpy as np
feature_names = np.array(X.columns)  #Convert dtype string?


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

##Split data into train and test to verify accuracy after fitting the model. 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

###########################################################################
# Define XGBOOST classifier to be used by Boruta
#import xgboost as xgb
#model = xgb.XGBClassifier()  #For Boruta
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()

"""
Create shadow features – random features and shuffle values in columns
Train Random Forest / XGBoost and calculate feature importance via mean decrease impurity
Check if real features have higher importance compared to shadow features 
Repeat this for every iteration
If original feature performed better, then mark it as important 
"""

from boruta import BorutaPy

# define Boruta feature selection method
feat_selector = BorutaPy(model, n_estimators='auto', verbose=2, random_state=1)

# find all relevant features
feat_selector.fit(X_train, y_train)


# check selected features
print(feat_selector.support_)  #Should we accept the feature

# check ranking of features
print(feat_selector.ranking_) #Rank 1 is the best

# call transform() on X to filter it down to selected features
X_filtered = feat_selector.transform(X_train)  #Apply feature selection and return transformed data

"""
Review the features
"""
# zip feature names, ranks, and decisions 
feature_ranks = list(zip(feature_names, 
                         feat_selector.ranking_, 
                         feat_selector.support_))

# print the results
for feat in feature_ranks:
    print('Feature: {:<30} Rank: {},  Keep: {}'.format(feat[0], feat[1], feat[2]))
    
    
############################################################
#Now use the subset of features to fit XGBoost model on training data
'''
import xgboost as xgb
xgb_model = xgb.XGBClassifier()

xgb_model.fit(X_filtered, y_train)

from sklearn.ensemble import RandomForestClassifier
RFM_model= RandomForestClassifier(criterion =  'entropy',n_estimators=100,max_depth=10,random_state=33)
RFM_model.fit(X_filtered, y_train)
#Now predict on test data using the trained model. 
'''
from sklearn.svm import SVC
SVM_model=SVC(kernel = 'rbf', random_state = 0)
SVM_model.fit(X_filtered, y_train)
#First apply feature selector transform to make sure same features are selected from test data
X_test_filtered = feat_selector.transform(X_test)
y_prediction = SVM_model.predict(X_test_filtered)

'''
from sklearn.tree import DecisionTreeClassifier
DT_model=DecisionTreeClassifier(criterion='entropy',max_depth=10,random_state=33)
DT_model.fit(X_filtered, y_train)
X_test_filtered = feat_selector.transform(X_test)
y_prediction = DT_model.predict(X_test_filtered)

from sklearn.ensemble import GradientBoostingClassifier
GBC_model=GradientBoostingClassifier(n_estimators=100,max_depth=10,random_state=33)
GBC_model.fit(X_filtered, y_train)

X_test_filtered = feat_selector.transform(X_test)
y_prediction = RFM_model.predict(X_test_filtered)
'''
#Print overall accuracy
from sklearn import metrics
#print ("Accuracy = ", metrics.accuracy_score(y_test, prediction_xgb))
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

print ("DT Train Accuracy score   is :  ", metrics.accuracy_score(y_test, y_prediction))
#print('RandomForestClassifierModel Train F1_Score is : \n' ,f1_score(y_test, y_prediction, average='micro'))
print('DT  classification report  is : \n' , classification_report(y_test, y_prediction ))



#Confusion Matrix - verify accuracy of each class
from sklearn.metrics import confusion_matrix
import seaborn as sns
cm = confusion_matrix(y_test, y_prediction)
print(cm)
TN = cm[0][0]
FN = cm[1][0]
TP = cm[1][1]
FP = cm[0][1]
TPR = TP/(TP+FN)
print ('TPR',TPR)   
# Specificity or true negative rate
TNR = TN/(TN+FP) 
print('TNR',TNR)
# Precision or Detection rate 
PPV = TP/(TP+FN)
print ('PPV', PPV)
# Negative predictive value
NPV = TN/(TN+FN)
print('NPV', NPV)
# Fall out or false alarm rate
FPR = FP/(FP+TN)
print('FPR',FPR)
# False negative rate
FNR = FN/(TP+FN)
print ('FNR',FNR)  
# False discovery rate
FDR = FP/(TP+FP)
print ('FDR',FDR)    
ax = sns.heatmap(cm/np.sum(cm), annot=True,fmt='.2%')
#ax.set_title('Seaborn Confusion Matrix with labels\n\n');
ax.set_xlabel('Predicted Value')
ax.set_ylabel('Actual Value ')
ax.xaxis.set_ticklabels(['Normal','Reconnaissance','Backdoor','DoS','Exploits','Analysis','Fuzzers','Generic','Worms','Shellcode'])
ax.yaxis.set_ticklabels(['Normal','Reconnaissance','Backdoor','DoS','Exploits','Analysis','Fuzzers','Generic','Worms','Shellcode'])

#ax = sns.heatmap(cm/np.sum(cm), annot=True,fmt='.2%')
plt.show()
'''
import seaborn as sns
#import matplotlib.pyplot as plt
#sns.heatmap(cm/np.sum(cm), center = True)
plt.show()
from sklearn.metrics import confusion_matrix
arwa= confusion_matrix(y_test, y_prediction)
print(arwa)
import seaborn as sns
sns.heatmap(arwa, center = True)
plt.show()   
#sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, 
            fmt='.2%', cmap='Blues')
'''