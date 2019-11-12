# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 15:20:33 2019

@author: pc
"""

# Import libraries
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split
from sklearn import tree
import graphviz
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

# Set Current directory
os.chdir("C:\\Users\\Vishal Tyagi\\Desktop\\R\\projects\\Santander")
os.getcwd()

#import data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
# explore data 
train.describe()
test.decribe()
# dimension of data 
train.shape
test.shape

# name of columns
list(train)
list(test)

# data detail
train.info()
test.info()

#################################  Missing value analysis  ###################
train.isnull().sum().sum()
test.isnull().sum().sum()


############################## outliers analysis ########################
cnames1 = train.columns[2:]
cnames2 = test.columns[1:]

#Detect and delete outliers from train data
for i in cnames1:
    q75, q25 = np.percentile(train.loc[:,i], [75 ,25])
    iqr = q75 - q25
    minimum = q25 - (iqr*1.5)
    maximum = q75 + (iqr*1.5)
    train = train.drop(train[train.loc[:,i] < minimum].index)
    train = train.drop(train[train.loc[:,i] > maximum].index)

#Detect and delete outliers from test data
for i in cnames2:
    q75, q25 = np.percentile(test.loc[:,i], [75 ,25])
    iqr = q75 - q25
    minimum = q25 - (iqr*1.5)
    maximum = q75 + (iqr*1.5)
    test = test.drop(test[test.loc[:,i] < minimum].index)
    test = test.drop(test[test.loc[:,i] > maximum].index)
 
########################################## Feature Selection ###########################
# Calculation of correlation between numerical variables
cnames1 = train.columns[2:]
 
df_num = train.loc[:,cnames1]
corr = df_num.corr()
print(corr)

# plotiing the heatmap
f, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),square=True, ax=ax)
plt.show()

######################################### Feature Scaling  ############################
train['target'].value_counts()
train['target'].value_counts().plot(kind="pie", figsize=(12,9), colormap="coolwarm")


print('Distribution plot for predictor variables')
plt.figure(figsize=(30, 185))
for i, col in enumerate(cnames1):
    plt.subplot(50, 4, i + 1)
    plt.hist(train[col])
    plt.title(col)
    
print('Distribution of predictor variables with respect to target variable')
plt.figure(figsize=(30, 185))
for i, col in enumerate(cnames1):
    plt.subplot(50, 4, i+1)
    plt.hist(train[train['target'] == 0][col], alpha = 0.5, label = '0', color = 'b')
    plt.hist(train[train['target'] == 1][col], alpha = 0.5, label = '1', color = 'r')
    plt.title(col)            
# Data is noramlly distributed,no need to use normalization and standarization techniques


##################################  Split into train and test data ##################

#Seperate target and predictor variables
y = train['target']

x = train.drop(['target', 'ID_code'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

#Model building

########################################  Logistic Regression  ################################

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train,y_train)
y_pred = logreg.predict(X_test)

#Model evaluation using confusion matrix
from sklearn import metrics
conf_matrix = metrics.confusion_matrix(y_test, y_pred)
conf_matrix

from matplotlib import pyplot as plt
class_names=[0,1] # name of classes
fig, ax = plt.subplots(figsize = (10,10))
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

# create heatmap
sns.heatmap(pd.DataFrame(conf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1, fontsize = 10)   
plt.ylabel('Actual label', fontsize =10)           
plt.xlabel('Predicted label', fontsize = 10)   

#Model Evaluation
print('Accuracy:', metrics.accuracy_score(y_test,y_pred))   #0.9170069970012852
print('Precision:', metrics.precision_score(y_test,y_pred))  #0.6906419180201083
print('Recall:', metrics.recall_score(y_test,y_pred))       #0.2627243306854957

# F-1 score
Accuracy =  metrics.accuracy_score(y_test,y_pred)
Precision = metrics.precision_score(y_test,y_pred)
Recall = metrics.recall_score(y_test,y_pred)
f1_score = 2*((Recall*Precision)/(Recall+Precision))
print(f1_score)
      
#Area under Receiver Operating Curve(AUC)
y_pred_prob = logreg.predict_proba(X_test)[::, 1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_prob)
auc = metrics.roc_auc_score(y_test, y_pred_prob)   #0.8564829425870722
plt.plot(fpr, tpr, label = 'data 1, auc ='+str(auc))
plt.legend(loc = 4)
plt.show()

##########################################  Decision Tree  #####################

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf = clf.fit(X_train,y_train)
y_pred_clf = clf.predict(X_test)

conf_matrix_clf = metrics.confusion_matrix(y_test, y_pred_clf)
conf_matrix_clf

class_names=[0,1] # name of classes
fig, ax = plt.subplots(figsize = (8,8))
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(conf_matrix_clf),
annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1, fontsize = 10)
plt.ylabel('Actual label', fontsize =10)
plt.xlabel('Predicted label', fontsize = 10)

#Model Evaluation
print('Accuracy:', metrics.accuracy_score(y_test,y_pred_clf))    #0.8422104812223333
print('Precision:', metrics.precision_score(y_test,y_pred_clf))  #0.19920769666100735
print('Recall:', metrics.recall_score(y_test,y_pred_clf))       #0.20711974110032363

# F-1 score
Accuracy =  metrics.accuracy_score(y_test,y_pred_clf)
Precision = metrics.precision_score(y_test,y_pred_clf)
Recall = metrics.recall_score(y_test,y_pred_clf)
f1_score = 2*((Recall*Precision)/(Recall+Precision))
print(f1_score) #0.20308668685994521

from sklearn.metrics import auc

fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_clf)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, color='blue', lw=2, label='SVM ROC area = %0.2f)' % roc_auc)
plt.legend(loc="lower right")
plt.show()

#Area under Receiver Operating Curve(AUC)
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_clf)
auc = metrics.roc_auc_score(y_test, y_pred_clf)
plt.plot(fpr, tpr, label = 'data 1, auc ='+str(auc))  #0.5588
plt.legend(loc = 4)
plt.show()


######################################## Naive Bayes Model  #######################
from sklearn.naive_bayes import GaussianNB

NB_model = GaussianNB().fit(X_train,y_train)
NB_predictions = NB_model.predict(X_test)
conf_matrix_NB = metrics.confusion_matrix(y_test, NB_predictions)
conf_matrix_NB

class_names=[0,1] # name of classes
fig, ax = plt.subplots(figsize = (8,8))
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(conf_matrix_NB),annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1, fontsize = 10)
plt.ylabel('Actual label', fontsize =10)
plt.xlabel('Predicted label', fontsize = 10)

#Model Evaluation
print('Accuracy:', metrics.accuracy_score(y_test,NB_predictions))   #0.9229187491075254
print('Precision:', metrics.precision_score(y_test,NB_predictions))  #0.7134502923976608
print('Recall:', metrics.recall_score(y_test,NB_predictions))        #0.3557888597258676

# F-1 score
Accuracy =  metrics.accuracy_score(y_test,NB_predictions)
Precision = metrics.precision_score(y_test,NB_predictions)
Recall = metrics.recall_score(y_test,NB_predictions)
f1_score = 2*((Recall*Precision)/(Recall+Precision))
print(f1_score)  #0.4613861386138614

#Area under Receiver Operating Curve(AUC)
fpr, tpr, _ = metrics.roc_curve(y_test, NB_predictions)
auc = metrics.roc_auc_score(y_test, NB_predictions)
plt.plot(fpr, tpr, label = 'data 1, auc ='+str(auc))
plt.legend(loc = 4)
plt.show()

# ### As we got best Accuracy And preccision with Naive Bayes Model we will use this Model to predict Fare

# test data
test.describe()
test.shape
# prediction on test data using Naive Bayes model;
predicted_fare=NB_model.predict(test)
# Saving predicted target in test data
test['predicted_target']=predicted_target

test.head(10)
# saving test data in our memory
test.to_csv("test_predicted.csv",index=False)
