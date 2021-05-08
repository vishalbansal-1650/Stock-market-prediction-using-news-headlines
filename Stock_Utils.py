# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 02:53:27 2021

@author: vishal
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score,plot_confusion_matrix,plot_roc_curve,precision_recall_curve
from sklearn.model_selection import cross_val_score


################################################ function for model fitting
def model_fit(estimator,x_train,y_train):
    return estimator.fit(x_train,y_train)

############################################ function for calc precision when recall is above given threshold
def precision_at_recall_threshold(y,probas_pred,recall_threshold):
    
    precision,recall,_ = precision_recall_curve(y,probas_pred)
    return precision[recall > recall_threshold][-1]

############################################## function for model building & scoring
def model_building(estimator,x_train,y_train,x_test,y_test,algo,Model_logging):

    model_dtls = pd.DataFrame()
    estimator = model_fit(estimator,x_train,y_train)
    
    model_dtls['Model'] = [str(algo)]
    model_dtls['Train_Score'] = [cross_val_score(estimator,x_train,y_train,cv=5).mean()]
    model_dtls['Test_Score'] = [cross_val_score(estimator,x_test,y_test,cv=5).mean()]
    
    y_train_pred = estimator.predict(x_train)
    y_test_pred = estimator.predict(x_test)
    y_test_proba = estimator.predict_proba(x_test)[:,1]
    
    model_dtls['ROC_AUC_Score'] = [np.round(roc_auc_score(y_test,y_test_pred),2)]

    
    ### calc precision at recall_threshold=0.85
    model_dtls['Precision_at_Recall'] = [precision_at_recall_threshold(y_test,y_test_proba,0.85)]
   
    Model_logging = pd.concat([Model_logging,model_dtls],axis=0)
    
    return estimator,y_train_pred,y_test_pred,Model_logging

 
###################################### function for generating all classification metrics at once
def custom_classification_metrics(estimator,x,y,y_pred,labels,dataset):
    
    ### accuracy score
    score = estimator.score(x,y)
    print(dataset," dataset has an accuracy of ",np.round(score,2)*100)
    
    ### classification report
    print(classification_report(y, y_pred,target_names=labels))
    
    ### ROC AUC Score
    y_scores = estimator.predict_proba(x)[:,1]
    roc_auc_val = np.round(roc_auc_score(y,y_scores),2)
    print("The ROC AUC Score on ",dataset," dataset : ",roc_auc_val)
    
    ### ROC AUC Curve
    plot_roc_curve(estimator,x,y);
    
    ### Confusion Matrix
    plot_confusion_matrix(estimator,x,y,display_labels=labels,values_format='',cmap=plt.cm.Oranges);
    



