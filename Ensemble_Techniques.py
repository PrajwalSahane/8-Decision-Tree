# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 14:55:47 2024

@author: user
"""
#Ensemble Technique

import pandas as pd






#0 500
#1 268
#There is slight imbalance in our dataset but since
#It is not major we will not worry about it
#Train test split
x=df.drop("Outcome",axis="columns")
y=df.Outcome
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x_scaled=scaler.fit_transform(x)
x_scaled[:3]
#In order to make your data balanced while splitting you can use stratify
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(x_scaled,y,stratify=y,random_state)










#Train using stand alone model
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
#Here k fold cross validation is used
scores=cross_val_score(DecisionTreeClassifier(),x,y,cv=5)
scores
scores.mean()
#Accuracy=0.7188778541
#Train using Bagging
from sklearn.ensemble import BaggingClassifier
bag_model=BaggingClassifier(
    base_estimator=DecisionTreeClassifier(),
    n_estimators=100,
    max_samples=0.8,
    oob_score=True,
    random_state=0)
bag_model.fit(X_train,y_train)
bag_model.oob_score_
#0.753472222222
#Note here we are not using test data,using OOB samples results are tested
bag_model.score(X_test,y_test)
#0.77604166666666
#Now let us apply cross validation
bag_model=BaggingClassifier(
    base_estimator=DecisionTreeClassifier(),
    n_estimatiors=100,
    max_samples=0.8,
    oob_score=True,
    random_state=0)
scores=cross_val_score(bag_model,X,y,cv=5)
scores
scores.mean()
#0.7578728461081402
#We can see some improvement in test score with bagging classifier as company

#Train using Random Forest
from sklearn.ensemble import RandomForestClassifier
scores=cross_val_score(RandomForestClassifier(n_estimators=50),X,y,cv=5)
scores.mean()
