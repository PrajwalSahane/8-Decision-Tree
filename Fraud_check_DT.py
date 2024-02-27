# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 16:47:20 2024

@author: user
"""

'''Business Objective:
    Define the business objective, which could be predicting
    the likelihood of diabetes based on various features in 
    the dataset.
    
 Data Dictionary:
     
Data Dictionary:
1)Undergrad: Whether the customer (Categorical: Yes/No)
2)Marital_Status: Marital status of the customer (Categorical: Single/Married/Divorced)
3)Taxable_Income: Taxable income of the customer(Continuous: Integer)
4)City_Population: Population of the city  (Continuous: Integer)
5)Work_Experience: Work experience of the custome (Continuous: Integer)
6)Urban: Whether the customer (Categorical: Yes/No)
7)Mortgage: Mortgage value in USD (Continuous: Integer)
8)House_Ownership: Type of house ownership (Categorical: Own/Rent)
9)Car_Ownership: Whether the customer owns a car or not (Categorical: Yes/No)
10)Income_Category: Income category of the customer (Categorical: Low/Medium/High)
11)Fraud_Taxable: Whether the customer's taxable income is fraudulent or not (Target Variable) (Categorical: Yes/No)"""
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
#2.Load the Dataset
df=pd.read_csv("Fraud_check.csv")
df.head(5)
df.tail(5)
df.isnull().sum()
df.info()
df.columns
df.describe()


for i in df["Taxable.Income"]:
    if (i<=30000):
        df["Taxable.Income"]=np.where(df["Taxable.Income"]==i,0,df["Taxable.Income"])
    else:
        df["Taxable.Income"]=np.where(df["Taxable.Income"]==i,1,df["Taxable.Income"])
            
le=LabelEncoder()
df["Undergrad"]=le.fit_transform(df["Undergrad"])
df["Marital.Status"]=le.fit_transform(df["Marital.Status"])
df["Urban"]=le.fit_transform(df["Urban"])
df.head()

df["Taxable.Income"].value_counts()
inputs=df.drop(["Taxable.Income"],axis=1)
target=df["Taxable.Income"]

x_train,x_test,y_train,y_test=train_test_split(inputs,target,test_size=0.2)

model=DecisionTreeClassifier()
model.fit(x_train,y_train)

y_pred=model.predict(x_test)
y_pred

acc=accuracy_score(y_test, y_pred)
acc