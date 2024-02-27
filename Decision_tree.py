# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 09:15:19 2024

@author: user
"""

import pandas as pd
df=pd.read_csv("salaries.csv")
df.head()
inputs=df.drop('salary_more_then_100k',axis='columns')
target=df['salary_more_then_100k']
from sklearn.preprocessing import LabelEncoder
le_company=LabelEncoder()
le_job=LabelEncoder()
le_degree=LabelEncoder()
inputs['company_n']=le_company.fit_transform(inputs['company'])
inputs['jobs_n']=le_job.fit_transform(inputs['job'])
inputs['degree_n']=le_degree.fit_transform(inputs['degree'])
inputs_n=inputs.drop(['company','job','degree'],axis='columns')
target
from sklearn import tree
model=tree.DecisionTreeClassifier()
model.fit(inputs_n,target)
#Is salary of Google computer engineer bachelors degree >100k?
model.predict([[2,1,0]])
#Is salary of Google computer engineer masters degree>100k?
model.predict([[2,1,1]])
