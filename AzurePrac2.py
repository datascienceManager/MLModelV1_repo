#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from azureml.core import Run

run = Run.get_context()

parser = argparse.ArgumentParser()

parser.add_argument('--min_samples_leaf',type=int)
parser.add_argument('--min_samples_split',type=int)
parser.add_argument('--max_depth',type=int)
parser.add_argument('--n_estimators',type=int)


args = parser.parse_args()

# In[2]:


data = pd.read_csv('fdata.csv')
data


# # Creating Independent and Dependent Variable

# In[3]:


data4Norm = data.select_dtypes(exclude = 'object')
data4Norm = data4Norm.drop(['Unnamed: 0'],axis=1)
data4Norm.info()


# In[4]:


dataScaled = data.copy()

# col = list(data4Norm.columns.values)
col = ['ApplicantIncome', 'CoapplicantIncome', 'Loan_Amount_Term', 'Credit_History']
print(col)
#run.log('ParametersUsed',col)

features = data[col]

# In[5]:


Normaliz = MinMaxScaler()

dataScaled[col] = Normaliz.fit_transform(features.values)

#dataScaled


# In[6]:


dataScaled = pd.get_dummies(dataScaled)



# # Creating Inde and Dependent Var

# In[7]:


x = dataScaled.drop(['LoanAmount','Unnamed: 0'] , axis=1)
y = dataScaled['LoanAmount']


# # Creating train and test dataset 

# In[8]:


xtrain,xtest,ytrain,ytest = train_test_split(x,y,train_size=0.7,random_state=27)


# In[9]:


#Regressor = RandomForestRegressor(n_estimators=200, n_jobs=-1,verbose=1,max_depth=5,oob_score=True)
Regressor = RandomForestRegressor(n_estimators=args.n_estimators,min_samples_leaf=args.min_samples_leaf,min_samples_split=args.min_samples_split, n_jobs=-1,verbose=1,max_depth=args.max_depth,oob_score=True)

# In[10]:


Regressor.fit(xtrain,ytrain)

joblib.dump(Regressor,'M:\\aazzureml\\RegressorModel.pkl')

# In[11]:


ypredict = Regressor.predict(xtest)


# In[12]:


FinalPred = pd.DataFrame({'Actual':ytest,'Predict':ypredict})




print('mean_absolute_error is ',mean_absolute_error(ytest,ypredict))

run.log('MAE',mean_absolute_error(ytest,ypredict))
# In[25]:


print('mean_squared_error is ',mean_squared_error(ytest,ypredict))

run.log('MSE',mean_squared_error(ytest,ypredict))
# In[26]:


print('mean_squared_error is ',np.sqrt(mean_squared_error(ytest,ypredict)))

run.log('RMSE',np.sqrt(mean_squared_error(ytest,ypredict)))
# In[27]:


# In[17]:


error = abs(ytest-ypredict)

MAPE = 100 *(error/ytest)

Accuracy = 100 - np.mean(MAPE)
Accuracy = round(Accuracy,2)

print('Accuracy' , Accuracy,"%")
run.log('Accuracy',Accuracy)

