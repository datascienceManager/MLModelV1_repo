#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


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


Regressor = RandomForestRegressor(n_estimators=200, n_jobs=-1,verbose=1,max_depth=5,oob_score=True)


# In[10]:


Regressor.fit(xtrain,ytrain)


# In[11]:


ypredict = Regressor.predict(xtest)


# In[12]:


FinalPred = pd.DataFrame({'Actual':ytest,'Predict':ypredict})


# In[13]:


FinalPred


# In[ ]:





# In[14]:


print(mean_absolute_error(ytest,ypredict))


# In[15]:


print(mean_squared_error(ytest,ypredict))


# In[16]:


print(np.sqrt(mean_squared_error(ytest,ypredict)))


# In[17]:


error = abs(ytest-ypredict)

MAPE = 100 *(error/ytest)

Accuracy = 100 - np.mean(MAPE)
Accuracy = round(Accuracy,2)

print('Accuracy' , Accuracy,"%")


# # Hyperparameter tuning

# In[18]:


regressor2 = RandomForestRegressor(random_state=27,n_jobs=-1)


# In[19]:


params ={
    'max_depth':[3,7,10,15,20],
    'min_samples_leaf':[10,25,50,100,150],
    'n_estimators':[27,34,52,115,200]
}


# In[20]:


from sklearn.model_selection import GridSearchCV


# In[21]:


GS = GridSearchCV(estimator=regressor2,param_grid=params,cv=5,verbose=-1,n_jobs=-1,scoring='accuracy')


# In[22]:


GS.fit(xtrain,ytrain)


# In[23]:


ypredict = GS.predict(xtest)


# In[ ]:





# In[24]:


print('mean_absolute_error is ',mean_absolute_error(ytest,ypredict))


# In[25]:


print('mean_squared_error is ',mean_squared_error(ytest,ypredict))


# In[26]:


print('mean_squared_error is ',np.sqrt(mean_squared_error(ytest,ypredict)))


# In[27]:


Error = abs(ytest-ypredict)

MAPE = 100 *(Error/ytest)

Accuracy2 = 100 - np.mean(MAPE)
print('Accuracy' , Accuracy2,"%")

