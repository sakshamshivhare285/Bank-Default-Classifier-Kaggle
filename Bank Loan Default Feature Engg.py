#!/usr/bin/env python
# coding: utf-8

# In[50]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.figure_factory as ff
import sklearn 
import os
from sklearn.preprocessing import OrdinalEncoder


# In[2]:


os.chdir('P:\\Data Sciece and machine learning projects\\DataSets\\Bank_default')


# In[3]:


data=pd.read_csv('train.csv')


# In[4]:


# checking the missing values
data.isnull().sum()


# In[5]:


# selecting the target and input feature
inputs=data.drop('Loan Status', axis=1)
targets=data['Loan Status']


# In[6]:


# selecting numeric and categorical feature
categorical_feature=[feature for feature in inputs.columns if inputs[feature].dtypes=='O']
numeric_feature=[feature for feature in inputs.columns if inputs[feature].dtypes != 'O']


# working first with catrgorical data

# In[38]:


fig=px.histogram(data,x='Grade',color='Loan Status')
fig.show()


# In[37]:


fig=px.histogram(data,x='Sub Grade',color='Loan Status')
fig.show()


# In[36]:


fig=px.histogram(data,x='Employment Duration',color='Loan Status')
fig.show()


# In[35]:


fig=px.histogram(data,x='Verification Status', color='Loan Status')
fig.show()


# In[39]:


fig=px.histogram(data,x='Initial List Status', color='Loan Status')
fig.show()


# In[40]:


fig=px.histogram(data,x='Application Type', color='Loan Status')
fig.show()


# In[44]:


sns.countplot(data['Loan Status'])


# In[45]:


# The above feature indicates that we have a imbalance data and we need to do resampling


# In[7]:


## function to lower down the cardinality in the categorical data
from collections import Counter
def cumulatively_categorise(column,threshold=0.4,return_categories_list=True):
  #Find the threshold value using the percentage and number of instances in the column
 # we can change the value of threshold to reduce the number of categories
  threshold_value=int(threshold*len(column))
  #Initialise an empty list for our new minimised categories
  categories_list=[]
  #Initialise a variable to calculate the sum of frequencies
  s=0
  #Create a counter dictionary of the form unique_value: frequency
  counts=Counter(column)

  #Loop through the category name and its corresponding frequency after sorting the categories by descending order of frequency
  for i,j in counts.most_common():
    #Add the frequency to the global sum
    s+=dict(counts)[i]
    #Append the category name to the list
    categories_list.append(i)
    #Check if the global sum has reached the threshold value, if so break the loop
    if s>=threshold_value:
        break
  #Append the category Other to the list
  categories_list.append('Other')

  #Replace all instances not in our new categories by Other  
  new_column=column.apply(lambda x: x if x in categories_list else 'Other')

  #Return transformed column and unique values if return_categories=True
  if(return_categories_list):
    return new_column,categories_list
  #Return only the transformed column if return_categories=False
  else:
    return new_column


# In[8]:


## function to work on categorical data
def categoricalFeatureEngg(data,categorical_feature):
    catdata=data[categorical_feature]
    
    # deleting the ID columns
    if ('Batch Enrolled') in catdata.columns:
        
        catdata.drop(['Batch Enrolled'], axis=1, inplace=True)
        
    for col in catdata.columns:   

        
        if len(catdata[col].unique()) in (2,3,4,5):
            dummies=pd.get_dummies(catdata[col], drop_first=True)
            catdata=pd.concat([catdata,dummies], axis=1)
            
            # dropping the columns when dumiess are addef
            catdata.drop(col, axis=1, inplace=True)
        elif (len(catdata[col].unique()) in range(6,11)):
            OE=OrdinalEncoder()
            X=OE.fit_transform(catdata[[col]])
            catdata[col]=X
        elif (len(catdata[col].unique()) in range(11,41)):
            # Cardinality reduction
            transformed_column,new_category_list=cumulatively_categorise(data[col],return_categories_list=True)
            catdata[col]=transformed_column
            # ordinal transform
            X=OE.fit_transform(catdata[[col]])
            catdata[col]=X
            
        elif len(catdata[col].unique()) > 40:
            # dropping these
            catdata.drop(col, axis=1, inplace=True)
        else:
            catdata.drop(col, axis=1, inplace=True)
            
    ## dropping the categorical data from the main dataframe    
    data=data.drop(categorical_feature, axis=1)
    # adding the processed categical data back to main data frame    
    Merged_data=pd.concat([data,catdata], axis=1)
            
    return Merged_data


# In[9]:


data_1=categoricalFeatureEngg(inputs,categorical_feature)


# In[10]:


data_1.info()


# In[11]:


# converting uint8 to int
uint8_list=[feature for feature in data_1.columns if data_1[feature].dtypes=='uint8']
for col in data_1[uint8_list]:
    data_1[col]=data_1[col].astype(int)


# In[56]:


data.info()


# Working on the Numeric data

# In[68]:


data_1.drop('ID',axis=1, inplace=True)


# In[47]:


# plotting loan amount
fig = px.histogram(data, x="Loan Amount", color="Loan Status",
                   marginal="box", # or violin, rug
                   hover_data=data.columns)
fig.show()


# In[48]:


#plotting the interest rates
# plotting loan amount
fig = px.histogram(data, x="Interest Rate", color="Loan Status",
                   marginal="box", 
                   hover_data=data.columns)
fig.show()


# In[55]:


#plotting the Open Account
# plotting loan amount
fig = px.histogram(data, x="Open Account", color="Loan Status",
                   marginal="box", 
                   hover_data=data.columns)
fig.show()


# In[57]:


#plotting the Open Account
# plotting loan amount
fig = px.histogram(data, x="Home Ownership", color="Loan Status",
                   marginal="box", 
                   hover_data=data.columns)
fig.show()


# In[58]:


# from above we can see that the data is not normally distributed


# In[69]:


# check for multi colinearity
# check for the Corelation 
from statsmodels.stats.outliers_influence import variance_inflation_factor
def vif_calc(X):
    # calculating vif
    vif=pd.DataFrame()
    vif['variables']=X.columns
    vif['VIF']=[variance_inflation_factor(X.values,i) for i in range(X.shape[1])]
    vif.set_index('variables',drop=True)
    return vif


# In[75]:


vif_calc(data_1)


# In[76]:


# removing the highly co related feature
vif_calc(data_1.drop('Term', axis=1))


# In[77]:


data_1.drop('Term', axis=1, inplace=True)


# In[78]:


# scaling the data
from sklearn.preprocessing import StandardScaler
STD=StandardScaler()

STD.fit(data_1)
scaled=STD.transform(data_1)
Inputs_scaled=pd.DataFrame(scaled, columns=data_1.columns)


# In[79]:





# In[80]:


Inputs_scaled.shape


# In[83]:


# Now doing the under sampling to get rid of imbalance nature of dataset
from imblearn.under_sampling import  NearMiss
# implementing NearMiss
nm=NearMiss()
X_res_un,Y_res_un=nm.fit_sample(Inputs_scaled,targets)


# In[84]:


# saving the preprocesse data
X_res_un.to_csv('inputs_preprocessed.csv')
Y_res_un.to_csv('targets.csv')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




