#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import necessary libraries

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import statsmodels.formula.api as sm
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import warnings


# In[2]:


warnings.simplefilter(action='ignore', category=FutureWarning)
os.getcwd()


# In[7]:


# Load dataset

df = pd.read_csv(r"C:\Users\manis\Downloads\archive (3)\Advertising.csv")


# In[8]:


# View the first few rows of the dataset

df.head()


# In[9]:


# Get the column names of the dataset

df.columns


# In[10]:


# To rename the column 'Unnamed: 0' to 'Index'
df.rename(columns={'Unnamed: 0': 'Index'}, inplace=True)


# In[11]:


df


# In[12]:


# Get the shape of the dataset (rows, columns)

df.shape


# In[13]:


# Check information about the dataset, data types, and missing values

df.info()


# In[14]:


# Get statistical summary of the numerical columns

df.describe().T


# In[15]:


# Check for missing values in the dataset

df.isnull().values.any()
df.isnull().sum()


# In[16]:


# Scatter plots to check the linearity assumption between each independent variable (TV, Radio, Newspaper) and the dependent variable (Sales)

sns.pairplot(df, x_vars=["TV", "Radio", "Newspaper"], y_vars="Sales", kind="reg")


# In[17]:


# Histograms to check the normality assumption of the dependent variable (Sales)

df.hist(bins=20)


# In[18]:


# Linear regression plots to visualize the relationship between each independent variable and the dependent variable

sns.lmplot(x='TV', y='Sales', data=df)
sns.lmplot(x='Radio', y='Sales', data=df)
sns.lmplot(x='Newspaper',y= 'Sales', data=df)


# In[19]:


# Correlation Heatmap to check for multicollinearity among independent/dependent variables

corrmat = df.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmin=0, vmax=1, square=True, cmap="YlGnBu", ax=ax)
plt.show()


# In[20]:


# Model Preparation

X = df.drop('Sales', axis=1)
y = df[["Sales"]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=46)


# In[21]:


# Linear Regression Model

lin_model = sm.ols(formula="Sales ~ TV + Radio + Newspaper", data=df).fit()


# In[22]:


# Print the coefficients of the linear model

print(lin_model.params, "\n")


# In[23]:


# Print the summary of the linear regression model

print(lin_model.summary())


# In[24]:


# Evaluate the model

results = []
names = []


# In[29]:


# Define a list of models to evaluate

models = [('LinearRegression', LinearRegression())]


# In[30]:


# Loop through each model, fit it to the data, and calculate the RMSE

for name, model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    result = np.sqrt(mean_squared_error(y_test, y_pred))
    results.append(result)
    names.append(name)
    msg = "%s: %f" % (name, result)
    print(msg)


# In[31]:


# Make predictions on new data

new_data = pd.DataFrame({'TV': [100], 'Radio': [50], 'Newspaper': [25]})
predicted_sales = lin_model.predict(new_data)
print("Predicted Sales:", predicted_sales)


# In[32]:


# Make predictions on new data

new_data = pd.DataFrame({'TV': [25], 'Radio': [63], 'Newspaper': [80]})
predicted_sales = lin_model.predict(new_data)
print("Predicted Sales:", predicted_sales)


# In[ ]:




