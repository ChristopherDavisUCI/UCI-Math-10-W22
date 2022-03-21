#!/usr/bin/env python
# coding: utf-8

# # Predict Pumpkin Seeds
# 
# Author: Zian Dong
# 
# Course Project, UC Irvine, Math 10, W22

# Student ID: 90294322
# 

# ## Introduction
# 
# Introduce your project here.  About 3 sentences.
# 
# The goal of my project is to predict the category of the pumpkin seeds given by a series of input data, like its perimeter, compactness, area and so on. The dataset I use only contain two category of the pumpkin seeds, so I use the Logistic Regression as my training model. Besides, I also try to find the relationship between each variable and the final output, and choose two most significant variables to plot a relationship chart. 

# ## Main portion of the project
# 
# (You can either have all one section or divide into multiple sections)

# In[1]:


pip install openpyxl 


# In[2]:


import pandas as pd
import altair as alt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
import torch.nn as nn
import numpy as np


# ## Import data

# In[3]:


df = pd.read_excel("Pumpkin_Seeds_Dataset.xlsx")


# In[4]:


df.columns


# ## Feature selection
# Select important features. Eliminate highly related x variables. In such a way we can prevent the model from overfitting to some extent.

# In[5]:


X_unsel = df[['Area', 'Perimeter', 'Major_Axis_Length', 'Minor_Axis_Length',
       'Convex_Area', 'Equiv_Diameter', 'Eccentricity', 'Solidity', 'Extent',
       'Roundness', 'Aspect_Ration', 'Compactness' ]]


# In[6]:


corr = X_unsel.corr()

corr.style.background_gradient(cmap='coolwarm')


# We saw Area, Equv_Diameter and Convex Area, Perimeter, and Major Axis Length are closely related. Compactness, Aspect_Ration, Roundness, and Eccentrity are closely related. (abs of corr > 0.9).Thus, We can just pick Perimeter and Compactness as two representative features from these six features. All other features do not have such close relationship. We can leave them unchanged.

# In[7]:


X = df[['Perimeter', 'Minor_Axis_Length', 'Solidity', 'Extent', 'Compactness' ]]
Y = df['Class']


# ## Check imbalanced data
# Next, we check whether it's balanced data set. If it's inbalanced, we cannot simply use the (correct_num_of_pred / total_num) to judge whether the model has a good performance.The result shows it's approximately balanced.

# In[8]:


Y.value_counts()


# ## Standardize the data

# In[9]:


scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)


# In[10]:


X_scaled


# ## Build training and test set

# In[11]:


X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y, test_size=0.2)


# ## Train the model

# In[12]:


clf = LogisticRegression()


# In[13]:


clf.fit(X_train, y_train)


# In[14]:


clf.predict_proba(X_train)


# In[15]:


clf.predict(X_train)


# ## Check whether overfits

# In[16]:


train_error = log_loss(y_train, clf.predict_proba(X_train))
test_error = log_loss(y_test, clf.predict_proba(X_test))


# In[17]:


train_error


# In[18]:


test_error


# In[19]:


test_accuracy =np.count_nonzero(clf.predict(X_test) == y_test)/len(X_test)
train_accuracy =np.count_nonzero(clf.predict(X_train) == y_train)/len(X_train)


# In[20]:


test_accuracy


# In[21]:


train_accuracy


# In[22]:


print(f"The log error for the training set is {train_error}, and the log error for the test set is {test_error}")
print(f"The accuracy for the training set is {train_accuracy}, and the accuracy for the test set is {test_accuracy}")


# ## Visualize the feature importance

# In[23]:


list(clf.coef_[0])


# In[24]:


df_coef = pd.DataFrame()
df_coef['feature'] = ['Perimeter', 'Minor_Axis_Length', 'Solidity', 'Extent', 'Compactness']
df_coef['value'] = list(abs(clf.coef_[0]))
df_coef['type'] = ['negative' if n < 0 else 'positive'for n in clf.coef_[0]]


# In[25]:


df_coef


# In[26]:


alt.Chart(df_coef).mark_bar().encode(
    x = 'feature',
    y = 'value',
    color = 'type',
    opacity=alt.value(0.5),
).properties(
    title = 'Feature Importance'
)


# Finally, we can plot the relationship chart using the two most important features, which are compactness and Perimeter, and find the relationship between these two x variables and y variables.

# In[27]:


tmp = []
for i in y_test.index:
    tmp.append(df['Compactness'][i])

tmp2 = []
for i in y_test.index:
    tmp2.append(df['Perimeter'][i])


# In[28]:


len(tmp2)


# In[29]:


df_pred = pd.DataFrame()
df_pred['Compactness'] = tmp
df_pred['Perimeter'] = tmp2
df_pred['type'] = clf.predict(X_test)


# In[30]:


df_pred


# In[31]:


c1 = alt.Chart(df_pred).mark_circle().encode(
    x = 'Compactness',
    y = 'Perimeter',
    color = 'type'   
).properties(
    title = 'pred'
)


# In[32]:


df_true = pd.DataFrame()
df_true['Compactness'] = tmp
df_true['Perimeter'] = tmp2
df_true['type'] = list(y_test)


# In[33]:


c2 = alt.Chart(df_true).mark_circle().encode(
    x = 'Compactness',
    y = 'Perimeter',
    color = 'type'   
).properties(
    title = 'true'
)


# In[34]:


c1|c2


# ## Summary
# 
# Either summarize what you did, or summarize the results.  About 3 sentences.
# 
# I trained a  Logistic Regression model to predict the category of pumpkin seeds. And the result showed the model approximately had 87.7% accuracy on the training dataset, and 88% on the test dataset. We also compared the importance of each variable on the final output, and find Compactness and Perimeter are the most two important ones. More specifically, compactness has much more impartance than the perimeter in determining the category of the pumpkin seeds. And we can verify that with the compactness-perimeter relation chart: when compactness is bigger then 0.7, the seeds are very likely to be Cercevelik, and when it's smaller than 0.7, the seeds are likely to be Urgup Sivrisi, but perimeter doesn't have such a clear boundary. And besides, we can also verify from the chart that compactness and perimeter are negatively correlated which is consistent with their feature importance.
# 

# ## References
# 
# Include references that you found helpful.  Also say where you found the dataset you used.
# 
# https://www.kaggle.com/mkoklu42/pumpkin-seeds-dataset

# 

# <a style='text-decoration:none;line-height:16px;display:flex;color:#5B5B62;padding:10px;justify-content:end;' href='https://deepnote.com?utm_source=created-in-deepnote-cell&projectId=384e5b91-32d3-4393-9938-a43d5229d6d7' target="_blank">
# <img alt='Created in deepnote.com' style='display:inline;max-height:16px;margin:0px;margin-right:7.5px;' src='data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iODBweCIgaGVpZ2h0PSI4MHB4IiB2aWV3Qm94PSIwIDAgODAgODAiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8IS0tIEdlbmVyYXRvcjogU2tldGNoIDU0LjEgKDc2NDkwKSAtIGh0dHBzOi8vc2tldGNoYXBwLmNvbSAtLT4KICAgIDx0aXRsZT5Hcm91cCAzPC90aXRsZT4KICAgIDxkZXNjPkNyZWF0ZWQgd2l0aCBTa2V0Y2guPC9kZXNjPgogICAgPGcgaWQ9IkxhbmRpbmciIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJBcnRib2FyZCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTEyMzUuMDAwMDAwLCAtNzkuMDAwMDAwKSI+CiAgICAgICAgICAgIDxnIGlkPSJHcm91cC0zIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxMjM1LjAwMDAwMCwgNzkuMDAwMDAwKSI+CiAgICAgICAgICAgICAgICA8cG9seWdvbiBpZD0iUGF0aC0yMCIgZmlsbD0iIzAyNjVCNCIgcG9pbnRzPSIyLjM3NjIzNzYyIDgwIDM4LjA0NzY2NjcgODAgNTcuODIxNzgyMiA3My44MDU3NTkyIDU3LjgyMTc4MjIgMzIuNzU5MjczOSAzOS4xNDAyMjc4IDMxLjY4MzE2ODMiPjwvcG9seWdvbj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0zNS4wMDc3MTgsODAgQzQyLjkwNjIwMDcsNzYuNDU0OTM1OCA0Ny41NjQ5MTY3LDcxLjU0MjI2NzEgNDguOTgzODY2LDY1LjI2MTk5MzkgQzUxLjExMjI4OTksNTUuODQxNTg0MiA0MS42NzcxNzk1LDQ5LjIxMjIyODQgMjUuNjIzOTg0Niw0OS4yMTIyMjg0IEMyNS40ODQ5Mjg5LDQ5LjEyNjg0NDggMjkuODI2MTI5Niw0My4yODM4MjQ4IDM4LjY0NzU4NjksMzEuNjgzMTY4MyBMNzIuODcxMjg3MSwzMi41NTQ0MjUgTDY1LjI4MDk3Myw2Ny42NzYzNDIxIEw1MS4xMTIyODk5LDc3LjM3NjE0NCBMMzUuMDA3NzE4LDgwIFoiIGlkPSJQYXRoLTIyIiBmaWxsPSIjMDAyODY4Ij48L3BhdGg+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMCwzNy43MzA0NDA1IEwyNy4xMTQ1MzcsMC4yNTcxMTE0MzYgQzYyLjM3MTUxMjMsLTEuOTkwNzE3MDEgODAsMTAuNTAwMzkyNyA4MCwzNy43MzA0NDA1IEM4MCw2NC45NjA0ODgyIDY0Ljc3NjUwMzgsNzkuMDUwMzQxNCAzNC4zMjk1MTEzLDgwIEM0Ny4wNTUzNDg5LDc3LjU2NzA4MDggNTMuNDE4MjY3Nyw3MC4zMTM2MTAzIDUzLjQxODI2NzcsNTguMjM5NTg4NSBDNTMuNDE4MjY3Nyw0MC4xMjg1NTU3IDM2LjMwMzk1NDQsMzcuNzMwNDQwNSAyNS4yMjc0MTcsMzcuNzMwNDQwNSBDMTcuODQzMDU4NiwzNy43MzA0NDA1IDkuNDMzOTE5NjYsMzcuNzMwNDQwNSAwLDM3LjczMDQ0MDUgWiIgaWQ9IlBhdGgtMTkiIGZpbGw9IiMzNzkzRUYiPjwvcGF0aD4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+' > </img>
# Created in <span style='font-weight:600;margin-left:4px;'>Deepnote</span></a>
