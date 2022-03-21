#!/usr/bin/env python
# coding: utf-8

# # Analyses on NICS Firearm Background Checks

# Nathan Samarasena
# 
# Course Project, UC Irvine, Math 10, W22

# ## Introduction
# 
# For my project, I will be doing analyses on the 'nics-firearm-background-checks.csv' file from a BuzzFeed github repository.
# 
# I will be exploring how different columns within the data set can better predict certain aspects of given background checks, and whether or not there are better combinations of columns to analyze.
# 
# The data set is structured to describe the findings of all background checks done per month per state.

# ## Main portion of the project

# ### Importing Libraries and Data

# In[ ]:


import seaborn as sns
import numpy as np
import pandas as pd
import altair as alt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import log_loss


# In[ ]:


df = pd.read_csv('nics-firearm-background-checks.csv')
df.shape


# In[ ]:


df.columns


# In[ ]:


df


# In[ ]:


df['dt_month'] = pd.to_datetime(df['month']).dt.month
df['dt_year'] = pd.to_datetime(df['month']).dt.year


# ### More Handguns than other Firearms

# #### permit and permit_recheck

# In[ ]:


df2 = df[df['permit'].notna()]
df2 = df2[df['permit_recheck'].notna()]

df2['more_handgun'] = df['handgun'] > (df['long_gun'] + df['other'])

df2.shape


# In[ ]:


df2.columns


# In[ ]:


df2


# In[ ]:


X_colnames = ['permit','permit_recheck']
y_colname = 'more_handgun'
X = df2.loc[:,X_colnames].copy()
y = df2.loc[:,y_colname].copy()


# In[ ]:


scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)

clf = KNeighborsClassifier(n_neighbors = 10)
clf.fit(X_scaled,y)


# In[ ]:


X_scaled_train, X_scaled_test, y_train, y_test = train_test_split(X_scaled,y,test_size=0.4)

clf2 = KNeighborsClassifier(n_neighbors = 10)
clf2.fit(X_scaled_train,y_train)


# In[ ]:


probs = clf2.predict_proba(X_scaled_test)
log_loss(y_test,probs)


# In[ ]:


probs


# In[ ]:


df2['probsSer'] = pd.Series(probs[:,1])


# #### private_sale_handgun an private_sale_not_handgun

# In[ ]:


df3 = df[df['private_sale_handgun'].notna()]
df3 = df3[df['handgun'].notna()]
df3 = df3[df['long_gun'].notna()]
df3 = df3[df['other'].notna()]

df3['more_handgun'] = df['handgun'] > (df['long_gun'] + df['other'])

df3.shape


# In[ ]:


df3['private_sale_not_handgun'] = df2.loc[:,'private_sale_long_gun'] + df2.loc[:,'private_sale_other']


# In[ ]:


## X2_colnames = ['private_sale_handgun','private_sale_long_gun','private_sale_other']
X2_colnames = ['private_sale_handgun','private_sale_not_handgun']
y2_colname = 'more_handgun'
X2 = df2.loc[:,X_colnames].copy()
y2 = df2.loc[:,y_colname].copy()


# In[ ]:


scaler = StandardScaler()
scaler.fit(X2)
X2_scaled = scaler.transform(X2)

clf3 = KNeighborsClassifier(n_neighbors = 10)
clf3.fit(X2_scaled,y2)


# In[ ]:


X2_scaled_train, X2_scaled_test, y2_train, y2_test = train_test_split(X2_scaled,y2,test_size=0.4)

clf4 = KNeighborsClassifier(n_neighbors = 10)
clf4.fit(X2_scaled_train,y2_train)


# In[ ]:


probs2 = clf4.predict_proba(X2_scaled_test)
log_loss(y2_test,probs2)


# In[ ]:


df3['probs2Ser'] = pd.Series(probs2[:,1])


# ### State

# #### handgun and long_gun

# In[ ]:


df4 = df[df['state'].notna()]
df4 = df4[df4['handgun'].notna()]
df4 = df4[df4['long_gun'].notna()]

df4.shape


# In[ ]:


X3_colnames = ['handgun','long_gun']
y3_colname = 'state'
X3 = df4.loc[:,X3_colnames].copy()
y3 = df4.loc[:,y3_colname].copy()


# In[ ]:


scaler = StandardScaler()
scaler.fit(X3)
X3_scaled = scaler.transform(X3)

clf5 = KNeighborsClassifier(n_neighbors = 5)
clf5.fit(X3_scaled,y3)


# In[ ]:


X3_scaled_train, X3_scaled_test, y3_train, y3_test = train_test_split(X3_scaled,y3,test_size=0.4)

clf6 = KNeighborsClassifier(n_neighbors = 5)
clf6.fit(X3_scaled_train,y3_train)


# In[ ]:


probs3 = clf6.predict_proba(X3_scaled_test)
log_loss(y3_test,probs3)


# ### Graphing

# #### permit and permit_recheck for predicting when there are more handguns

# In[ ]:


alt.data_transformers.disable_max_rows()
graph = alt.Chart(df2).mark_bar().encode(
    x = 'more_handgun',
    y = 'permit',
    color = 'more_handgun',
    tooltip = ['more_handgun']
)

graph2 = alt.Chart(df2).mark_bar().encode(
    x = 'more_handgun',
    y = 'permit_recheck',
    color = 'more_handgun',
    tooltip = ['more_handgun']
)

graph|graph2


# In[ ]:


alt.data_transformers.disable_max_rows()
graph3 = alt.Chart(df2).mark_bar().encode(
    x = 'more_handgun',
    y = 'permit',
    color = 'probsSer',
    tooltip = ['probsSer','more_handgun']
)

alt.data_transformers.disable_max_rows()
graph4 = alt.Chart(df2).mark_bar().encode(
    x = 'more_handgun',
    y = 'permit',
    color = 'probsSer',
    tooltip = ['probsSer','more_handgun']
)

graph3|graph4


# #### private_sale_handgun an private_sale_not_handgun for predicting when there are more handguns

# In[ ]:


alt.data_transformers.disable_max_rows()
graph5 = alt.Chart(df3).mark_bar().encode(
    x = 'more_handgun',
    y = 'private_sale_handgun',
    color = 'more_handgun',
    tooltip = ['more_handgun']
)

graph6 = alt.Chart(df3).mark_bar().encode(
    x = 'more_handgun',
    y = 'private_sale_not_handgun',
    color = 'more_handgun',
    tooltip = ['more_handgun']
)

graph5|graph6


# In[ ]:


alt.data_transformers.disable_max_rows()
graph7 = alt.Chart(df3).mark_bar().encode(
    x = 'more_handgun',
    y = 'private_sale_handgun',
    color = 'probs2Ser',
    tooltip = ['probs2Ser','more_handgun']
)

alt.data_transformers.disable_max_rows()
graph8 = alt.Chart(df3).mark_bar().encode(
    x = 'more_handgun',
    y = 'private_sale_not_handgun',
    color = 'probs2Ser',
    tooltip = ['probs2Ser','more_handgun']
)

graph7|graph8


# ## Accompanying Documentation

# ###Importing Libraries and Data
# 
# To start, we will import all required libraries.
# 
# Next, we will import the nics-firearm-background-checks.csv file into the project to be analyzed, initializing it as a DataFrame df. We take note of the shape as well as the columns of df.
# 
# We will also create some datetime columns to be used later in the project.
# 
# ### More Handguns than other Firearms
# From there, we will create a new DataFrame for each new comparison between columns, as we may need to remove different NA data rows depending on what we are analyzing.
# 
# To start, we will find a good option to predict whether the given background checks have more handguns as opposed to long guns and other types of firearms. We use KNeighborsClassifier for this data. My initial idea was to use how many permits and permit rechecks were seen in the background checks, but my log_loss was fairly high, with the number being around 0.9, so this might not be the best for predicting when the background checks have more handguns. 
# 
# This time we try to use private_sale_handgun and a new column that we just made called private_sale_not_handgun. After running through it just like before with KNeighborsClassifier, we get a lower log_loss of approximately 0.75.
# 
# ### State
# 
# To predict state, I went with 
# 
# When trying to predict the state from the data of background checks, we run into a few problems. Namely, based on the nature of how states' populations tend to treat gun control and the stigma around owning firearms, it is highly likely that our program would mistake many states for others. We also cannot replace one of the comparative columns with another, as they will yield similar results.
# 
# ### Graphing
# 
# For graphing, I decided to show what each column looked like with respect to what we were trying to predict. I used a bar charts through Altair.
# 
# Earlier in the project I also created columns to represent the probability for there to be more handguns with probsSer and probs2Ser to allow for a bar graph that would show more insightful information for each data point.

# ## Summary
# 
# Through this project, I unfortunately found that making predictions based on the data provided is much more difficult than I initially thought. The log_loss was the biggest indicator of whether or not the insight provided by using machine learning was useful, as the numbers were uncomfortably high even after adjusting test size and the n_neighbors significantly (within reason for the size of the data). However, the use of probsSer told me a lot about how well the machine learning works when compared directly with the data.

# ## References
# 
# https://github.com/BuzzFeedNews/nics-firearm-background-checks

# <a style='text-decoration:none;line-height:16px;display:flex;color:#5B5B62;padding:10px;justify-content:end;' href='https://deepnote.com?utm_source=created-in-deepnote-cell&projectId=f4a3faa7-4f6d-4720-abea-1143c65c1b68' target="_blank">
# <img alt='Created in deepnote.com' style='display:inline;max-height:16px;margin:0px;margin-right:7.5px;' src='data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iODBweCIgaGVpZ2h0PSI4MHB4IiB2aWV3Qm94PSIwIDAgODAgODAiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8IS0tIEdlbmVyYXRvcjogU2tldGNoIDU0LjEgKDc2NDkwKSAtIGh0dHBzOi8vc2tldGNoYXBwLmNvbSAtLT4KICAgIDx0aXRsZT5Hcm91cCAzPC90aXRsZT4KICAgIDxkZXNjPkNyZWF0ZWQgd2l0aCBTa2V0Y2guPC9kZXNjPgogICAgPGcgaWQ9IkxhbmRpbmciIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJBcnRib2FyZCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTEyMzUuMDAwMDAwLCAtNzkuMDAwMDAwKSI+CiAgICAgICAgICAgIDxnIGlkPSJHcm91cC0zIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxMjM1LjAwMDAwMCwgNzkuMDAwMDAwKSI+CiAgICAgICAgICAgICAgICA8cG9seWdvbiBpZD0iUGF0aC0yMCIgZmlsbD0iIzAyNjVCNCIgcG9pbnRzPSIyLjM3NjIzNzYyIDgwIDM4LjA0NzY2NjcgODAgNTcuODIxNzgyMiA3My44MDU3NTkyIDU3LjgyMTc4MjIgMzIuNzU5MjczOSAzOS4xNDAyMjc4IDMxLjY4MzE2ODMiPjwvcG9seWdvbj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0zNS4wMDc3MTgsODAgQzQyLjkwNjIwMDcsNzYuNDU0OTM1OCA0Ny41NjQ5MTY3LDcxLjU0MjI2NzEgNDguOTgzODY2LDY1LjI2MTk5MzkgQzUxLjExMjI4OTksNTUuODQxNTg0MiA0MS42NzcxNzk1LDQ5LjIxMjIyODQgMjUuNjIzOTg0Niw0OS4yMTIyMjg0IEMyNS40ODQ5Mjg5LDQ5LjEyNjg0NDggMjkuODI2MTI5Niw0My4yODM4MjQ4IDM4LjY0NzU4NjksMzEuNjgzMTY4MyBMNzIuODcxMjg3MSwzMi41NTQ0MjUgTDY1LjI4MDk3Myw2Ny42NzYzNDIxIEw1MS4xMTIyODk5LDc3LjM3NjE0NCBMMzUuMDA3NzE4LDgwIFoiIGlkPSJQYXRoLTIyIiBmaWxsPSIjMDAyODY4Ij48L3BhdGg+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMCwzNy43MzA0NDA1IEwyNy4xMTQ1MzcsMC4yNTcxMTE0MzYgQzYyLjM3MTUxMjMsLTEuOTkwNzE3MDEgODAsMTAuNTAwMzkyNyA4MCwzNy43MzA0NDA1IEM4MCw2NC45NjA0ODgyIDY0Ljc3NjUwMzgsNzkuMDUwMzQxNCAzNC4zMjk1MTEzLDgwIEM0Ny4wNTUzNDg5LDc3LjU2NzA4MDggNTMuNDE4MjY3Nyw3MC4zMTM2MTAzIDUzLjQxODI2NzcsNTguMjM5NTg4NSBDNTMuNDE4MjY3Nyw0MC4xMjg1NTU3IDM2LjMwMzk1NDQsMzcuNzMwNDQwNSAyNS4yMjc0MTcsMzcuNzMwNDQwNSBDMTcuODQzMDU4NiwzNy43MzA0NDA1IDkuNDMzOTE5NjYsMzcuNzMwNDQwNSAwLDM3LjczMDQ0MDUgWiIgaWQ9IlBhdGgtMTkiIGZpbGw9IiMzNzkzRUYiPjwvcGF0aD4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+' > </img>
# Created in <span style='font-weight:600;margin-left:4px;'>Deepnote</span></a>
