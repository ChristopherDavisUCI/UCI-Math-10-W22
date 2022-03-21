#!/usr/bin/env python
# coding: utf-8

# # Movie's Gross 
# 
# Author: Dana Albakri
# 
# Course Project, UC Irvine, Math 10, W22

# ## Introduction
# The main aspects we are going to explore in the Movies dataset would be the Rotten Tomatoes Ratings in respect to the highest gross. We will be using both the US Gross and the Worldwide Gross to get a better understanding of how popular these movies are in the world. After determining which of the Rotten Tomatoes Ratings movies have the biggest gross we will be able to see if the rating affects the gross for the US vs the World. 
# 
# 
# 

# ## Main portion of the project
# 
# 

# In[ ]:


import numpy as np
import pandas as pd
import vega_datasets
import altair as alt
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error,log_loss
from sklearn.preprocessing import StandardScaler


# In[ ]:


df = pd.read_csv("/work/movies.csv", na_values=" ")
df


# Cleaning the Data to get more accurate results:

# In[ ]:


df.dropna(inplace=True)
df


# In[ ]:


df.info()


# In[ ]:


df.dtypes


# In[ ]:


print(f"The number of rows in this dataset is {df.shape[0]}")


# Now we will see using groupby to anaylze if for every movie the gross has a rotten Tomato rating.

# In[ ]:


df.groupby(["US_Gross", "Worldwide_Gross"])["Rotten_Tomatoes_Rating"].count()


# Making a graph to getting a better understaning on how the Data looks.

# In[ ]:


c1= alt.Chart(df).mark_circle().encode(
    x = "US_Gross",
    y = "Worldwide_Gross",
    color="Rotten_Tomatoes_Rating"
).properties(
    title= "Gross Depending on Rotten Tomato Rating",
        width=700,
        height=100,
)
c1


# The graph above displays as worldwide gross increases then US Gross also increases. Addtionally the Rotten Tomatos rating displays that as the rating gets higher it becomes a darker blue. 

# Now I am going to use K-Nearest Neighbors Regressor, and  K Neighbors Classifier to see which data has the best results.

# Testing with K-Nearest Neighbors:

# First I am going to rescale the data:

# In[ ]:


scaler = StandardScaler()


# In[ ]:


scaler.fit(df[["US_Gross","Worldwide_Gross"]])


# In[ ]:


df[["US_Gross","Worldwide_Gross"]]= scaler.transform(df[["US_Gross","Worldwide_Gross"]])


# Next I am going to use train_test_split to train the data.

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(df[["US_Gross","Worldwide_Gross"]],df["Rotten_Tomatoes_Rating"],test_size=0.5)


# In[ ]:


print(f"The number of rows in this dataset is {y_train.shape[0]}")


# In[ ]:


print(f"The shape of this dataset using Train {X_train.shape}")


# In[ ]:


K_reg = KNeighborsRegressor(n_neighbors=10)


# In[ ]:


K_reg.fit(X_train,y_train)


# In[ ]:


mean_absolute_error(K_reg.predict(X_train), y_train)


# In[ ]:


mean_absolute_error(K_reg.predict(X_test), y_test)


# Finding the mean absolute error showed us that the data will not be overfitting when we use n_neighbors=10.

# Now we will be examining which k values gives us the least test error for K-Nearest Regressor.

# In[ ]:


def get_scores(k):
    K_reg = KNeighborsRegressor(n_neighbors=k)
    K_reg.fit(X_train, y_train)
    train_error = mean_absolute_error(K_reg.predict(X_train), y_train)
    test_error = mean_absolute_error(K_reg.predict(X_test), y_test)
    return (train_error, test_error)


# In[ ]:


K_reg_scores = pd.DataFrame({"k":range(1,88),"train_error":np.nan,"test_error":np.nan})


# In[ ]:


for i in K_reg_scores.index:
    K_reg_scores.loc[i,["train_error","test_error"]] = get_scores(K_reg_scores.loc[i,"k"])


# In[ ]:


K_reg_scores


# Anaylzing for the least test error:

# In[ ]:


(K_reg_scores["test_error"]).min()


# In[ ]:


(K_reg_scores["test_error"]<25).sum()


# This means that choosing k=10 was a good choice because the test error was around 24.8, meaning it is close to the least test error.

# Now we will plot the test error curve to get a better understanding of the flexibilty and variance of K.

# In[ ]:


K_reg_scores["kinv"] = 1/K_reg_scores.k


# In[ ]:


K_regtest = alt.Chart(K_reg_scores).mark_line(color="green").encode(
    x = "kinv",
    y = "test_error"
)


# In[ ]:


K_regtrain = alt.Chart(K_reg_scores).mark_line().encode(
    x = "kinv",
    y = "train_error"
    ).properties( 
        title= "Error"
)


# In[ ]:


K_regtest+K_regtrain


# Looking at the graph we can see the K values are at a good high flexibility and high variance in the begining and later on all the underfitting where the graph has been seperated meaning that there is lower flexibility.

# K-Nearest Neighbors Classifier

# In[ ]:


clf = KNeighborsClassifier(n_neighbors=6)


# In[ ]:


clf.fit(X_train, y_train)


# In[ ]:


mean_absolute_error(clf.predict(X_train), y_train)


# In[ ]:


mean_absolute_error(clf.predict(X_test), y_test)


# Finding the mean absolute error showed us that the data will not be overfitting when we use n_neighbors=6.

# Now we will be examining which k values gives us the least test error for K-Nearest Classifier.

# In[ ]:


def get_clf_scores(k):
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(X_train, y_train)
    train_error = mean_absolute_error(clf.predict(X_train), y_train)
    test_error = mean_absolute_error(clf.predict(X_test), y_test)
    return (train_error, test_error)


# In[ ]:


clf_scores = pd.DataFrame({"k":range(1,88),"train_error":np.nan,"test_error":np.nan})


# In[ ]:


for i in clf_scores.index:
    clf_scores.loc[i,["train_error","test_error"]] = get_clf_scores(clf_scores.loc[i,"k"])


# In[ ]:


clf_scores


# Anaylzing for the least test error:

# In[ ]:


clf_scores["test_error"].min()


# In[ ]:


(clf_scores["test_error"]< 30).sum()


# This means that choosing k=6 was a bad choice because the test error was around 38.8, meaning it is far from the least test error.

# Now we will plot the test error curve to get a better understanding of the flexibilty and variance of K.

# In[ ]:


clf_scores["kinv"] = 1/clf_scores.k


# In[ ]:


clftrain = alt.Chart(clf_scores).mark_line().encode(
    x = "kinv",
    y = "train_error"
)


# In[ ]:


clftest = alt.Chart(clf_scores).mark_line(color="green").encode(
    x = "kinv",
    y = "test_error"
  ).properties(
      title= "Error",
       
    
)


# In[ ]:


clftrain+clftest


# Looking at the graph we can see the overfitting in the begining and later on all the underfitting where the graph has been seperated meaning that there is lower flexibility.

# The whole reason of why we looked at the test error is analyze whether this Data had any correlation to the previous claim, so to find the best fitting K that will give us the least Test error.

# ## Summary
# Throughout our analyzing of the data using K-Nearest Regressers and Classifiers we can see how Regressor would be the better choice for our Data.Overall our Test error displays that we have around 25% error using K-Nearest Regressor, meaning for our US Gross and the Worldwide Gross the rotten tomato rating will have a around 25% error. We can conclude that rotten tomato ratings mostly has some impact on US Gross and Worldwide Gross. 

# ## References
# 
# Dataset was found in deepnote from Thursday (Week 8) under the Vega files.
# 
# [Pandas Groupby Link](https://realpython.com/pandas-groupby/)

# <a style='text-decoration:none;line-height:16px;display:flex;color:#5B5B62;padding:10px;justify-content:end;' href='https://deepnote.com?utm_source=created-in-deepnote-cell&projectId=28e99662-c742-4859-ae49-71229868940a' target="_blank">
# <img alt='Created in deepnote.com' style='display:inline;max-height:16px;margin:0px;margin-right:7.5px;' src='data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iODBweCIgaGVpZ2h0PSI4MHB4IiB2aWV3Qm94PSIwIDAgODAgODAiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8IS0tIEdlbmVyYXRvcjogU2tldGNoIDU0LjEgKDc2NDkwKSAtIGh0dHBzOi8vc2tldGNoYXBwLmNvbSAtLT4KICAgIDx0aXRsZT5Hcm91cCAzPC90aXRsZT4KICAgIDxkZXNjPkNyZWF0ZWQgd2l0aCBTa2V0Y2guPC9kZXNjPgogICAgPGcgaWQ9IkxhbmRpbmciIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJBcnRib2FyZCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTEyMzUuMDAwMDAwLCAtNzkuMDAwMDAwKSI+CiAgICAgICAgICAgIDxnIGlkPSJHcm91cC0zIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxMjM1LjAwMDAwMCwgNzkuMDAwMDAwKSI+CiAgICAgICAgICAgICAgICA8cG9seWdvbiBpZD0iUGF0aC0yMCIgZmlsbD0iIzAyNjVCNCIgcG9pbnRzPSIyLjM3NjIzNzYyIDgwIDM4LjA0NzY2NjcgODAgNTcuODIxNzgyMiA3My44MDU3NTkyIDU3LjgyMTc4MjIgMzIuNzU5MjczOSAzOS4xNDAyMjc4IDMxLjY4MzE2ODMiPjwvcG9seWdvbj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0zNS4wMDc3MTgsODAgQzQyLjkwNjIwMDcsNzYuNDU0OTM1OCA0Ny41NjQ5MTY3LDcxLjU0MjI2NzEgNDguOTgzODY2LDY1LjI2MTk5MzkgQzUxLjExMjI4OTksNTUuODQxNTg0MiA0MS42NzcxNzk1LDQ5LjIxMjIyODQgMjUuNjIzOTg0Niw0OS4yMTIyMjg0IEMyNS40ODQ5Mjg5LDQ5LjEyNjg0NDggMjkuODI2MTI5Niw0My4yODM4MjQ4IDM4LjY0NzU4NjksMzEuNjgzMTY4MyBMNzIuODcxMjg3MSwzMi41NTQ0MjUgTDY1LjI4MDk3Myw2Ny42NzYzNDIxIEw1MS4xMTIyODk5LDc3LjM3NjE0NCBMMzUuMDA3NzE4LDgwIFoiIGlkPSJQYXRoLTIyIiBmaWxsPSIjMDAyODY4Ij48L3BhdGg+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMCwzNy43MzA0NDA1IEwyNy4xMTQ1MzcsMC4yNTcxMTE0MzYgQzYyLjM3MTUxMjMsLTEuOTkwNzE3MDEgODAsMTAuNTAwMzkyNyA4MCwzNy43MzA0NDA1IEM4MCw2NC45NjA0ODgyIDY0Ljc3NjUwMzgsNzkuMDUwMzQxNCAzNC4zMjk1MTEzLDgwIEM0Ny4wNTUzNDg5LDc3LjU2NzA4MDggNTMuNDE4MjY3Nyw3MC4zMTM2MTAzIDUzLjQxODI2NzcsNTguMjM5NTg4NSBDNTMuNDE4MjY3Nyw0MC4xMjg1NTU3IDM2LjMwMzk1NDQsMzcuNzMwNDQwNSAyNS4yMjc0MTcsMzcuNzMwNDQwNSBDMTcuODQzMDU4NiwzNy43MzA0NDA1IDkuNDMzOTE5NjYsMzcuNzMwNDQwNSAwLDM3LjczMDQ0MDUgWiIgaWQ9IlBhdGgtMTkiIGZpbGw9IiMzNzkzRUYiPjwvcGF0aD4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+' > </img>
# Created in <span style='font-weight:600;margin-left:4px;'>Deepnote</span></a>
