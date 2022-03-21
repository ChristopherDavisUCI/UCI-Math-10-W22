#!/usr/bin/env python
# coding: utf-8

# # Project Title
# 
# Author: Siyu Sun
# 
# Course Project, UC Irvine, Math 10, W22

# ## Introduction
# 
# Introduce your project here.  About 3 sentences.

# I import a dataset about sales of PS4 games around the world. I want to investigate basic information about the dataset for example, what is the most welcoming publisher and what is the genre with the highest sales.
# And is it possible for us to know genre type after we know its sales around the world?

# ## Main portion of the project
# 
# (You can either have all one section or divide into multiple sections)

# In[ ]:


import numpy as np
import pandas as pd
import altair as alt


# This is the dataset that I am investigating

# In[ ]:


df = pd.read_csv('PS4_GamesSales.csv',encoding='unicode_escape')
df.dropna(inplace=True)
df.head()


# 
# 
# ## Some basic information about the dataset
# 
# 

# In[ ]:


df.shape


# In[ ]:


print(f"The number of rows in this dataset is {df.shape[0]}")


# In[ ]:


df.dtypes


# The chart shows global sales of each genre.(From the chart, we can tell Action and Shooter have the most global  sales)

# In[ ]:


alt.Chart(df).mark_bar().encode(
    x="Genre",
    y="Global",
    color=alt.Color("Genre", title="Genre type"),
).properties(
    title="Global sales of each genre",
    width=1000,
    height=200,
)


# How many games are more welcomed in North America than Europe? (345 games are more welcomed in North America while 480 games are more welcomed in Europe)

# In[ ]:


(df.loc[:,"North America"] > df.loc[:,"Europe"]).value_counts()


# Find the most frequent genre of the dataset. The most frequent genre is "Action"

# In[ ]:


df['Genre'].value_counts().idxmax()


# Global total sales of each genre. (The genre with the most global sales is "Action".)

# In[ ]:


A=df.groupby(['Genre']).sum().Global
A


# Global average sales of each genre. (The genre with highest global average sales is Shooter)

# In[ ]:


B=df.groupby(['Genre']).mean().Global
B


# ## Predict genre  from sales of the game from each region of the world

# In this case, I am using KNeighborsClassifier to predict genre since it is a classification problem. And the reason that I choose n_neighbors=3 is because I find out that the predcition accuracy will become higher when the value of n_neighbors is a small value.

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
clf = KNeighborsClassifier(n_neighbors=3)
X=df[['North America','Europe','Japan','Rest of World']].copy()
y=df['Genre']
clf.fit(X,y)


# On the "pred" column, it shows the result of the prediction of the genre using KNeighborsClassifier

# In[ ]:


X["pred"] = clf.predict(X)
X.head()


# Find the correctness rate of the prediction. However, the result shows that it is not a good estimation.

# In[ ]:


np.count_nonzero(X['pred'] == df['Genre'])/825


# I want to test in this case, is it over-fitting or under-fitting?

# In[ ]:


del X["pred"]
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)


# In[ ]:


np.count_nonzero(clf.predict(X_train) == y_train)/len(X_train)


# In[ ]:


np.count_nonzero(clf.predict(X_test) == y_test)/len(X_test)


# Since test score is 0.46 and train score is 0.41 and test score is higher than train socre, the result is under-fitting.

# ## A new dataframe df2 which contains only genre of Action and Shooter

# In[ ]:


df2 = df[df["Genre"].isin(["Action","Shooter"])]
df2.head()


# In[ ]:


c = alt.Chart(df2).mark_circle().encode(
    x="North America",
    y="Global",
    color=alt.Color("Genre", title="Genre type"),
).properties(
    title="Sales of Action and shooter from North America and Global",
    width=1000,
    height=200,
)
c


# Will the prediction becomes more accurate if I include only two genres in the dataset? The result shows that it is still a bad estimation and the accuracy is the same as the original dateset.

# In[ ]:


X2=df[['North America','Europe','Japan','Rest of World']].copy()
y2=df['Genre']
clf.fit(X2,y2)
clf.score(X2,y2)


# ## Summary
# 
# Either summarize what you did, or summarize the results.  About 3 sentences.

# I import data, and get some basic information about the data: draw a graph to show global sales of each genre and find the most welcome genre. I try to predict genre from knowing its sales from each region but the prediction only has an accuracy of 0.42. I wonder if the accuracy of the prediction will become better if I only include the most popular genre(Action and Shooter), however the prediction doesn't improve and it has the same accuracy as the original dateset.
# 

# ## References
# 
# Include references that you found helpful.  Also say where you found the dataset you used.

# The extra thing I use is pandas.DataFrame.groupby.
# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.groupby.html
# 
# I find "PS4_Games Sales" from Kaggle.

# <a style='text-decoration:none;line-height:16px;display:flex;color:#5B5B62;padding:10px;justify-content:end;' href='https://deepnote.com?utm_source=created-in-deepnote-cell&projectId=23ae12f7-97a0-48d1-bd43-3134ca235d42' target="_blank">
# <img alt='Created in deepnote.com' style='display:inline;max-height:16px;margin:0px;margin-right:7.5px;' src='data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iODBweCIgaGVpZ2h0PSI4MHB4IiB2aWV3Qm94PSIwIDAgODAgODAiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8IS0tIEdlbmVyYXRvcjogU2tldGNoIDU0LjEgKDc2NDkwKSAtIGh0dHBzOi8vc2tldGNoYXBwLmNvbSAtLT4KICAgIDx0aXRsZT5Hcm91cCAzPC90aXRsZT4KICAgIDxkZXNjPkNyZWF0ZWQgd2l0aCBTa2V0Y2guPC9kZXNjPgogICAgPGcgaWQ9IkxhbmRpbmciIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJBcnRib2FyZCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTEyMzUuMDAwMDAwLCAtNzkuMDAwMDAwKSI+CiAgICAgICAgICAgIDxnIGlkPSJHcm91cC0zIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxMjM1LjAwMDAwMCwgNzkuMDAwMDAwKSI+CiAgICAgICAgICAgICAgICA8cG9seWdvbiBpZD0iUGF0aC0yMCIgZmlsbD0iIzAyNjVCNCIgcG9pbnRzPSIyLjM3NjIzNzYyIDgwIDM4LjA0NzY2NjcgODAgNTcuODIxNzgyMiA3My44MDU3NTkyIDU3LjgyMTc4MjIgMzIuNzU5MjczOSAzOS4xNDAyMjc4IDMxLjY4MzE2ODMiPjwvcG9seWdvbj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0zNS4wMDc3MTgsODAgQzQyLjkwNjIwMDcsNzYuNDU0OTM1OCA0Ny41NjQ5MTY3LDcxLjU0MjI2NzEgNDguOTgzODY2LDY1LjI2MTk5MzkgQzUxLjExMjI4OTksNTUuODQxNTg0MiA0MS42NzcxNzk1LDQ5LjIxMjIyODQgMjUuNjIzOTg0Niw0OS4yMTIyMjg0IEMyNS40ODQ5Mjg5LDQ5LjEyNjg0NDggMjkuODI2MTI5Niw0My4yODM4MjQ4IDM4LjY0NzU4NjksMzEuNjgzMTY4MyBMNzIuODcxMjg3MSwzMi41NTQ0MjUgTDY1LjI4MDk3Myw2Ny42NzYzNDIxIEw1MS4xMTIyODk5LDc3LjM3NjE0NCBMMzUuMDA3NzE4LDgwIFoiIGlkPSJQYXRoLTIyIiBmaWxsPSIjMDAyODY4Ij48L3BhdGg+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMCwzNy43MzA0NDA1IEwyNy4xMTQ1MzcsMC4yNTcxMTE0MzYgQzYyLjM3MTUxMjMsLTEuOTkwNzE3MDEgODAsMTAuNTAwMzkyNyA4MCwzNy43MzA0NDA1IEM4MCw2NC45NjA0ODgyIDY0Ljc3NjUwMzgsNzkuMDUwMzQxNCAzNC4zMjk1MTEzLDgwIEM0Ny4wNTUzNDg5LDc3LjU2NzA4MDggNTMuNDE4MjY3Nyw3MC4zMTM2MTAzIDUzLjQxODI2NzcsNTguMjM5NTg4NSBDNTMuNDE4MjY3Nyw0MC4xMjg1NTU3IDM2LjMwMzk1NDQsMzcuNzMwNDQwNSAyNS4yMjc0MTcsMzcuNzMwNDQwNSBDMTcuODQzMDU4NiwzNy43MzA0NDA1IDkuNDMzOTE5NjYsMzcuNzMwNDQwNSAwLDM3LjczMDQ0MDUgWiIgaWQ9IlBhdGgtMTkiIGZpbGw9IiMzNzkzRUYiPjwvcGF0aD4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+' > </img>
# Created in <span style='font-weight:600;margin-left:4px;'>Deepnote</span></a>
