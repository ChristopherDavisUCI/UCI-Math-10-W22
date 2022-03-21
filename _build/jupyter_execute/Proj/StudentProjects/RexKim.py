#!/usr/bin/env python
# coding: utf-8

# # Final Project NBA Players
# 
# Author: Rex Kim 
# 
# Course Project Preparation, UC Irvine, Math 10, W22

# ## Introduction
# 
# This project dives into advanced statistics of current NBA players in the 2021-2022 NBA season. Different statistical categories include win shares (WS), value over replacement player (VORP), and box plus/minus. I'm going to discover how the NBA ranks its top 10 current NBA players based on the categories just listed as well as other main stats such as points, rebounds, and assists. 

# ## Main portion of the project
# 
# (You can either have all one section or divide into multiple sections)

# In[ ]:


import seaborn as sns
import numpy as np
import pandas as pd
import altair as alt
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import log_loss


df = pd.read_csv('nba_2020_adv.csv')
df


# I will locate all players in the NBA and the position they play. 

# In[ ]:


df.loc[:,["Player","Pos"]]


# Because how much popularized the point guard position is, I decided to check all point guards and their statistics. 

# In[ ]:


df.loc[df['Pos'] == 'PG']


# In[ ]:


df.loc[df['Pos'] == 'PG']


# The following assigns a standard scaler to scaler and also initializes our training and testing data. 

# In[ ]:



scaler = StandardScaler()
df[["BPM","USG%"]] = scaler.fit_transform(df[["BPM","USG%"]])

X_train, X_test, y_train, y_test = train_test_split(df[["BPM","USG%"]], df["Age"], test_size=0.2)


# In[ ]:


X = df[['USG%', 'BPM']]
X


# Below, I'm ensuring that the dataframe X which holds usage rate and box plus/minus are resized for distribution.

# In[ ]:


scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)


# In[ ]:


y = df['Age']
y


# In[ ]:


y_test


# The K-neighbors classifier is assigned to clf and this is where we create an Altair scatter plot with box plus/minus being on the x-axis and usage rate being on the y axis. I am also configuring the orientiation of the legend. 

# The Altair chart below shows when nearest neighbors has a k value of 1 meaning that the bias is 0. If k = 1, then the object is simply assigned to the class of that single nearest neighbor. By hovering over the rightmost players, we can see that the more notable players like James Harden and Lebron James lead the lead in box plus/minus but there are also players who played minimal minutes but had productive games.

# In[ ]:


clf = KNeighborsClassifier(n_neighbors=1)
clf.fit(X_train, y_train)
df["pred"] = clf.predict(df[["BPM","USG%"]])
c = alt.Chart(df).mark_circle().encode(
    x="BPM",
    y="USG%",
    color=alt.Color("pred", title="Age"),
    tooltip = ('Player:N','Age:Q','BPM:Q','USG%:Q','MP:Q')
).properties(
    title="Box Plus/Minus vs. Usage",
    width=1000,
    height=400,
)

c.configure_legend(
    strokeColor='gray',
    fillColor='#EEEEEE',
    padding=10,
    cornerRadius=10,
    orient='top-right'
)
alt.data_transformers.disable_max_rows()
print(f"The number of rows in this dataset is {df.shape[0]}")
c


# Below I am checking to see whether there is overfitting or underfitting. 

# In[ ]:


clf.score(X_train,y_train)


# In[ ]:


clf.score(X_test,y_test)


# The data looks to be underfitting since the training set is lower than the testing set.

# Here, I am creating another ten Altair scatter plots but when k neighbors increments from 1 to 9. Larger values of k will have smoother decision boundaries which mean lower variance but increased bias. In turn, this can significantly influence our result. 

# In[ ]:



def make_chart(k):
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(X_train, y_train)
    df[f"pred{k}"] = clf.predict(df[["BPM","USG%"]])
    test_score = clf.score(X_test,y_test) 
    train_score = clf.score(X_train,y_train)
    c = alt.Chart(df).mark_circle().encode(
        x="BPM",
        y="USG%",
        color=alt.Color(f"pred{k}", title="Predicted Age"),
        tooltip = ('Player:N','Age:Q','BPM:Q','USG%:Q','MP:Q')
    ).properties(
        title=f"n_neighbors = {k}",
        width=1000,
        height=200,
    )
    c.configure_legend(
    strokeColor='gray',
    fillColor='#EEEEEE',
    padding=10,
    cornerRadius=10,
    orient='top-right'
)
    
    return c 


# In[ ]:


alt.vconcat(*[make_chart(k) for k in range(1,10)])


# Below I will now attempt to create a boxplot for all players' age, box plus/minus, and usage rate. I will include how to read a boxplot
# 
# Minimum: Smallest number in the dataset.
# First quartile: Middle number between the minimum and the median.
# Second quartile (Median): Middle number of the (sorted) dataset.
# Third quartile: Middle number between median and maximum.
# Maximum: Highest number in the dataset.

# The boxplot below shows that the average age of NBA players is roughly 25 years. With outliers like Jamal Crawford being age 43.

# In[ ]:


ageBoxPlot = df.boxplot(column=['Age'])  


# The boxplot below measures all NBA players box plus/minus. The lower the box plus/minus the lesser the player actually performs in game and contributes to winning. This isn't exactly the best representation of NBA players as Jamal Crawford only played 6 total minutes this recorded season. 

# In[ ]:


bpmBoxPlot = df.boxplot(column=['BPM'])  


# As k gets larger, there is lower variance and greater bias. This is how players like Tyler Zeller have such high usage rates despite playing such little minutes. 

# In[ ]:


usgBoxPlot = df.boxplot(column=["USG%"]) 


# In[ ]:


boxplot = df.boxplot(column=["Age","BPM","USG%"]) 


# I found that specifying age, BPM, and USG% by position and top 10 players in the NBA was somewhat difficult. The plots and altair charts above represent the idea that there are only a select few players whose advanced statistics still remain high despite aging. Players like Stephen Curry, Kevin Durant, and Lebron James are namely a few. 

# ## Summary
# In the NBA, win shares are calculated using main stats. In this project, I tried to determine if a player's age of could be predicted by a player's usage rate(USG%) and box plus/minus. The models above show that with increased age, majority of players' USG% and BPM decrease significantly. 

# ## References
# 
# Include references that you found helpful.  Also say where you found the dataset you used.

# Information on pie charts and box plots
# https://deepnote.com/@a11s/Data-Visualisation-with-Python-EgfTyjpfS129FYXEnB2U7Q 
# 
# 
# URL to dataset
# https://www.kaggle.com/nicklauskim/nba-per-game-stats-201920 

# 

# <a style='text-decoration:none;line-height:16px;display:flex;color:#5B5B62;padding:10px;justify-content:end;' href='https://deepnote.com?utm_source=created-in-deepnote-cell&projectId=abeeb5a2-9a64-4252-a853-c7a1924df0f6' target="_blank">
# <img alt='Created in deepnote.com' style='display:inline;max-height:16px;margin:0px;margin-right:7.5px;' src='data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iODBweCIgaGVpZ2h0PSI4MHB4IiB2aWV3Qm94PSIwIDAgODAgODAiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8IS0tIEdlbmVyYXRvcjogU2tldGNoIDU0LjEgKDc2NDkwKSAtIGh0dHBzOi8vc2tldGNoYXBwLmNvbSAtLT4KICAgIDx0aXRsZT5Hcm91cCAzPC90aXRsZT4KICAgIDxkZXNjPkNyZWF0ZWQgd2l0aCBTa2V0Y2guPC9kZXNjPgogICAgPGcgaWQ9IkxhbmRpbmciIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJBcnRib2FyZCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTEyMzUuMDAwMDAwLCAtNzkuMDAwMDAwKSI+CiAgICAgICAgICAgIDxnIGlkPSJHcm91cC0zIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxMjM1LjAwMDAwMCwgNzkuMDAwMDAwKSI+CiAgICAgICAgICAgICAgICA8cG9seWdvbiBpZD0iUGF0aC0yMCIgZmlsbD0iIzAyNjVCNCIgcG9pbnRzPSIyLjM3NjIzNzYyIDgwIDM4LjA0NzY2NjcgODAgNTcuODIxNzgyMiA3My44MDU3NTkyIDU3LjgyMTc4MjIgMzIuNzU5MjczOSAzOS4xNDAyMjc4IDMxLjY4MzE2ODMiPjwvcG9seWdvbj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0zNS4wMDc3MTgsODAgQzQyLjkwNjIwMDcsNzYuNDU0OTM1OCA0Ny41NjQ5MTY3LDcxLjU0MjI2NzEgNDguOTgzODY2LDY1LjI2MTk5MzkgQzUxLjExMjI4OTksNTUuODQxNTg0MiA0MS42NzcxNzk1LDQ5LjIxMjIyODQgMjUuNjIzOTg0Niw0OS4yMTIyMjg0IEMyNS40ODQ5Mjg5LDQ5LjEyNjg0NDggMjkuODI2MTI5Niw0My4yODM4MjQ4IDM4LjY0NzU4NjksMzEuNjgzMTY4MyBMNzIuODcxMjg3MSwzMi41NTQ0MjUgTDY1LjI4MDk3Myw2Ny42NzYzNDIxIEw1MS4xMTIyODk5LDc3LjM3NjE0NCBMMzUuMDA3NzE4LDgwIFoiIGlkPSJQYXRoLTIyIiBmaWxsPSIjMDAyODY4Ij48L3BhdGg+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMCwzNy43MzA0NDA1IEwyNy4xMTQ1MzcsMC4yNTcxMTE0MzYgQzYyLjM3MTUxMjMsLTEuOTkwNzE3MDEgODAsMTAuNTAwMzkyNyA4MCwzNy43MzA0NDA1IEM4MCw2NC45NjA0ODgyIDY0Ljc3NjUwMzgsNzkuMDUwMzQxNCAzNC4zMjk1MTEzLDgwIEM0Ny4wNTUzNDg5LDc3LjU2NzA4MDggNTMuNDE4MjY3Nyw3MC4zMTM2MTAzIDUzLjQxODI2NzcsNTguMjM5NTg4NSBDNTMuNDE4MjY3Nyw0MC4xMjg1NTU3IDM2LjMwMzk1NDQsMzcuNzMwNDQwNSAyNS4yMjc0MTcsMzcuNzMwNDQwNSBDMTcuODQzMDU4NiwzNy43MzA0NDA1IDkuNDMzOTE5NjYsMzcuNzMwNDQwNSAwLDM3LjczMDQ0MDUgWiIgaWQ9IlBhdGgtMTkiIGZpbGw9IiMzNzkzRUYiPjwvcGF0aD4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+' > </img>
# Created in <span style='font-weight:600;margin-left:4px;'>Deepnote</span></a>
