#!/usr/bin/env python
# coding: utf-8

# # Board Games
# 
# Author: Alicia Chuang
# 
# Student ID: 37703653
# 
# Course Project, UC Irvine, Math 10, W22

# ## Introduction
# 

# The goal of this project is to explore the correlation between different aspects of board games using regression and neural networks. Aspects explored include year of publication and number of user ratings and the effects of different factors on the board game rating. 

# ## Main portion of the project
# 

# **_Importing libraries_**

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import altair as alt

import torch
from torch import nn
from torchvision.transforms import ToTensor

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import log_loss
from sklearn.metrics import mean_absolute_error

from statistics import mean


# **_Loading file_**

# In[2]:


df = pd.read_csv("/work/bgg_dataset-20220302-203730.csv/bgg_dataset.csv", delimiter=';')
df.dropna(inplace=True)


# **_Display format of dataframe_**

# In[3]:


df.head()


# **_Section 1_**

# **_Goal: Check correlation between users owned, users rated, and year published_**
# 
# Data Used: Rows that contain years 2010-2022 and owned users between 5000 and 40000
# 
# Method: K nearest neighbors regressor and linear regression

# In[4]:


df2 = df[(df['Year Published'].isin(range(2010, 2023))) & (df['Owned Users'].isin(range(5000, 40001)))]


# **_Splitting and fitting data_**

# In[5]:


X_train, X_test, y_train, y_test = train_test_split(df2[['Owned Users']], df2['Users Rated'], test_size = 0.2)
reg = KNeighborsRegressor(n_neighbors=7)
pred = reg.fit(X_train, y_train)


# **_Plot K Nearest Neighbors Regression_**

# In[6]:


plt.scatter(X_train, y_train, s=5, color="black", label="original")
plt.plot(X_train, reg.predict(X_train), lw=0.5, color="green", label="predicted")
plt.legend()
plt.show()


# In[7]:


mean_absolute_error(reg.predict(X_test), y_test)


# In[8]:


mean_absolute_error(reg.predict(X_train), y_train)


# **_Error Analysis_**
# 
# The mean aboslute error for the training data and the test data are relatively close, so there is no sign of overfitting of the data. While the absolute errors may seem large, because of the large step sizes, the relative errors are acceptable.

# In[9]:


reg = LinearRegression()
reg.fit(X_train, y_train)
print(f"The slope of the linear regression is {reg.coef_[0]}")


# **_Plot Linear Regression_**

# In[10]:


alt.Chart(df2).mark_point(opacity=0.7).encode(
    x = alt.X("Owned Users",scale = alt.Scale(zero=False)),
    y = alt.Y("Users Rated", scale = alt.Scale(zero=False)),
    color = "Year Published"
).properties(
    title = "Owned Users vs Users Rated"
).interactive()


# **_Analysis_**
# 
# The number of owned users is positively correlated with the number of users rated by a factor of 0.69. As the number of people who own the game increases, the number of people who rate the game also increases, and as a general trend, if the game is published earlier, the percentage of the people who rate the game is higher than for the games published later.

# **_Section 2_**

# **_Goal: Predict the rating of a game based on the features of the game_**
# 
# Data Used: The original dataframe with rows containing na values dropped
# 
# Method: Neural networks

# **_Converting data types_**

# In[11]:


df['Rating Average'] = df['Rating Average'].apply(lambda x: np.char.replace(x, ',', '.'))
df['Complexity Average'] = df['Complexity Average'].apply(lambda x: np.char.replace(x, ',', '.'))
df['Rating Average'] = pd.to_numeric(df['Rating Average']).astype(int)
df['Complexity Average'] = pd.to_numeric(df['Complexity Average'])


# **_Reformating data for input_**

# In[12]:


X_train, X_test, y_train, y_test = train_test_split(df[['Year Published', 'Min Players', 'Max Players', 'Play Time', 'Min Age', 'Complexity Average']], df['Rating Average'], test_size = 0.2)
X_train = [[list(X_train['Year Published'])[i], list(X_train['Min Players'])[i], list(X_train['Max Players'])[i], list(X_train['Play Time'])[i], list(X_train['Min Age'])[i],list(X_train['Complexity Average'])[i]] for i in range(len(X_train))]
X_test = [[list(X_test['Year Published'])[i], list(X_test['Min Players'])[i], list(X_test['Max Players'])[i], list(X_test['Play Time'])[i], list(X_test['Min Age'])[i],list(X_test['Complexity Average'])[i]] for i in range(len(X_test))]


# **_Creating neural network_**

# In[13]:


class Boardgames(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(6,5),
            nn.Sigmoid(),
            nn.Linear(5,3),
            nn.ReLU(),
            nn.Linear(3,10),
            nn.LogSoftmax(dim=1)
        )

    def forward(self,x):
        x = x
        z = self.layers(x)
        return z


# In[14]:


obj = Boardgames()


# **_Training neural network_**

# In[15]:


loss_fn = nn.NLLLoss()
optimizer = torch.optim.SGD(obj.parameters(), lr=4)
epochs = 10

for i in range(epochs):
    y_true = torch.tensor(list(y_train)) - 1
    y_pred = obj(torch.tensor(X_train))
    loss = loss_fn(y_pred, y_true)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("Training loss:")
    print(loss)

    y_pred = obj(torch.tensor(X_test))
    y_true = torch.tensor(list(y_test)) - 1
    loss = loss_fn(y_pred, y_true)
    print("Test loss:")
    print(loss)
    print('\n')


# **_Analysis_**
# 
# The loss for the training data and the test data are both decreasing, so the neural network is working, and the training and test losses are very close, so there is no sign of overfitting.

# **_Section 3_**

# **_Goal: Visualize the data to put things into context_**
# 
# Data Used: Year published (1600-2022) and owned users

# In[16]:


df3 = df[df['Year Published'].isin(range(1600, 2023))][['Year Published', 'Owned Users']]
df3['Year Published'] = df3['Year Published'] // 100
df3.head()


# In[17]:


pub_vals = {}
size = 0
for i in set(df3['Year Published']):
    pub_vals[i] = sum(df3['Year Published'] == i)
    size += pub_vals[i]
pub_vals


# In[18]:


avg_vals = {}
for i in set(df3['Year Published']):
    avg_vals[i*100] = mean(df3[df3['Year Published']==i]['Owned Users'])
avg_vals


# **_Plot Bar Chart_**

# In[19]:


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
labels = []
sizes = []
for key, val in pub_vals.items():
    labels.append(key*100)
    sizes.append(val)
ax.bar(labels,sizes, 80)
plt.title("Number of Ranked Board Games from 1600-2022")
plt.show()


# **_Plot Pie Chart_**

# In[20]:


labels = []
sizes = []
for key, val in avg_vals.items():
    labels.append(key)
    sizes.append(val/size)

fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
ax1.axis('equal') 
plt.title("Owned Users of Board Games published 1600-2022")

plt.show()


# **_Analysis_**
# 
# Although there has only been 22 years in the 2000s so far, the number of boardgames published outnumber any of the other centuries by a significant amount. However, the average number of users for board games published in different years number roughly the same with 1700 being about 1/4 of the other centuries and 1900 about 1/2.
# 

# ## Summary
# 
# The project is split into 3 sections. The first analyzes the correlation between users owned, users rated, and the year published using regression. The second is a neural network that predicts the rating of a board game given six aspects of the game. The third gives a visualization of the dataset in terms of boardgames published and users owned in each century.

# ## References

# Import statements: https://christopherdavisuci.github.io/UCI-Math-10-W22/Week7/Week7-Monday.html
# 
# Dataset: https://www.kaggle.com/andrewmvd/board-games
# 
# Convert string to ints: https://stackoverflow.com/questions/21291259/convert-floats-to-ints-in-pandas
# 
# Plotting K nearest neighbors: https://www.datatechnotes.com/2019/04/regression-example-with-k-nearest.html
# 
# Customizing altair charts: https://altair-viz.github.io/user_guide/customization.html
# 
# Matplotlib bar chart: https://www.tutorialspoint.com/matplotlib/matplotlib_bar_plot.htm
# 
# Matplotlib pie chart: https://matplotlib.org/stable/gallery/pie_and_polar_charts/pie_features.html
# 
# Mean function: https://www.geeksforgeeks.org/python-statistics-mean-function/

# 

# <a style='text-decoration:none;line-height:16px;display:flex;color:#5B5B62;padding:10px;justify-content:end;' href='https://deepnote.com?utm_source=created-in-deepnote-cell&projectId=5a3c711c-de92-4ae4-b61d-d6a8182ff089' target="_blank">
# <img alt='Created in deepnote.com' style='display:inline;max-height:16px;margin:0px;margin-right:7.5px;' src='data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iODBweCIgaGVpZ2h0PSI4MHB4IiB2aWV3Qm94PSIwIDAgODAgODAiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8IS0tIEdlbmVyYXRvcjogU2tldGNoIDU0LjEgKDc2NDkwKSAtIGh0dHBzOi8vc2tldGNoYXBwLmNvbSAtLT4KICAgIDx0aXRsZT5Hcm91cCAzPC90aXRsZT4KICAgIDxkZXNjPkNyZWF0ZWQgd2l0aCBTa2V0Y2guPC9kZXNjPgogICAgPGcgaWQ9IkxhbmRpbmciIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJBcnRib2FyZCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTEyMzUuMDAwMDAwLCAtNzkuMDAwMDAwKSI+CiAgICAgICAgICAgIDxnIGlkPSJHcm91cC0zIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxMjM1LjAwMDAwMCwgNzkuMDAwMDAwKSI+CiAgICAgICAgICAgICAgICA8cG9seWdvbiBpZD0iUGF0aC0yMCIgZmlsbD0iIzAyNjVCNCIgcG9pbnRzPSIyLjM3NjIzNzYyIDgwIDM4LjA0NzY2NjcgODAgNTcuODIxNzgyMiA3My44MDU3NTkyIDU3LjgyMTc4MjIgMzIuNzU5MjczOSAzOS4xNDAyMjc4IDMxLjY4MzE2ODMiPjwvcG9seWdvbj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0zNS4wMDc3MTgsODAgQzQyLjkwNjIwMDcsNzYuNDU0OTM1OCA0Ny41NjQ5MTY3LDcxLjU0MjI2NzEgNDguOTgzODY2LDY1LjI2MTk5MzkgQzUxLjExMjI4OTksNTUuODQxNTg0MiA0MS42NzcxNzk1LDQ5LjIxMjIyODQgMjUuNjIzOTg0Niw0OS4yMTIyMjg0IEMyNS40ODQ5Mjg5LDQ5LjEyNjg0NDggMjkuODI2MTI5Niw0My4yODM4MjQ4IDM4LjY0NzU4NjksMzEuNjgzMTY4MyBMNzIuODcxMjg3MSwzMi41NTQ0MjUgTDY1LjI4MDk3Myw2Ny42NzYzNDIxIEw1MS4xMTIyODk5LDc3LjM3NjE0NCBMMzUuMDA3NzE4LDgwIFoiIGlkPSJQYXRoLTIyIiBmaWxsPSIjMDAyODY4Ij48L3BhdGg+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMCwzNy43MzA0NDA1IEwyNy4xMTQ1MzcsMC4yNTcxMTE0MzYgQzYyLjM3MTUxMjMsLTEuOTkwNzE3MDEgODAsMTAuNTAwMzkyNyA4MCwzNy43MzA0NDA1IEM4MCw2NC45NjA0ODgyIDY0Ljc3NjUwMzgsNzkuMDUwMzQxNCAzNC4zMjk1MTEzLDgwIEM0Ny4wNTUzNDg5LDc3LjU2NzA4MDggNTMuNDE4MjY3Nyw3MC4zMTM2MTAzIDUzLjQxODI2NzcsNTguMjM5NTg4NSBDNTMuNDE4MjY3Nyw0MC4xMjg1NTU3IDM2LjMwMzk1NDQsMzcuNzMwNDQwNSAyNS4yMjc0MTcsMzcuNzMwNDQwNSBDMTcuODQzMDU4NiwzNy43MzA0NDA1IDkuNDMzOTE5NjYsMzcuNzMwNDQwNSAwLDM3LjczMDQ0MDUgWiIgaWQ9IlBhdGgtMTkiIGZpbGw9IiMzNzkzRUYiPjwvcGF0aD4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+' > </img>
# Created in <span style='font-weight:600;margin-left:4px;'>Deepnote</span></a>
