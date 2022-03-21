#!/usr/bin/env python
# coding: utf-8

# # McDonald's Menu Analysis
# 
# Author: Jenny Tran
# 
# Course Project, UC Irvine, Math 10, W22

# ## Introduction

# McDonald's is one of the most popular fast food chains across the United States known for their affordable and unhealthy foods and beverages. We will use the McDonald's Nutrition Facts dataset to find which food item and category appears to be most healthy and unhealthy.
# 
# We will define healthy foods as something with the most proteins, least calories, and etc. We will define unhealthy foods as items with the most sugar, calories, least proteins, and etc. 

# ## Main portion of the project
# 

# In[3]:


import numpy as np
import pandas as pd
import seaborn as sns
import altair as alt
import plotly.express as px
import plotly.offline as py
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, log_loss


# ### Import the Data

# In[4]:


df = pd.read_csv("menu.csv")
df


# In[5]:


df['Category'].value_counts()


# In[6]:


print(f"Coffee & Tea has the most items with a total of {(df['Category'] == 'Coffee & Tea').sum()}")


# ### KNeighborsClassifier
# 
# Use KNeighborClassifier to predict the Category using Calories and Sodium.
# 
# 

# In[7]:


clf = KNeighborsClassifier(n_neighbors=4)


# In[8]:


X = df[["Calories", "Sodium"]]
y = df['Category']


# In[9]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6)


# In[10]:


clf.fit(X_test,y_test)


# In[11]:


df['Prediction'] = clf.predict(X)


# ### Altair
# 
# Use Altair to plot.

# In[12]:


c1 = alt.Chart(df).mark_circle().encode(
    x = "Sodium",
    y = "Calories",
    color = 'Category',
    tooltip = 'Item'
)


# In[13]:


c2 = alt.Chart(df).mark_circle().encode(
        x = "Sodium",
        y = "Calories",
        color = 'Prediction',
        tooltip = 'Item'
    )


# In[14]:


c1 | c2


# The 1st graph shows the actual scatterplot and the 2nd graph shows the predicted scatterplot. There is a lot more Breakfast foods that are shown in the predicted scatterplot. In both scatterplots, it looks like we have a positive correlation between Sodium and Calories. We can also see that the drinks and beverages have the least sodium.

# Next, we will find the number of neighbors (k) that will give us the best fit graph.

# In[15]:


for k in range(10,50):
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(X_train, y_train)
    loss = log_loss(y_test, clf.predict_proba(X_test))
    print(k)
    print(loss)


# The log_loss is the smallest when k=45. This means we will have the best fitted graph when we set n_neighbors equal to 45. (This number may change if we run the notebook multiple times. The ideal k number should be between 35-50.)

# ### Plotly
# Create a bar chart using Plotly. This bar chart will show us the food category and the amount of average proteins and sugars each category has. 

# In[16]:


avgprotein = pd.DataFrame(df.groupby('Category')['Protein'].mean())

fig = px.bar(df, x=avgprotein.index, y=avgprotein["Protein"])
fig.show()


# Chicken &amp; Fish has the most average protein. Lets find what food item in this category has the most protein.

# In[17]:


chifi = df[df['Category'] == 'Chicken & Fish']
chifi = chifi[['Item', 'Protein']]
chifi.sort_values(by=['Protein'], ascending=False)


# The Chicken McNuggets (40 Pieces) have the most protein followed by the Chicken McNuggets (20 Pieces).

# Next, we will check the average sugar of each category.

# In[18]:


avgsugar = pd.DataFrame(df.groupby('Category')['Sugars'].mean())

fig = px.bar(df, x=avgsugar.index, y=avgsugar["Sugars"])
fig.show()


# Smoothies &amp; Shakes has the most average sugar. Lets find what item has the most sugar in this category.

# In[19]:


smsh = df[df['Category'] == 'Smoothies & Shakes']
smsh = smsh[['Item', 'Sugars']]
smsh.sort_values(by=['Sugars'], ascending=False)


# The McFlurry with M&amp;M's Candies (Medium) has the most sugar.

# Next, we will find which category has the most total fats.

# In[20]:


avgfat = pd.DataFrame(df.groupby('Category')['Total Fat'].mean())

fig = px.bar(df, x=avgfat.index, y=avgfat["Total Fat"])
fig.show()


# The Beef &amp; Pork, Breakfast, and Chicken &amp; Fish categories contain the most average total fat.

# ### Correlation between Nutrients (Vitamins, Iron, Fat, etc.)

# In[21]:


dailyper = df[['Vitamin A (% Daily Value)','Vitamin C (% Daily Value)','Calcium (% Daily Value)',
      'Iron (% Daily Value)','Total Fat (% Daily Value)',
      'Cholesterol (% Daily Value)','Carbohydrates (% Daily Value)']]

dailyper.corr()


# From this we can see that Cholesterol and Iron is positively correlated with Total Fat (meaning the more Cholesterol and Iron, the more Total Fat we have). Ideally, we want foods with less Cholesterol and Total Fat, and more Iron. Instead, we may want to choose foods with high Carbohydrates as they are next to be positively correlated with Iron after Total Fat and Cholesterol.

# ### Salads
# 
# Salads are known to be very healthy. Lets find which salad contains the most dietary fibers.

# In[22]:


salads = df[df['Category']=='Salads']
salads


# In[23]:


fig = px.pie(salads, values='Dietary Fiber', names='Item',
             title='Dietary Fiber in Salads',)
fig.show()


# The Southwest Salad overall has the most dietary fibers with the highest being the Premium Southwest Salad with Crispy Chicken.

# ## Summary
# 

# There are multiple ways to interpret the results depending what diet someone chooses to use. For a high protein diet, it's best to choose food items within the Chicken &amp; Fish Category. For a low calorie diet, it's best to avoid foods with high sodium since they positively correlate with calories. This includes items within the Breakfast and Chicken &amp; Fish Categories. Overall, Smoothies &amp; Shakes should be avoided as they contain an overwhelming amount of sugar compared to the other categories. It's hard to find foods with high protein and iron with low total fat, cholesterol, and sodium. The Premium Southwest Salad is probably the best item to choose for an overall healthy diet. Salads, overall, didn't contain a lot of sugar and fat and had a good amount of protein, and this salad in particular had a lot of dietary fibers. 

# ## References

# Where the Dataset was found: https://www.kaggle.com/mcdonalds/nutrition-facts
# 
# Insightful References: 
# 
# https://www.kaggle.com/vaishnavipatil4848/insights-nutritional-facts-in-mcdonald-s-menu
# 
# https://plotly.com/python/pie-charts/

# <a style='text-decoration:none;line-height:16px;display:flex;color:#5B5B62;padding:10px;justify-content:end;' href='https://deepnote.com?utm_source=created-in-deepnote-cell&projectId=bd4244e8-6f61-43b0-80ec-abb92d8a6c91' target="_blank">
# <img alt='Created in deepnote.com' style='display:inline;max-height:16px;margin:0px;margin-right:7.5px;' src='data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iODBweCIgaGVpZ2h0PSI4MHB4IiB2aWV3Qm94PSIwIDAgODAgODAiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8IS0tIEdlbmVyYXRvcjogU2tldGNoIDU0LjEgKDc2NDkwKSAtIGh0dHBzOi8vc2tldGNoYXBwLmNvbSAtLT4KICAgIDx0aXRsZT5Hcm91cCAzPC90aXRsZT4KICAgIDxkZXNjPkNyZWF0ZWQgd2l0aCBTa2V0Y2guPC9kZXNjPgogICAgPGcgaWQ9IkxhbmRpbmciIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJBcnRib2FyZCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTEyMzUuMDAwMDAwLCAtNzkuMDAwMDAwKSI+CiAgICAgICAgICAgIDxnIGlkPSJHcm91cC0zIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxMjM1LjAwMDAwMCwgNzkuMDAwMDAwKSI+CiAgICAgICAgICAgICAgICA8cG9seWdvbiBpZD0iUGF0aC0yMCIgZmlsbD0iIzAyNjVCNCIgcG9pbnRzPSIyLjM3NjIzNzYyIDgwIDM4LjA0NzY2NjcgODAgNTcuODIxNzgyMiA3My44MDU3NTkyIDU3LjgyMTc4MjIgMzIuNzU5MjczOSAzOS4xNDAyMjc4IDMxLjY4MzE2ODMiPjwvcG9seWdvbj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0zNS4wMDc3MTgsODAgQzQyLjkwNjIwMDcsNzYuNDU0OTM1OCA0Ny41NjQ5MTY3LDcxLjU0MjI2NzEgNDguOTgzODY2LDY1LjI2MTk5MzkgQzUxLjExMjI4OTksNTUuODQxNTg0MiA0MS42NzcxNzk1LDQ5LjIxMjIyODQgMjUuNjIzOTg0Niw0OS4yMTIyMjg0IEMyNS40ODQ5Mjg5LDQ5LjEyNjg0NDggMjkuODI2MTI5Niw0My4yODM4MjQ4IDM4LjY0NzU4NjksMzEuNjgzMTY4MyBMNzIuODcxMjg3MSwzMi41NTQ0MjUgTDY1LjI4MDk3Myw2Ny42NzYzNDIxIEw1MS4xMTIyODk5LDc3LjM3NjE0NCBMMzUuMDA3NzE4LDgwIFoiIGlkPSJQYXRoLTIyIiBmaWxsPSIjMDAyODY4Ij48L3BhdGg+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMCwzNy43MzA0NDA1IEwyNy4xMTQ1MzcsMC4yNTcxMTE0MzYgQzYyLjM3MTUxMjMsLTEuOTkwNzE3MDEgODAsMTAuNTAwMzkyNyA4MCwzNy43MzA0NDA1IEM4MCw2NC45NjA0ODgyIDY0Ljc3NjUwMzgsNzkuMDUwMzQxNCAzNC4zMjk1MTEzLDgwIEM0Ny4wNTUzNDg5LDc3LjU2NzA4MDggNTMuNDE4MjY3Nyw3MC4zMTM2MTAzIDUzLjQxODI2NzcsNTguMjM5NTg4NSBDNTMuNDE4MjY3Nyw0MC4xMjg1NTU3IDM2LjMwMzk1NDQsMzcuNzMwNDQwNSAyNS4yMjc0MTcsMzcuNzMwNDQwNSBDMTcuODQzMDU4NiwzNy43MzA0NDA1IDkuNDMzOTE5NjYsMzcuNzMwNDQwNSAwLDM3LjczMDQ0MDUgWiIgaWQ9IlBhdGgtMTkiIGZpbGw9IiMzNzkzRUYiPjwvcGF0aD4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+' > </img>
# Created in <span style='font-weight:600;margin-left:4px;'>Deepnote</span></a>
