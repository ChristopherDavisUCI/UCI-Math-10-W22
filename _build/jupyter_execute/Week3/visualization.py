#!/usr/bin/env python
# coding: utf-8

# # Visualization in Python
# 
# [Recording of lecture from 1/19/2022](https://uci.zoom.us/rec/play/hTqfyDlKjP85g3nw5_mQDak9OKyIFmY5kn6dOiW7nDzZsKce_heexreKHBh9gKJ-sEIbJzX4UQ9MalE.sOsJfZkgmX4ZunE0)
# 
# The most important visualization library in Math 10 is Altair.  Today I want to introduce Altair and two other similar libraries, Seaborn and Plotly Express.  All three of these are based on a concept called the *Grammar of Graphics*, which I believe was invented in this book, [The Grammar of Graphics](https://link.springer.com/book/10.1007/0-387-28695-0), which is free to download from on campus or using VPN.
# 
# The most famous visualization library in Python is Matplotlib.  We won't talk about Matplotlib today.  It is quite different from the libraries we will discuss today (Seaborn is built on top of Matplotlib).

# In[1]:


import numpy as np
import pandas as pd
import altair as alt
import seaborn as sns
import plotly.express as px


# In[2]:


np.arange(0,1.1,0.25)


# Here is the "by hand" way to make a pandas DataFrame.
# 
# Using `np.arange` is a little difficult in this context because we need to make sure its length is the same as the length of the other columns.

# In[3]:


df = pd.DataFrame({"a":[3,1,4,2],"b":[10,5,6,8],"c":["first","second","third","fourth"],
    "d":np.arange(0.2,1.1,0.25)})
df


# Put your mouse over one of the points to see the effect of the tooltip.

# In[4]:


alt.Chart(df).mark_circle().encode(
    x = "a",
    y = "b",
    color = "d",
    size = "d",
    tooltip = ["a","c"]
)


# In[5]:


alt.Chart(df).mark_bar().encode(
    x = "a",
    y = "b"
)


# In[6]:


alt.Chart(df).mark_bar(width=30).encode(
    x = "a",
    y = "b"
)


# To make a scatter plot in Altair, you use `mark_circle`.  In Seaborn, you use `scatterplot`.  The syntax is very similar.

# In[7]:


sns.scatterplot(
    data = df,
    x = "a",
    y = "b",
    hue = "d",
    size = "d",
)


# The same thing for Plotly Express.

# In[8]:


px.scatter(
    data_frame=df,
    x = "a",
    y = "b",
    color = "d",
    size = "d",
)


# ## Penguins dataset from Seaborn

# In[9]:


df = sns.load_dataset("penguins")


# In[10]:


df.head()


# In[11]:


df.shape


# In[12]:


df.dtypes


# In[13]:


alt.Chart(df).mark_circle().encode(
    x = "bill_length_mm",
    y = "bill_depth_mm",
    color = "species"
)


# In[14]:


px.scatter(
    data_frame=df,
    x = "bill_length_mm",
    y = "bill_depth_mm",
    color = "species"
)


# In[15]:


sns.scatterplot(
    data = df,
    x = "bill_length_mm",
    y = "bill_depth_mm",
    hue = "species"
)


# By default, the Altair axes will include 0.  If you want to remove them, the code gets a little longer.

# In[16]:


alt.Chart(df).mark_circle().encode(
    x = alt.X("bill_length_mm",scale = alt.Scale(zero=False)),
    y = alt.Y("bill_depth_mm", scale = alt.Scale(zero=False)),
    color = "species"
)


# Adding a tooltip that includes all the data.

# In[17]:


alt.Chart(df).mark_circle().encode(
    x = alt.X("bill_length_mm",scale = alt.Scale(zero=False)),
    y = alt.Y("bill_depth_mm", scale = alt.Scale(zero=False)),
    color = "species",
    size = "body_mass_g",
    opacity = "body_mass_g",
    tooltip = list(df.columns)
)


# Plotting just the data from rows 200 to 300.

# In[18]:


alt.Chart(df[200:300]).mark_circle().encode(
    x = alt.X("bill_length_mm",scale = alt.Scale(zero=False)),
    y = alt.Y("bill_depth_mm", scale = alt.Scale(zero=False)),
    color = "species",
    size = "body_mass_g",
    opacity = "body_mass_g",
    tooltip = list(df.columns)
)


# `df[200:300]` and `df.iloc[200:300]` mean the same thing; the one is just an abbreviation for the other.

# Can you find the point on the above plot that corresponds to row 200?

# In[19]:


df.iloc[200:300]


# In[20]:


df.columns


# In[21]:


type(df.columns)


# In[22]:


list(df.columns)


# The best way to convert `df.columns` from a pandas Index into a list is to use `list(df.columns)`.  Just for practice, we also convert it into a list using list comprehension.

# In[23]:


[c for c in df.columns]


# Instead of the penguins dataset, there are others we could have imported also.

# In[24]:


sns.get_dataset_names()


# In[25]:


df_tips = sns.load_dataset("tips")


# In[26]:


df_tips


# We can save this using the `to_csv` method.  If you don't want the row names included (the "index"), then set `index = false`.

# In[27]:


df_tips.to_csv("tips.csv", index=False)


# In Deepnote, if you click on the corresponding csv file in the files section, it will automatically sort the rows.  Here is how you do that same thing using pandas.

# In[28]:


df_tips.sort_values("total_bill",ascending=False)

