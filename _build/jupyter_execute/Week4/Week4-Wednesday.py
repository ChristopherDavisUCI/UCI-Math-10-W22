#!/usr/bin/env python
# coding: utf-8

# # More practice with the Spotify dataset
# 
# [Recording of lecture from 1/26/2022](https://uci.zoom.us/rec/share/o1cAbrjVq2FsYYENMQ9IitH5OweHK3aL5I5doNyNl_fet0crIRrF92-Sfbua2HIQ.rR3BmcjBPf2cZ4k_?startTime=1643212665000)
# 
# The best way to import this dataset is to use
# ```
# pd.read_csv("spotify_dataset.csv", na_values=" ")
# ```
# That is what we did last time.  But it's also good practice to try making the conversions ourselves.  This will give us a chance to try using two important pandas DataFrame methods:
# * `apply`
# * `applymap`
# 
# These two methods fit into the same family as the pandas Series method
# * `map`

# In[1]:


import numpy as np
import pandas as pd
import altair as alt


# In[2]:


# Leaving out the useful na_values keyword argument.
# We will have to do some cleaning of this dataset by hand.
df = pd.read_csv("../data/spotify_dataset.csv")


# In[3]:


df.head()


# In[4]:


df.dtypes


# In[5]:


pd.to_numeric(df["Energy"])


# In[6]:


df.replace(" ",np.nan)


# In[7]:


df.head()


# In[8]:


df.replace(1,"Hello")


# In[9]:


df.replace("100","Hello")


# In[10]:


df = df.replace(" ",np.nan)


# We could just as well have used
# `df.replace(" ",np.nan, inplace=True)`.  Not the same as `df = df.replace(" ",np.nan, inplace=True)`.

# In[11]:


pd.to_numeric(df.Energy)


# In[12]:


# if s is a blank space, replace it with Not a Number
# otherwise leave s the same
def rep_blank(s):
    if s == " ":
        return np.nan
    else:
        return s


# In[13]:


rep_blank(7)


# In[14]:


rep_blank(" ")


# In[15]:


help(df.applymap)


# In[16]:


# Apply this function to every entry in the DataFrame
df = df.applymap(rep_blank)


# In[17]:


pd.to_numeric(df["Energy"])


# In[18]:


# Our rep_blank function is not very Pythonic
# there's a more concise way to make the same thing


# In[19]:


df = pd.read_csv("../data/spotify_dataset.csv")


# In[20]:


pd.to_numeric(df["Energy"])


# In[21]:


# if s is a blank space, replace it with Not a Number
# otherwise leave s the same
def rep_blank(s):
    if s == " ":
        return np.nan
    else:
        return s


# In[22]:


df.applymap(lambda s: np.nan if s == " ")


# `map` is a method for pandas Series
# 
# `applymap` is a method for pandas DataFrames

# In[23]:


# more Pythonic
df.applymap(lambda s: np.nan if s == " " else s)


# In[24]:


df.applymap(lambda s: s if s != " " else np.nan)


# In[25]:


# more Pythonic
df = df.applymap(lambda s: np.nan if s == " " else s)


# In[26]:


pd.to_numeric(df.Energy)


# In[27]:


df.dtypes


# In[28]:


# How can we get the columns from Popularity to Valence (inclusive)?
df_sub = df.loc[:,"Popularity":"Valence"]


# In[29]:


# applymap: its input is a single entry
# apply: its input is an entire row or an entire column
df_sub = df_sub.apply(pd.to_numeric,axis=0)


# In[30]:


df_sub.dtypes


# In[31]:


df.loc[:,"Popularity":"Valence"] = df_sub


# In[32]:


df.dtypes


# In[33]:


pd.to_numeric(df.Streams)


# In[34]:


pd.to_numeric(df.Streams.map(lambda s: s.replace(",","")))


# In[35]:


df.head()


# In[36]:


# Try to swap 3rd row with 1st row
temp = df.iloc[1]


# In[37]:


type(temp)


# In[38]:


df.iloc[1] = df.iloc[3]


# In[39]:


df.head()


# In[40]:


df.iloc[3] = temp


# In[41]:


df.head()


# In[42]:


my_list = [1,10,3,5]


# In[43]:


temp = my_list[2]
my_list[2] = my_list[1]
my_list[1] = temp


# In[44]:


my_list


# In[ ]:





# <a style='text-decoration:none;line-height:16px;display:flex;color:#5B5B62;padding:10px;justify-content:end;' href='https://deepnote.com?utm_source=created-in-deepnote-cell&projectId=d2dc5a23-8dc3-440f-9052-badb90ea4d5f' target="_blank">
# <img alt='Created in deepnote.com' style='display:inline;max-height:16px;margin:0px;margin-right:7.5px;' src='data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iODBweCIgaGVpZ2h0PSI4MHB4IiB2aWV3Qm94PSIwIDAgODAgODAiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8IS0tIEdlbmVyYXRvcjogU2tldGNoIDU0LjEgKDc2NDkwKSAtIGh0dHBzOi8vc2tldGNoYXBwLmNvbSAtLT4KICAgIDx0aXRsZT5Hcm91cCAzPC90aXRsZT4KICAgIDxkZXNjPkNyZWF0ZWQgd2l0aCBTa2V0Y2guPC9kZXNjPgogICAgPGcgaWQ9IkxhbmRpbmciIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJBcnRib2FyZCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTEyMzUuMDAwMDAwLCAtNzkuMDAwMDAwKSI+CiAgICAgICAgICAgIDxnIGlkPSJHcm91cC0zIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxMjM1LjAwMDAwMCwgNzkuMDAwMDAwKSI+CiAgICAgICAgICAgICAgICA8cG9seWdvbiBpZD0iUGF0aC0yMCIgZmlsbD0iIzAyNjVCNCIgcG9pbnRzPSIyLjM3NjIzNzYyIDgwIDM4LjA0NzY2NjcgODAgNTcuODIxNzgyMiA3My44MDU3NTkyIDU3LjgyMTc4MjIgMzIuNzU5MjczOSAzOS4xNDAyMjc4IDMxLjY4MzE2ODMiPjwvcG9seWdvbj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0zNS4wMDc3MTgsODAgQzQyLjkwNjIwMDcsNzYuNDU0OTM1OCA0Ny41NjQ5MTY3LDcxLjU0MjI2NzEgNDguOTgzODY2LDY1LjI2MTk5MzkgQzUxLjExMjI4OTksNTUuODQxNTg0MiA0MS42NzcxNzk1LDQ5LjIxMjIyODQgMjUuNjIzOTg0Niw0OS4yMTIyMjg0IEMyNS40ODQ5Mjg5LDQ5LjEyNjg0NDggMjkuODI2MTI5Niw0My4yODM4MjQ4IDM4LjY0NzU4NjksMzEuNjgzMTY4MyBMNzIuODcxMjg3MSwzMi41NTQ0MjUgTDY1LjI4MDk3Myw2Ny42NzYzNDIxIEw1MS4xMTIyODk5LDc3LjM3NjE0NCBMMzUuMDA3NzE4LDgwIFoiIGlkPSJQYXRoLTIyIiBmaWxsPSIjMDAyODY4Ij48L3BhdGg+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMCwzNy43MzA0NDA1IEwyNy4xMTQ1MzcsMC4yNTcxMTE0MzYgQzYyLjM3MTUxMjMsLTEuOTkwNzE3MDEgODAsMTAuNTAwMzkyNyA4MCwzNy43MzA0NDA1IEM4MCw2NC45NjA0ODgyIDY0Ljc3NjUwMzgsNzkuMDUwMzQxNCAzNC4zMjk1MTEzLDgwIEM0Ny4wNTUzNDg5LDc3LjU2NzA4MDggNTMuNDE4MjY3Nyw3MC4zMTM2MTAzIDUzLjQxODI2NzcsNTguMjM5NTg4NSBDNTMuNDE4MjY3Nyw0MC4xMjg1NTU3IDM2LjMwMzk1NDQsMzcuNzMwNDQwNSAyNS4yMjc0MTcsMzcuNzMwNDQwNSBDMTcuODQzMDU4NiwzNy43MzA0NDA1IDkuNDMzOTE5NjYsMzcuNzMwNDQwNSAwLDM3LjczMDQ0MDUgWiIgaWQ9IlBhdGgtMTkiIGZpbGw9IiMzNzkzRUYiPjwvcGF0aD4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+' > </img>
# Created in <span style='font-weight:600;margin-left:4px;'>Deepnote</span></a>
