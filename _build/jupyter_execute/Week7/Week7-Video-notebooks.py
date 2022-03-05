#!/usr/bin/env python
# coding: utf-8

# # Week 7 Video notebooks

# In[1]:


import numpy as np
import pandas as pd
import altair as alt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss


# ## Using copy to avoid pandas warnings

# In[2]:


df = pd.read_csv("../data/spotify_dataset.csv", na_values=" ")


# In[3]:


df.shape


# In[4]:


df.dropna(inplace=True)


# In[5]:


df.shape


# In[6]:


df.Chord.value_counts()[:4]


# In[7]:


df.Chord.value_counts()[:4].index


# In[8]:


df.Chord.isin(df.Chord.value_counts()[:4].index).sum()


# In[9]:


df.Chord.isin(df.Chord.value_counts()[:4].index)


# In[10]:


df.head()


# In[11]:


df2 = df[df.Chord.isin(df.Chord.value_counts()[:4].index)]


# In[12]:


df2.head()


# In[13]:


df2["pred"] = np.nan


# In[14]:


df3 = df[df.Chord.isin(df.Chord.value_counts()[:4].index)].copy()


# In[15]:


df3["pred"] = np.nan


# ## KNeighborsClassifier

# In[16]:


df3


# In[17]:


df3.dtypes


# In[18]:


X = df3[["Artist Followers", "Danceability", "Energy", "Loudness", "Acousticness"]].copy()


# In[19]:


X


# In[20]:


y = df3["Chord"]


# In[21]:


clf = KNeighborsClassifier(n_neighbors=4)


# In[22]:


clf.fit(X,y)


# In[23]:


X["pred"] = clf.predict(X)


# In[24]:


X


# In[25]:


alt.Chart(X).mark_circle().encode(
    x = "Energy",
    y = "Danceability",
    color = "pred"
)


# In[26]:


c1 = alt.Chart(X).mark_circle().encode(
    x = "Energy",
    y = "Artist Followers",
    color = "pred"
)


# In[27]:


c0 = alt.Chart(df3).mark_circle().encode(
    x = "Energy",
    y = "Artist Followers",
    color = "Chord"
)


# In[28]:


c0 | c1


# In[29]:


X


# ## StandardScaler

# In[30]:


num_cols = ["Artist Followers", "Danceability", "Energy", "Loudness", "Acousticness"]


# In[31]:


scaler = StandardScaler()


# In[32]:


scaler.fit(df3[num_cols])


# In[33]:


X_scaled = scaler.transform(df3[num_cols])


# In[34]:


clf = KNeighborsClassifier(n_neighbors=4)


# In[35]:


clf.fit(X_scaled,df3["Chord"])


# In[36]:


df3["pred"] = clf.predict(X_scaled)


# In[37]:


c1 = alt.Chart(df3).mark_circle().encode(
    x = "Energy",
    y = "Artist Followers",
    color = "pred"
)


# In[38]:


df4 = df3.copy()


# In[39]:


df4[cols] = X_scaled


# In[40]:


c1 = alt.Chart(df4).mark_circle().encode(
    x = "Energy",
    y = "Artist Followers",
    color = "pred"
)


# In[41]:


c1


# In[42]:


df4.mean(axis=0)


# In[43]:


cols


# In[44]:


df4.std(axis=0)


# ## log_loss

# In[45]:


clf.score(X_scaled,df3["Chord"])


# In[46]:


clf.predict_proba(X_scaled).shape


# In[47]:


clf.predict_proba(X_scaled)


# In[48]:


clf.classes_


# In[49]:


log_loss(df3["Chord"],clf.predict_proba(X_scaled))


# In[ ]:





# <a style='text-decoration:none;line-height:16px;display:flex;color:#5B5B62;padding:10px;justify-content:end;' href='https://deepnote.com?utm_source=created-in-deepnote-cell&projectId=b0408aef-1919-421b-9c38-66800207e6a9' target="_blank">
# <img alt='Created in deepnote.com' style='display:inline;max-height:16px;margin:0px;margin-right:7.5px;' src='data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iODBweCIgaGVpZ2h0PSI4MHB4IiB2aWV3Qm94PSIwIDAgODAgODAiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8IS0tIEdlbmVyYXRvcjogU2tldGNoIDU0LjEgKDc2NDkwKSAtIGh0dHBzOi8vc2tldGNoYXBwLmNvbSAtLT4KICAgIDx0aXRsZT5Hcm91cCAzPC90aXRsZT4KICAgIDxkZXNjPkNyZWF0ZWQgd2l0aCBTa2V0Y2guPC9kZXNjPgogICAgPGcgaWQ9IkxhbmRpbmciIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJBcnRib2FyZCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTEyMzUuMDAwMDAwLCAtNzkuMDAwMDAwKSI+CiAgICAgICAgICAgIDxnIGlkPSJHcm91cC0zIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxMjM1LjAwMDAwMCwgNzkuMDAwMDAwKSI+CiAgICAgICAgICAgICAgICA8cG9seWdvbiBpZD0iUGF0aC0yMCIgZmlsbD0iIzAyNjVCNCIgcG9pbnRzPSIyLjM3NjIzNzYyIDgwIDM4LjA0NzY2NjcgODAgNTcuODIxNzgyMiA3My44MDU3NTkyIDU3LjgyMTc4MjIgMzIuNzU5MjczOSAzOS4xNDAyMjc4IDMxLjY4MzE2ODMiPjwvcG9seWdvbj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0zNS4wMDc3MTgsODAgQzQyLjkwNjIwMDcsNzYuNDU0OTM1OCA0Ny41NjQ5MTY3LDcxLjU0MjI2NzEgNDguOTgzODY2LDY1LjI2MTk5MzkgQzUxLjExMjI4OTksNTUuODQxNTg0MiA0MS42NzcxNzk1LDQ5LjIxMjIyODQgMjUuNjIzOTg0Niw0OS4yMTIyMjg0IEMyNS40ODQ5Mjg5LDQ5LjEyNjg0NDggMjkuODI2MTI5Niw0My4yODM4MjQ4IDM4LjY0NzU4NjksMzEuNjgzMTY4MyBMNzIuODcxMjg3MSwzMi41NTQ0MjUgTDY1LjI4MDk3Myw2Ny42NzYzNDIxIEw1MS4xMTIyODk5LDc3LjM3NjE0NCBMMzUuMDA3NzE4LDgwIFoiIGlkPSJQYXRoLTIyIiBmaWxsPSIjMDAyODY4Ij48L3BhdGg+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMCwzNy43MzA0NDA1IEwyNy4xMTQ1MzcsMC4yNTcxMTE0MzYgQzYyLjM3MTUxMjMsLTEuOTkwNzE3MDEgODAsMTAuNTAwMzkyNyA4MCwzNy43MzA0NDA1IEM4MCw2NC45NjA0ODgyIDY0Ljc3NjUwMzgsNzkuMDUwMzQxNCAzNC4zMjk1MTEzLDgwIEM0Ny4wNTUzNDg5LDc3LjU2NzA4MDggNTMuNDE4MjY3Nyw3MC4zMTM2MTAzIDUzLjQxODI2NzcsNTguMjM5NTg4NSBDNTMuNDE4MjY3Nyw0MC4xMjg1NTU3IDM2LjMwMzk1NDQsMzcuNzMwNDQwNSAyNS4yMjc0MTcsMzcuNzMwNDQwNSBDMTcuODQzMDU4NiwzNy43MzA0NDA1IDkuNDMzOTE5NjYsMzcuNzMwNDQwNSAwLDM3LjczMDQ0MDUgWiIgaWQ9IlBhdGgtMTkiIGZpbGw9IiMzNzkzRUYiPjwvcGF0aD4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+' > </img>
# Created in <span style='font-weight:600;margin-left:4px;'>Deepnote</span></a>
