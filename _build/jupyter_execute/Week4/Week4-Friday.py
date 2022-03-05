#!/usr/bin/env python
# coding: utf-8

# # Review
# 
# Today's class was review.
# 
# [Recording of lecture from 1/28/2022](https://uci.zoom.us/rec/share/OxfhTaj6tVZ4ysRJC_U2Nj3vGo0Hiqn-vm8WgGK-ANcfV5hahZLYPRCQFj60xI2a.MTVnvxxTkIXzkvLS?startTime=1643385481000)

# In[1]:


import numpy as np
import pandas as pd
import altair as alt


# ## Dictionaries, and their relationship to pandas Series and pandas DataFrames.

# In[2]:


d = {"a":5,"b":10,"chris":30}


# In[3]:


d["b"]


# In[4]:


e = {"a":[5,6,7],"b":10,"chris":[30,2.4,-20]}


# Making a DataFrame from a dictionary.

# In[5]:


df = pd.DataFrame(e)


# In[6]:


df


# In[7]:


df.a


# In[8]:


df["a"]


# Making a Series from a dictionary.

# In[9]:


pd.Series(d)


# In[10]:


df


# In[11]:


df.columns


# ## List comprehension

# Practice exercise:
# * Make a list of `True`s (bool type) using list comprehension, where the length of the list is the number of rows in `df`.

# In[12]:


# These are strings, not bools
['true' for x in range(3)]


# In[13]:


["true" for x in range(len(df))]


# In[14]:


[true for x in range(len(df))]


# Here is the correct answer.

# In[15]:


[True for x in range(len(df))]


# ## Putting a new column in a DataFrame
# 
# One way to create a new column.

# In[16]:


df["new column"] = [True for x in range(len(df))]


# In[17]:


df


# If the new column is filled with a single value, then you can make the new column faster:

# In[18]:


df["new column 2"] = False


# In[19]:


df


# ## Indexing
# 
# Indexing using `iloc`.

# In[20]:


df.iloc[1,0] = 20
df


# Indexing using `loc`.

# In[21]:


df.loc[1,'a'] = 20
df


# ## Second largest value in a column.
# 
# Find the second largest value in the "a" column of df.

# In[22]:


df["a"]


# In[23]:


df["a"].sort_values()


# In[24]:


df["a"].sort_values(ascending=False)


# I kept accidentally trying to use `[1]` instead of `iloc[1]`.  Here is the correct way to find the element at index 1 in a pandas Series.

# In[25]:


df["a"].sort_values(ascending=False).iloc[1]


# Without the `ascending` keyword argument, we have to take the second-to-last entry.

# In[26]:


df["a"].sort_values().iloc[-2]


# ## Practice with axis
# 
# We haven't used `median` before, I don't think, but the following should make sense.  The most important part is recognizing what `axis=0` means.

# In[27]:


df


# In[28]:


df.median(axis=0)


# ## Flattening a NumPy array

# In[29]:


A = np.array([[2,5,1],[3,1,10]])


# In[30]:


A.reshape((-1))


# ## Slicing

# In[31]:


df


# Access every other element in the row with label 1 using slicing

# In[32]:


df.loc[1,::2]


# Change every other element in the row with label 1 using slicing

# In[33]:


df.loc[1,::2] = [i**2 for i in range(3)]


# Change every other element in the row with label 1 using slicing

# In[34]:


df.loc[1,::2] = [0,1,4]


# In[35]:


df


# ## Square every element in a DataFrame

# In[36]:


df2 = df.iloc[:,:3]
df2


# In[37]:


df2.loc[1] = df2.loc[1]**2


# In[38]:


df2


# In[39]:


df2**2


# Try doing that same thing (squaring every entry in df2) using `map`, `apply`, or `applymap`.

# In[40]:


df2.applymap(lambda x: x**2)


# **Warning**: notice that `applymap` doesn't change the original DataFrame.

# In[41]:


df2


# ## Example using apply

# In[42]:


df2.apply(lambda c: c.sum(), axis = 0)


# In[43]:


df2.apply(lambda r: r.sum(), axis = 1)


# What if we tried to use `applymap` instead?

# In[44]:


df2.applymap(lambda a: a.sum())


# Sample exercise: What causes the above error?
# 
# Sample answer: `a` will be a number in the dataframe, and `number.sum()` does not make sense.  Should use `apply` and `axis` instead.

# A smaller piece of code that raises the same error:

# In[45]:


a = 5.1
a.sum()


# Of the three methods, `map`, `applymap`, and `apply`, definitely `apply` is the trickiest to understand.  The most natural example with `apply` we have seen was using `pd.to_numeric` on every column.  Notice how the input to `pd.to_numeric` should be an entire Series, not an individual entry.

# ## Working with datetime entries

# In[46]:


df = pd.read_csv("../data/spotify_dataset.csv", na_values=" ")


# In[47]:


df.columns


# This way of getting a column from a DataFrame is called *attribute* access:

# In[48]:


df.Genre


# It doesn't always work.  For example, in the following, we can't use attribute access because *Release Date* has a space in it.

# In[49]:


df.Release Date


# In[50]:


df["Release Date"]


# In[51]:


pd.to_datetime(df["Release Date"]).dt.day


# In[52]:


year_series = pd.to_datetime(df["Release Date"]).dt.year


# Sample exercise: How many `2019`s are there in `year_series`?

# In[53]:


(year_series == 2019).sum()


# Here we're trying a different method, but it doesn't work at first because of null values.  I realized after class that we could have used a keyword argument to `map` called `na_action`, but during class we removed the null values by hand.

# In[54]:


df["Release Date"].map(lambda s: s[:4] == 2019)


# In[55]:


np.nan[:4]


# During class I made another mistake before getting to the next cell, but I deleted it from this notebook because it's more confusing than helpful.
# 
# Here is an alternate approach.

# In[56]:


clean = df["Release Date"][~df["Release Date"].isna()]


# In[57]:


type(clean)


# In[58]:


clean


# Two different ways to count `2019`s in this pandas Series of strings.  Notice that they give the same answer as the `.dt.year` method from above.

# In[59]:


clean.map(lambda s: s[:4] == "2019").sum()


# In[60]:


clean.map(lambda s: int(s[:4]) == 2019).sum()

