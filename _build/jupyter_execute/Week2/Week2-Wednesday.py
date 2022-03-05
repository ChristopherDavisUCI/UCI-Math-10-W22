#!/usr/bin/env python
# coding: utf-8

# # NumPy and pandas
# 
# Review of NumPy and examples using pandas.
# 
# [Recording of lecture from 1/12/2022](https://uci.zoom.us/rec/share/ZX9cbbE7zeJR-MRWu9Rmj1_r7IMlliQOe01P27fln1RgxddHgchdl8x6HYmFKnvU.DVx5i65Sb_b3JrC_)

# ## Warm-up exercise
# 
# 1. Define an 8x4 NumPy array A of random integers between 1 and 10 (inclusive).
# 
# 2. Each row of A has four columns.  Let [x,y,z,w] denote one of these rows.  What is the probability that x > y?

# In[1]:


import numpy as np


# In[2]:


rng  = np.random.default_rng()


# In[3]:


help(rng.integers)


# In[4]:


A = rng.integers(1,11,size=(8,4))
A


# In mathematics, it doesn't make sense to ask if a vector is strictly greater than another vector.  In NumPy, this comparison is done *elementwise*.

# In[5]:


A[:,0] > A[:,1]


# It's the same with equality: they are compared elementwise.

# In[6]:


A[:,0] == A[:,1]


# With a `list` instead of an `np.array`, then equality means "are the lists exactly the same, with the same elements in the same positions?"

# In[7]:


[4,2,3] == [4,2,3]


# In[8]:


[4,2,3] == [4,3,2]


# In[9]:


np.array([1,2,3]) == np.array([4,2,3])


# In[10]:


set([4,2,3]) == set([4,3,2,2,4,2,2])


# In[11]:


[1,2,3] == [4,2,3]


# This next cell produces an example of a Boolean array.

# In[12]:


A[:,0] > A[:,1]


# Counting how often `True` appears.

# In[13]:


np.count_nonzero(A[:,0] > A[:,1])


# We think of each row as being one "experiment".  We can find the number of rows using `len`.

# In[14]:


# number of experiments = number of rows
len(A)


# We estimate the probability using "number of successes"/"number of experiments".  It won't be accurate yet, because we are using so few experiments.

# In[15]:


# prob estimate using len(A) experiments
np.count_nonzero(A[:,0] > A[:,1])/len(A)


# Using ten million experiments.

# In[16]:


A = rng.integers(1,11,size=(10**7,4))
np.count_nonzero(A[:,0] > A[:,1])/len(A)


# If we do the same thing, we should get a very similar answer, but it won't be exactly the same, since these are estimates using random experiments.

# In[17]:


A = rng.integers(1,11,size=(10**7,4))
np.count_nonzero(A[:,0] > A[:,1])/len(A)


# ## pandas
# 
# Probably the most important Python library for Math 10.  Essentially everything we did earlier in this notebook, we can also do in pandas.  The library pandas also has a lot of extra functionality that will help us work with datasets.

# In[18]:


import pandas as pd


# In[19]:


A = rng.integers(1,11,size=(8,4))
type(A)


# In[20]:


A.shape


# We convert this NumPy array to a pandas DataFrame.  (Make sure you capitalize DataFrame correctly.)

# In[21]:


df = pd.DataFrame(A)
df


# The syntax for getting the zeroth column of a pandas DataFrame is a little longer than the NumPy syntax.

# In[22]:


# zeroth column of df
df.iloc[:,0]


# This column is a pandas Series.

# In[23]:


type(df.iloc[:,0])


# We can compare the entries in these columns elementwise, just like we did using NumPy.

# In[24]:


df.iloc[:,0] > df.iloc[:,1]


# Here is the most efficient way I know to count `True`s in a pandas Boolean Series.

# In[25]:


(df.iloc[:,0] > df.iloc[:,1]).sum()


# We can again get the number of rows using `len`.

# In[26]:


len(df)


# In[27]:


df.shape


# Here is the probability estimate.

# In[28]:


# Not using enough experiments
((df.iloc[:,0] > df.iloc[:,1]).sum())/len(df)


# Here we increase the number of experiments, but we forget to change `df`.

# In[29]:


# forgot to update df
A = rng.integers(1,11,size=(10**7,4))
((df.iloc[:,0] > df.iloc[:,1]).sum())/len(df)


# Here is the correct version.

# In[30]:


A = rng.integers(1,11,size=(10**7,4))
df = pd.DataFrame(A)
((df.iloc[:,0] > df.iloc[:,1]).sum())/len(df)


# In[31]:


A = rng.integers(1,11,size=(8,4))
df = pd.DataFrame(A)


# In[32]:


A


# In[33]:


df


# Changing column names.

# In[34]:


df.columns = ["a","b","m","chris"]


# In[35]:


df


# There are two similar operations, `df.loc` and `df.iloc`.  The operation `df.loc` refers to rows and columns by their names, whereas `df.iloc` refers to rows and columns by their index.

# In[36]:


df.loc[:,"b"]


# In[37]:


df.iloc[:,1]


# There is a common shortcut for referring to a column by its name.

# In[38]:


# abbreviation
df["b"]


# This next command says, give me the 1st-4th rows (not including the right endpoint) in the 2nd column.

# In[39]:


df.iloc[1:4,2]


# Somewhat confusingly, right endpoints are included when using `loc`.

# In[40]:


df.loc[1:4,"m"]


# You can use this same sort of notation to set values.

# In[41]:


df


# In[42]:


df.iloc[1:4,2] = -1000


# In[43]:


df


# That same sort of notation also works for NumPy arrays.

# In[44]:


B = np.array(df)
B


# In[45]:


B[1:4,0] = 3


# In[46]:


B


# You can also set multiple different values.  The following says, in the 1st column (remember that we start counting at 0), set the elements from the 5th, 6th, 7th rows to be 100, 200, 300, respectively.

# In[47]:


B[5:,1] = [100,200,300]


# In[48]:


B

