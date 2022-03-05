#!/usr/bin/env python
# coding: utf-8

# # Random numbers in NumPy

# ## Random integers
# 
# Here is the recommended way to make random integers in NumPy.  We first instantiate a "random number generator" that we call `rng`.

# In[1]:


import numpy as np
rng = np.random.default_rng()


# In[2]:


help(rng.integers)


# Making a 10x2 NumPy array of random integers between 1 (inclusive) and 5 (exclusive).

# In[3]:


rng.integers(1,5,size=(10,2))


# Here are two ways to include 5.

# In[4]:


rng.integers(1,6,size=(10,2))


# In[5]:


rng.integers(1,5,size=(10,2),endpoint=True)


# ## Random real numbers
# 
# If making random real numbers, the range is always between 0 and 1; there is no way to specify the upper and lower bounds as inputs to the function.  So to increase the range of outputs, you multiply, and to shift the range of outputs, you add.

# In[6]:


rng.random(size=(10,2))


# Random real numbers between 0 and 30:

# In[7]:


30*rng.random(size=(10,2))


# Random real numbers between 5 and 35:

# In[8]:


5 + 30*rng.random(size=(10,2))

