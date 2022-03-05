#!/usr/bin/env python
# coding: utf-8

# # Boolean arrays in NumPy
# 
# A Boolean array by itself is not very interesting; it's just a NumPy array whose entries are either `True` or `False`.

# In[1]:


import numpy as np


# In[2]:


bool_arr = np.array([True,True,False,True])
bool_arr


# The reason Boolean arrays are important is that they are often produced by other operations.

# In[3]:


arr = np.array([3,1,4,1])
arr < 3.5


# The number of `True`s in a Boolean array can be counted very efficiently using `np.count_nonzero`.  Reminders:
# * s means seconds;
# * ms means milliseconds, $10^{-3}$;
# * µs means microseconds, $10^{-6}$;
# * ns means nanoseconds, $10^{-9}$.

# From a small example, it might seem like the NumPy method is slower:

# In[4]:


my_list = [3,1,4,3,5]
my_array = np.array(my_list)


# In[5]:


my_list.count(3)


# In[6]:


get_ipython().run_cell_magic('timeit', '', 'my_list.count(3)')


# In[7]:


np.count_nonzero(my_array==3)


# In[8]:


get_ipython().run_cell_magic('timeit', '', 'np.count_nonzero(my_array==3)')


# But for a longer example, it will be clear that the NumPy method is faster.  In this example, our array and list have length ten million.

# In[9]:


rng = np.random.default_rng()
my_array = rng.integers(1,6,size=10**7)
my_list = list(my_array)


# In[10]:


my_list.count(3)


# In[11]:


np.count_nonzero(my_array==3)


# In[12]:


get_ipython().run_cell_magic('timeit', '', 'my_list.count(3)')


# In[13]:


get_ipython().run_cell_magic('timeit', '', 'np.count_nonzero(my_array==3)')

