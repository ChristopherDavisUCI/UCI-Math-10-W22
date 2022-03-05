#!/usr/bin/env python
# coding: utf-8

# # Reading a csv file
# 
# pandas is probably the most important library for Math 10.
# 
# ## Exploring the data
# 
# Here are some first things you might try when importing a new dataset.

# In[1]:


import pandas as pd


# On my personal computer, the `cars.csv` file is located in a different folder from this notebook.  If it's in the same folder (like it is on Deepnote), you can just type `pd.read_csv("cars.csv")`.

# In[2]:


df = pd.read_csv("../data/cars.csv")


# Viewing the first 5 rows of the dataset.  You should think of each row as corresponding to one *instance* or one *data point*.

# In[3]:


df.head()


# Or the first 10 rows.

# In[4]:


df.head(10)


# The number of rows and columns.

# In[5]:


df.shape


# The names of the columns.

# In[6]:


df.columns


# Some data about the numeric columns.

# In[7]:


df.describe()


# Some more information.  You can use this next information to determine which columns have missing values.

# In[8]:


df.info()


# The data types of the columns.

# In[9]:


df.dtypes


# You can see how the numeric columns are *correlated* with each other.  These correlation values range between -1 and 1, with 1 meaning the two columns are perfectly correlated.

# In[10]:


df.corr()


# ## Indexing
# 
# There are many different ways to select data within a pandas DataFrame.  The best way to remember them is to practice using them.

# In[11]:


# Reminder of how df starts
df.head()


# The entry in the 2nd row, 3rd column (remember we start counting at 0).

# In[12]:


df.iloc[2,3]


# The entry in the row with label 2 and the column with label Displacement.  (Notice that the index and the label is the same for the rows; this is not uncommon.)

# In[13]:


df.loc[2,"Displacement"]


# You can get an entire row or column with the same syntax, using a colon `:` to represent "all rows or all columns".

# In[14]:


# The row at index 2.
df.iloc[2,:]


# In[15]:


# The column at index 3.
df.iloc[:,3]


# In[16]:


# The column with label "Displacement".
df.loc[:,"Displacement"]


# There is an abbreviation for getting a certain column, using its label.

# In[17]:


df["Displacement"]


# The next abbreviation does not always work, but can be a further shortcut.  It is called "attribute" access.  The subtleties of attribute access won't be important for us in Math 10; you can read about those subtleties in the [pandas documentation](https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#attribute-access).

# In[18]:


df.Displacement


# ## Missing values
# 
# An important concept (especially when working with real-world datasets) is the concept of missing data.  This particular dataset has missing values in the `Miles_per_Gallon` column and in the `Horsepower` column.  In this DataFrame, the missing data is denoted by the NumPy object `np.nan` which stands for "not a number".

# In[19]:


df.info()


# We create a Boolean DataFrame using the method `isna()`.  This DataFrame will be `True` where there are null values.

# In[20]:


df.isna()


# Let's find where the `np.nan` values are in the `Miles_per_Gallon` column.

# In[21]:


df[df["Miles_per_Gallon"].isna()]


# The same thing for the `Horsepower` column.

# In[22]:


df[df["Horsepower"].isna()]


# In Python, logical `or` is usually spelled out.

# In[23]:


True or True


# In[24]:


True or False


# In[25]:


False or False


# The equivalent of `or` in pandas is denoted with a vertical line `|`, which is sometimes called "pipe".

# In[26]:


df[df["Miles_per_Gallon"].isna() | df["Horsepower"].isna()]


# A fancier and more robust method is to use `any`.  In this example, `axis = 1` is saying, look one row at a time.  So `df.isna().any(axis=1)` is asking if there are any missing values in the entire row.

# In[27]:


df[df.isna().any(axis=1)]


# If we were to instead use `axis=0`, it would ask if there were any missing values in the entire column.

# In[28]:


df.isna().any(axis=0)

