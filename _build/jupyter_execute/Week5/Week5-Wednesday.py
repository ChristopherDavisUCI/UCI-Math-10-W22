#!/usr/bin/env python
# coding: utf-8

# # Introduction to scikit-learn
# 
# A recording is available on [YuJa](https://uci.yuja.com/V/Video?v=4314691&node=14560623&a=33255951&autoplay=1)
# 
# Before that video (and this notebook) we spent about 15 minutes introducing the Machine Learning portion of Math 10.  The most important concept covered was the concept of a *cost function* or *loss function*, that can be used to measure the performance of a model.  For example, when trying to decide which line (or plane or ...) best fits data using linear regression, the word *best* means the equation which minimizes the cost function.  Natural choices of cost function for linear regression are *Mean Squared Error* (MSE) and *Mean Absolute Error* (MAE).

# In[1]:


import numpy as np
import pandas as pd
import altair as alt


# In[2]:


df = pd.read_csv("../data/spotify_dataset.csv", na_values=" ")


# In[3]:


alt.Chart(df).mark_circle().encode(
    x = "Acousticness",
    y = "Energy"
)


# In[4]:


from sklearn.linear_model import LinearRegression


# In[5]:


reg = LinearRegression()


# If that syntax looks strange, maybe this will make it look more familiar.

# In[6]:


from numpy.random import default_rng
# vs our usual rng = np.random.default_rng()
rng = default_rng()


# The following is a very common error.  The problem is that `df["Acousticness"]` is one-dimensional, and scikit-learn wants something two-dimensional.  (It's fine and maybe required for the second input, `df["Energy"]`, to be one-dimensional.)

# In[7]:


reg.fit(df["Acousticness"], df["Energy"])


# In[8]:


# Double brackets say to return a DataFrame, not a Series
df[["Acousticness"]]


# In[9]:


df[["Acousticness","Song Name", "Artist"]]


# Another common error, not removing missing values.

# In[10]:


reg.fit(df[["Acousticness"]], df["Energy"])


# In[11]:


df_clean = df[~df.isna().any(axis=1)]


# In[12]:


df_clean.shape


# In[13]:


reg.fit(df_clean[["Acousticness"]], df_clean["Energy"])


# Look at the scatter plot above.  This coefficient and this intercept should look very reasonable.

# In[14]:


reg.coef_


# In[15]:


reg.intercept_


# In[16]:


reg.predict(df_clean[["Acousticness"]])


# In[17]:


# could fix this warning using .copy() earlier
df_clean["pred"] = reg.predict(df_clean[["Acousticness"]])


# In[18]:


df_clean


# In[19]:


c1 = alt.Chart(df_clean).mark_circle().encode(
    x = "Acousticness",
    y = "Energy"
)


# This is our first time using `mark_line`, which is used to draw line charts (like the default plot in Matlab).  This method is probably a little inefficient, because it is using all 1545 rows of `df_clean` to make the straight line.

# In[20]:


c2 = alt.Chart(df_clean).mark_line(color="red").encode(
    x = "Acousticness",
    y = "pred"
)


# Unlike `c1|c2` which puts the charts side-by-side, `c1+c2` layers the charts on top of each other.

# In[21]:


c1+c2


# In[22]:


df_clean.columns


# Linear regression works basically the same with multiple input variables.

# In[23]:


reg2 = LinearRegression()


# In[24]:


df_clean[["Acousticness","Speechiness","Valence"]]


# In[25]:


reg2.fit(df_clean[["Acousticness","Speechiness","Valence"]], df_clean["Energy"])


# In[26]:


reg2.coef_


# ## Interpretability
# 
# Linear regression is not the fanciest machine learning model, but it is probably the most interpretable model.  For example, the coefficients above suggest that Acousticness correlates to having less energy, Valence suggests more energy.  (I'm not using "correlates" in a technical sense.)

# In[27]:


reg2.predict(df_clean[["Acousticness","Speechiness","Valence"]])


# In[28]:


from sklearn.metrics import mean_squared_error


# In[29]:


mean_squared_error(df_clean["Energy"],df_clean["pred"])


# In[30]:


((df_clean["Energy"] - df_clean["pred"])**2).mean()


# In[31]:


mean_squared_error(reg2.predict(df_clean[["Acousticness","Speechiness","Valence"]]), df_clean["Energy"])

