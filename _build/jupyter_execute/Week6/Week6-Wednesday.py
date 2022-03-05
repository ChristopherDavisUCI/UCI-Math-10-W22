#!/usr/bin/env python
# coding: utf-8

# # K-Nearest Neighbors Regressor
# 
# [YuJa recording](https://uci.yuja.com/V/Video?v=4348961&node=14654381&a=1301700135&autoplay=1)
# 
# Before the recording, we introduced the K-Nearest Neighbors Classifier and the K-Nearest Neighbors Regressor.  We mentioned that larger K corresponds to smaller variance (so over-fitting is more likely to occur with smaller values of K).  We also discussed the training error curve and test error curve, like from the figures in Chapter 2 of *Introduction to Statistical Learning*.

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import altair as alt
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error


# In[2]:


df = sns.load_dataset("penguins")
#df = df.dropna()
df.dropna(inplace=True)


# In[3]:


df.info()


# It would be better to rescale the data first (i.e., to normalize the data).  We'll talk about that soon but we're skipping it for now.

# In[4]:


X_train, X_test, y_train, y_test = train_test_split(
    df[["bill_length_mm","bill_depth_mm","flipper_length_mm"]], df["body_mass_g"], test_size = 0.5)


# In[5]:


X_train.shape


# The syntax for performing K-Nearest Neighbors regression using scikit-learn is essentially the same as the syntax for performing linear regression.

# In[6]:


reg = KNeighborsRegressor(n_neighbors=10)


# In[7]:


reg.fit(X_train, y_train)


# In[8]:


reg.predict(X_test)


# In[9]:


X_test.shape


# In[10]:


mean_absolute_error(reg.predict(X_test), y_test)


# In[11]:


mean_absolute_error(reg.predict(X_train), y_train)


# The above numbers are similar, with `reg` performing just slightly better on the training data.  That suggests that for this training set, we are not overfitting the data when using K=10.

# In[12]:


def get_scores(k):
    reg = KNeighborsRegressor(n_neighbors=k)
    reg.fit(X_train, y_train)
    train_error = mean_absolute_error(reg.predict(X_train), y_train)
    test_error = mean_absolute_error(reg.predict(X_test), y_test)
    return (train_error, test_error)


# In[13]:


get_scores(10)


# In[14]:


get_scores(1)


# In[15]:


df_scores = pd.DataFrame({"k":range(1,150),"train_error":np.nan,"test_error":np.nan})


# In[16]:


df_scores


# In[17]:


df_scores.loc[0,["train_error","test_error"]] = get_scores(1)


# In[18]:


df_scores.head()


# We often avoid using `for` loops in Math 10, but I couldn't find a better way to fill in this data.  Let me know if you see a more Pythonic approach!

# In[19]:


for i in df_scores.index:
    df_scores.loc[i,["train_error","test_error"]] = get_scores(df_scores.loc[i,"k"])


# In[20]:


df_scores


# Usually when we plot a test error curve, we want higher flexibility (= higher variance) on the right.  Since higher values of K correspond to lower flexibility, we are going to add a column to the DataFrame containing the reciprocals of the K values.

# In[21]:


df_scores["kinv"] = 1/df_scores.k


# In[22]:


ctrain = alt.Chart(df_scores).mark_line().encode(
    x = "kinv",
    y = "train_error"
)


# In[23]:


ctest = alt.Chart(df_scores).mark_line(color="orange").encode(
    x = "kinv",
    y = "test_error"
)


# The blue curve is the training error, while the orange curve is the test error.  Notice how underfitting occurs for very high values of K and notice how overfitting occurs for smaller values of K.

# In[24]:


ctrain+ctest

