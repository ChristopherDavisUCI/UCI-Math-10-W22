#!/usr/bin/env python
# coding: utf-8

# # Video notebooks
# 
# ## MNIST dataset
# 
# Based on Chapter 3 of [Hands-On Machine Learning (2nd edition)](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/) by Aurélien Géron.

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


from sklearn.datasets import fetch_openml


# In[3]:


# Will take about one minute to run
mnist = fetch_openml('mnist_784', version = 1)


# In[4]:


mnist.keys()


# In[5]:


type(mnist)


# In[6]:


X = mnist['data']


# In[7]:


type(X)


# In[8]:


y = mnist['target']


# In[9]:


type(y)


# In[10]:


X.shape


# In[11]:


X.info()


# In[12]:


X.head()


# In[13]:


X.iloc[0]


# In[14]:


A_pre = X.iloc[0].to_numpy()


# In[15]:


type(A_pre)


# In[16]:


A_pre.shape


# In[17]:


28**2


# In[18]:


A = A_pre.reshape(28,28)


# In[19]:


A.shape


# In[20]:


A


# In[21]:


import matplotlib.pyplot as plt


# In[22]:


fig, ax = plt.subplots()


# In[23]:


ax.imshow(A)


# In[24]:


fig


# In[25]:


ax.imshow(A, cmap='binary')


# In[26]:


fig


# In[27]:


y


# In[28]:


B = X.iloc[2].to_numpy().reshape(28,28)
fig2, ax2 = plt.subplots()


# In[29]:


ax2.imshow(B)
fig2


# In[30]:


y.iloc[2]


# ## train_test_split

# In[31]:


from sklearn.model_selection import train_test_split


# In[32]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)


# In[33]:


X_train.shape


# In[34]:


type(X_train)


# In[35]:


y_train.shape


# In[36]:


X_test.shape


# In[37]:


y_test.shape


# ## Logistic Regression

# In[38]:


from sklearn.linear_model import LogisticRegression


# In[39]:


clf = LogisticRegression()


# In[40]:


clf.fit(X_train, y_train)


# In[41]:


clf.predict(X_train)


# In[42]:


clf.predict(X_test)


# In[43]:


y_test


# In[44]:


clf.predict(X_test) == y_test


# In[45]:


np.count_nonzero(clf.predict(X_test) == y_test)/len(X_test)


# It's a little more accurate on the training set, but not too much higher:

# In[46]:


np.count_nonzero(clf.predict(X_train) == y_train)/len(X_train)


# ## Predicted probabilities

# In[47]:


clf.predict(X_test)


# In[48]:


probs = clf.predict_proba(X_test)


# In[49]:


probs.shape


# In[50]:


probs[0]


# In[51]:


probs[0].argmax()


# In[52]:


clf.classes_


# In[53]:


probs[0].sum()


# In[54]:


probs.sum(axis=1)


# In[55]:


probs.argmax(axis=1)


# In[56]:


probs[2]


# In[57]:


probs[2,3]


# In[58]:


probs[2,8]


# In[59]:


B = X_test.iloc[2].to_numpy().reshape(28,28)
fig2, ax2 = plt.subplots()
ax2.imshow(B)

