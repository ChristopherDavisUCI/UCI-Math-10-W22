#!/usr/bin/env python
# coding: utf-8

# # Week 8 Friday
# 
# [YuJa recording](https://uci.yuja.com/V/Video?v=4481600&node=14988658&a=174537677&autoplay=1)

# ## Code from Sample Midterm 2, Question 1

# In[1]:


import pandas as pd
import altair as alt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import log_loss

df = pd.read_csv("../data/spotify_dataset.csv", na_values=" ")
df.dropna(inplace=True)
df = df[(df["Artist"] == "Taylor Swift")|(df["Artist"] == "Billie Eilish")]
df = df[df["Artist"].isin(["Taylor Swift", "Billie Eilish"])]

alt.Chart(df).mark_circle().encode(
    x="Tempo",
    y="Energy",
    color="Artist"
)


# ## 1a
# 
# Rewrite the Taylor Swift/Billie Eilish line so that it uses `isin`.
# 
# We can replace
# ```
# df = df[(df["Artist"] == "Taylor Swift")|(df["Artist"] == "Billie Eilish")]
# ```
# by
# ```
# df = df[df["Artist"].isin(["Taylor Swift", "Billie Eilish"])]
# ```

# In[2]:


import pandas as pd
import altair as alt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import log_loss

df = pd.read_csv("../data/spotify_dataset.csv", na_values=" ")
df.dropna(inplace=True)
df = df[df["Artist"].isin(["Taylor Swift", "Billie Eilish"])]

alt.Chart(df).mark_circle().encode(
    x="Tempo",
    y="Energy",
    color="Artist"
)


# ## When do we need to use index?
# 
# For some similar problems using `isin`, we have also had to use `index`.  Why didn't we in this case?
# 
# Here is a typical example where we have used index:
# * Find the sub-DataFrame containing only the 5 most frequent artists.
# 
# As a preliminary step, we look at the top 5 rows of `value_counts()`.

# In[3]:


df = pd.read_csv("../data/spotify_dataset.csv", na_values=" ")
df.dropna(inplace=True)
df["Artist"].value_counts()[:5]


# What if we want to replace the `["Taylor Swift", "Billie Eilish"]` portion above with this list of 5 artists?
# 
# Directly converting this Series into a list *does not* work, because it yields the values, whereas we want the keys.

# In[4]:


list(df["Artist"].value_counts()[:5])


# We can access the keys (or maybe I should call it *index*) as follows.

# In[5]:


df["Artist"].value_counts()[:5].index


# You could also go in the other order, first extracting the index, and then taking the first 5 entries.

# In[6]:


df["Artist"].value_counts().index[:5]


# Here we use `isin` together with `index`.

# In[7]:


df = pd.read_csv("../data/spotify_dataset.csv", na_values=" ")
df.dropna(inplace=True)
df = df[df["Artist"].isin(df["Artist"].value_counts()[:5].index)]


# In[8]:


df.head(5)


# Only the 5 most frequent artists remain.  (There is actually a tie for 5th, so your results may be different.)

# In[9]:


df["Artist"].value_counts()


# ## Back to the smaller DataFrame

# We changed `df` in the previous part, so we get back to the result of 1a.

# In[10]:


import pandas as pd
import altair as alt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import log_loss

df = pd.read_csv("../data/spotify_dataset.csv", na_values=" ")
df.dropna(inplace=True)
df = df[df["Artist"].isin(["Taylor Swift", "Billie Eilish"])]

alt.Chart(df).mark_circle().encode(
    x="Tempo",
    y="Energy",
    color="Artist"
)


# ## 1b
# 
# Rescale the "Tempo" and "Energy" data using `StandardScaler`. Overwrite the current columns in `df` using the rescaled data.

# In[11]:


scaler = StandardScaler()


# In[12]:


scaler.fit(df[["Tempo","Energy"]]) # Just the Tempo and Energy columns


# In[13]:


df[["Tempo","Energy"]] = scaler.transform(df[["Tempo","Energy"]])


# Notice how the "Tempo" and "Energy" columns have been rescaled.

# In[14]:


df.head(10)


# Aside: It's also possible to do the `fit` and the `transform` all in a single line.  Here we apply both `fit` and `transform` to the "Valence" and "Danceability" columns.

# In[15]:


scaler.fit_transform(df[["Valence","Danceability"]])


# ## 1c
# 
# We will eventually use K-Nearest Neighbors on these two columns. Why is rescaling them natural?
# 
# If you look at the Altair chart and imagine it using the same scale for both the x and y-axes, the chart could get extremely spread out in the x direction, because the x values range from about 60 to 210, whereas the y values only range from about 0 to 1.  So if we want to compute distance between these, we should rescale.
# 
# (Another possible answer is that the units are not the same.)

# ## 1d
# 
# Our goal is to predict the Artist using the two scaled columns. Divide this data into a training set and a test set using `train_test_split`.
# 
# We'll save 20% of the data as our test set.

# In[16]:


X_train, X_test, y_train, y_test = train_test_split(df[["Tempo","Energy"]], df["Artist"], test_size=0.2)


# In[17]:


X_train


# ## 1e
# 
# Fit either `KNeighborsClassifier` or `KNeighborsRegressor` to this data using the training set.
# 
# Only `KNeighborsClassifier` makes sense, because the "Artist" values are categorical, not quantitative/numerical.  (For something like MNIST, the values are arguably numerical, but one should still use `KNeighborsClassifier` in that case, because the values are discrete, and their order is not significant.
# 
# Here we use 6 neighbors.

# In[18]:


clf = KNeighborsClassifier(n_neighbors=6)


# In[19]:


clf.fit(X_train, y_train)


# ## 1f
# 
# Evaluate the performance of the model on the test set using `log_loss`.

# In[20]:


clf.predict(X_test)


# In[21]:


clf.classes_


# In[22]:


clf.predict_proba(X_test)


# In[23]:


log_loss(y_test, clf.predict_proba(X_test), labels=['Billie Eilish', 'Taylor Swift'])


# This `log_loss` value by itself doesn't mean much, but if we make a change to the model and evaluate it again, then it is more meaningful.  For example, let's try using 10 neighbors instead of 6 neighbors.

# In[24]:


clf = KNeighborsClassifier(n_neighbors=10)
clf.fit(X_train, y_train)
log_loss(y_test, clf.predict_proba(X_test), labels=['Billie Eilish', 'Taylor Swift'])


# Because the loss score decreased, this evidence suggests for this particular data, K-Nearest Neighbors performs better with 10 neighbors than with 6 neighbors.

# ## 1g
# 
# Change the Altair chart so that it uses the predicted Artist class, not the actual Artist.
# 
# Here are a few examples with different numbers of neighbors.  Notice how the model appears to become less flexible (i.e., appears to have more bias) as the number of neighbors increases.

# In[25]:


alt.Chart(df).mark_circle().encode(
    x="Tempo",
    y="Energy",
    color="Artist"
).properties(
    title="Original"
)


# In[26]:


clf = KNeighborsClassifier(n_neighbors=1)
clf.fit(X_train, y_train)
df["pred"] = clf.predict(df[["Tempo","Energy"]])
alt.Chart(df).mark_circle().encode(
    x="Tempo",
    y="Energy",
    color="pred"
).properties(
    title="n_neighbors = 1"
)


# In[27]:


clf = KNeighborsClassifier(n_neighbors=5)
clf.fit(X_train, y_train)
df["pred"] = clf.predict(df[["Tempo","Energy"]])
alt.Chart(df).mark_circle().encode(
    x="Tempo",
    y="Energy",
    color="pred"
).properties(
    title="n_neighbors = 5"
)


# In[28]:


clf = KNeighborsClassifier(n_neighbors=12)
clf.fit(X_train, y_train)
df["pred"] = clf.predict(df[["Tempo","Energy"]])
alt.Chart(df).mark_circle().encode(
    x="Tempo",
    y="Energy",
    color="pred"
).properties(
    title="n_neighbors = 12"
)


# In[29]:


clf = KNeighborsClassifier(n_neighbors=20)
clf.fit(X_train, y_train)
df["pred"] = clf.predict(df[["Tempo","Energy"]])
alt.Chart(df).mark_circle().encode(
    x="Tempo",
    y="Energy",
    color="pred"
).properties(
    title="n_neighbors = 20"
)

