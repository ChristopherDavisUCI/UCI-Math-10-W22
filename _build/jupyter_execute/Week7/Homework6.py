#!/usr/bin/env python
# coding: utf-8

# # Homework 6
# 
# List your name and the names of any collaborators at the top of this notebook.
# 
# (Reminder: It’s encouraged to work together; you can even submit the exact same homework as another student or two students, but you must list each other’s names at the top.)

# ## Introduction
# 
# This homework will be good preparation for the final project, because it is more open-ended than the typical homeworks.
# 
# The goal is to use `KNeighborsClassifier` to investigate some aspect of the taxis dataset from Seaborn.  Originally I was going to tell you specifically what columns to use, but I wasn't satisfied with my results and I think you can come up with something better.

# In[1]:


import seaborn as sns
import numpy as np
import pandas as pd
import altair as alt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import log_loss


# In[2]:


df = sns.load_dataset("taxis")


# In[3]:


df.head()


# ## Assignment
# 
# Pose a question related to the taxis dataset loaded above, and investigate that question using `KNeighborsClassifier`.  For example, if we were instead working with the penguins dataset, the question might be something like, "Can we use flipper length and bill length to predict the species of penguin?"  Make sure you're posing a *classification* problem and not a regression problem.
# 
# Address the following points.
# 
# 1. State explicitly what question you are investigating.  (It doesn't need to be a question with a definitive answer.)
# 
# 2.  Convert at least one of the `pickup` and/or `dropoff` column into a `datetime` data type, and use some aspect of that column in your analysis.  (For example, you could use `.dt.hour` or `.dt.day_name()`... for some reason `hour` does not include parentheses but `day_name()` does include parentheses.)
# 
# 3.  Include at least one Boolean column in your `X` data.  (There aren't any Boolean columns in this dataset, so you will have to produce one.  Producing new columns like this is called *feature engineering*.  For example, with the penguins dataset, we could create a Boolean column indicating if the bill length is over 5cm.)
# 
# 3.  For numerical columns (or Boolean columns) that you use in your `X` data, rescale them using `StandardScaler` and use the scaled versions when fitting (and predicting) with `KNeighborsClassifier`.  (Actually, every column fed to the `X` portion of the `KNeighborsClassifier` should be either numerical or Boolean... it does not accept categorical values in the `X`.  If you want to use a categorical value in the `X`, you need to convert it somehow into a numerical or Boolean value.)
# 
# 4.  Use `train_test_split` to attempt to detect over-fitting or under-fitting.  Evaluate the performance of your classifier using the `log_loss` metric that was imported above.
# 
# 6.  Make a plot in Altair related to your question.  (It's okay if the plot is just loosely related to your question.  For example, if you are using many different columns, it would be difficult to show all of that information in a plot.)  This dataset is about 6000 rows long, which is too long for Altair by default, but you can disable that using `alt.data_transformers.disable_max_rows()`.  (This would be a bad idea for a huge dataset, but with this dataset it should be fine.)
# 
# 8.  State a specific value of `k` for which this `KNeighborsClassifier` seems to perform best (meaning the `log_loss` error for the test set is lowest... it's okay if your k is just an estimate).  For example, if you look at the test error curve at the bottom of the notebook from [Wednesday Week 6](https://christopherdavisuci.github.io/UCI-Math-10-W22/Week6/Week6-Wednesday.html), you'll see that for that problem, the regressor performed best when 1/k was between 0.1 and 0.2, so when k was between 5 and 10.  (If you find that the performance is best with the biggest possible `k`, that probably means that `KNeighborsClassifier` is not an effective tool for your specific choice of `X` data and `y` data.  That's okay but it would be even better if you could make some adjustment.)

# ## Submission
# Download the .ipynb file for this notebook (click on the folder icon to the left, then the … next to the file name) and upload the file on Canvas.
