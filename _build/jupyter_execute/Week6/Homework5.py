#!/usr/bin/env python
# coding: utf-8

# # Homework 5
# 
# List your name and the names of any collaborators at the top of this notebook.
# 
# (Reminder: It’s encouraged to work together; you can even submit the exact same homework as another student or two students, but you must list each other’s names at the top.)
# 
# This homework is divided into two parts (please submit both together).  In the first part, you will practice with scikit-learn and the MNIST dataset.  In the second part, you will be introduced to some of the most famous concepts in machine learning (over-fitting and the bias-variance tradeoff).

# ## Part 1: scikit-learn and MNIST
# 
# ### Question 1
# 
# In the language of the Altair [data encoding types](https://altair-viz.github.io/user_guide/encoding.html#encoding-data-types), why should the labels (0,1,...,9) from the MNIST handwritten digit dataset be considered as a *nominal* data type, rather than a *quantitative* or an *ordinal* data type?  (This should seem counter-intuitive at first, since the labels do have a clear ordering.)

# ### Question 2
# 
# Lost the MNIST data and many useful scikit-learn functions by evaluating the cell below.  It will probably take about one minute to execute. (**Warning**.  I tried loading this  twice, and I ran out of memory.  So try to only evaluate this cell once per session.)

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.datasets import fetch_openml
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

mnist = fetch_openml('mnist_784', version = 1)


# Get the image data from `mnist` and call it `X`, also get the label data and call it `y`.

# ### Question 3
# 
# Replace `y` with a numeric Series using `pd.to_numeric`.  Also call this new Series `y`.  (Check.  If you look at `y`, it should be a length 70000 pandas Series with dtype `int64`.)

# ### Question 4

# Create `X_train, X_test, y_train, y_test` using `train_test_split` with a test size of `0.9`.  (This is a larger than usual test size.  Something like `0.2` is more common.)

# ### Question 5
# 
# How many data points are in the training set?  (There are two ways to answer this, either mathematically using the fact that the test size is `0.9` and the full data set contains 70000 samples, or by evaluating the length of `X_train` or `y_train`.)

# ### Question 6
# 
# Fit a LogisticRegression classifier using `X_train` and `y_train`.  (A warning shows up, the same as in the video.  We are just ignoring this warning for now.)

# ### Question 7
# 
# Use the `score` method of the classifier to evaluate the performance on `X_train` and `y_train`.  You can read about the score method in the [scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression.score).

# ### Question 8
# 
# That score value should be the same as the proportion of correct predictions by your classifier.  Verify that they are the same.  (Use the `predict` method, then create a Boolean array by comparing the result to `y_train`, then use `np.count_nonzero` and divide by the length.)

# ### Question 9
# 
# What is the score for the test set?

# ### Question 10
# 
# Do the results suggest over-fitting?  Why or why not?

# ### Question 11
# 
# We are now going to do the same thing using Linear Regression instead of Logistic Regression.  Why is that a bad idea on the MNIST dataset?

# ### Question 12
# 
# Fit a Linear Regression object using the same `X_train` and `y_train` as above.

# ### Question 13
# 
# What is the `mean_absolute_error` between the predicted value on `X_train` (use `predict`) and `y_train` (don't use predict)?

# ### Question 14
# 
# What is the `mean_absolute_error` between the predicted value on `X_test` and `y_test`?

# ### Question 15
# 
# Do the results suggest over-fitting? Why or why not?

# ## Part 2: Reading questions
# 
# This portion of the homework is based on Chapter 2 (Statistical Learning) of *Introduction to Statistical Learning*.  You can download this chapter from on campus using [SpringerLink](https://link.springer.com/book/10.1007/978-1-4614-7138-7) or you can find a download link in the [Week 6 checklist](https://canvas.eee.uci.edu/courses/42645/pages/week-6-checklist?module_item_id=1550957) on Canvas.
# 
# 1.  In the *income* formula from page 22, what is the significance of $\beta_1$ and $\beta_2$ being positive or negative?
# 
# 1.  Which of the curves in Figure 2.9 is *underfitting* the data?  Which is *overfitting* the data?
# 
# 1.  What is meant by the two green dots in the right-hand panel of Figure 2.9?
# 
# 1.  Describe in your own words what is meant by the *Bias-Variance Tradeoff* (section 2.2.2). 
# 
# 1.  Why is Equation (2.5) on page 29 not reasonable to use for a classification problem?

# ## Submission
# Download the .ipynb file for this notebook (click on the folder icon to the left, then the … next to the file name) and upload the file on Canvas.
