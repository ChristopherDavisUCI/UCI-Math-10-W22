#!/usr/bin/env python
# coding: utf-8

# # Can simple volume formulas predict mass of complex diamond shapes?
# 
# Author: Tyler Brown
# 
# Course Project, UC Irvine, Math 10, W22

# ## Introduction

# The project is going to based around the diamonds dataset. We will be looking for a correlation in depth of the diamond and its width (defined by its x-axis times its z-axis) and seeing if we can use that to predict the carat. Since the carat of a diamond is a measure of its weight, we're essentially looking to see if we can use the  dimensions of the diamond to predict the weight of the diamond. Since diamonds are tapered objects and not rectangular prisms, I'm eager to see how the machine does in determining carat.  

# ## Beginning of Code

# In these following blocks of code, we import all the necessary libraries for this project and the "diamonds" dataset from seaborn.

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import altair as alt
#from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
#from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import log_loss, mean_squared_error


# In[ ]:


df = sns.load_dataset('diamonds')


# Here, we begin to form the dataframe that we will use. We will be making an initial value-chart because we will reuse the initial one for later tests. 

# In[ ]:


valChart = pd.DataFrame([])
valChart["depth"] = df["depth"]
#valChart["table"] = df["table"]
valChart["width"] = df["x"]*df["z"]
valChart["width"] = valChart["width"].map(lambda x: round(x,4))


# In[ ]:


valChartFirstTest = pd.DataFrame([])
valChartFirstTest = valChart
valChartSecondTest = pd.DataFrame([])
valChartSecondTest = valChart


# Here, we are feeding the machine the data it will be predicting with and testing against. We will be predicting data using Linear Regression

# In[ ]:


X1 = valChartFirstTest[["depth","width"]] 
reg = LinearRegression()
reg.fit(X1, df["carat"])
valChartFirstTest["predicted carat"] = reg.predict(X1)


# Now for initializing the training sets, which are randomly chosen rows from our val-chart. The machine learns from the randomly chosen training rows, and tests the machine's capability for prediction using the remaining rows, the test rows.

# In[ ]:


X1_train, X1_test, y1_train, y1_test = train_test_split(X1, df["carat"],test_size=0.7)
reg.fit(X1_train, y1_train)


# These two values here are a mean-squared error, a way of determining the error of predicted numbers compared to the true numbers (lower values means better performance)

# In[ ]:


mean_squared_error(reg.predict(X1_test),y1_test) 


# In[ ]:


mean_squared_error(reg.predict(X1_train),y1_train)


# As we can see here, the mean squared error is pretty low, however given that the carats are measured to the 100th decimal space, this error is actually mildly high.
# 
#  To curve this error, I would like to direct your attention to a chart below the following code: 

# In[ ]:


valChartFirstTest["carat"] = df["carat"]
firstChartReduced = pd.DataFrame([])
firstChartReduced = valChartFirstTest
firstChartReduced = firstChartReduced[(df["carat"] < .5) | (df["carat"] >= 1.5) & (df["carat"] <= 2)]
stylingChart = firstChartReduced.style


# In[ ]:


def highlight_max(s, props=''):
    #print(s.shape)
    if abs(s[0]-s[1])/s[1] > .1:
        return np.where(s > 0, props, '')
    return
stylingChart.apply(highlight_max, props='color:white;background-color:#7300c4', axis=1, subset=["predicted carat", "carat"]);


# ## Narrowing our Data Set

# As you'll see in the chart below here, all lines that are highlighed purple are where the difference between the predicted value and the true value is too high (i.e. when the percent error was above 10%). What you might notice like I did while scrolling through it is that these large errors existed mostly in the lowest values of "carat", and less in the higher values.

# In[ ]:


stylingChart


# Due to this, I decided to do some tests where I narrowed down the values of our training set to carat values between .75 and 2 in the following code.

# In[ ]:


valChartSecondTest = valChartSecondTest[(df["carat"] >= .75) & (df["carat"] <= 2)]
y2 = df["carat"][(df["carat"] >= .75) & (df["carat"] <= 2)]
y2 = y2[valChartSecondTest["width"] != 0]
valChartSecondTest = valChartSecondTest[valChartSecondTest["width"] != 0]


# Again, we will train the machine and test.

# In[ ]:


X2 = valChartSecondTest[["depth","width"]]
reg = LinearRegression()
reg.fit(X2, y2)
valChartSecondTest["predicted carat"] = reg.predict(X2)


# In[ ]:


X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2,test_size=0.7)
reg.fit(X2_train, y2_train)


# In[ ]:


mean_squared_error(reg.predict(X2_test),y2_test)


# In[ ]:


mean_squared_error(reg.predict(X2_train),y2_train)


# As you can see, the error is now a fraction of what it was in the previous test after we narrowed down the sample size. I believe this is due to the dimensions of the diamond relating to the volume differently as size increases. So narrowing down the volume would be more beneficial
# 
# Below I'm going to remove the data's outliers (that seem to appear due to human errors while inputing data) and graph our predicted carats vs our true carats.

# In[ ]:


valChartSecondTest = valChartSecondTest[valChartSecondTest["width"].between(valChartSecondTest["width"].quantile(.15), valChartSecondTest["width"].quantile(.85))]
valChartSecondTest = valChartSecondTest[valChartSecondTest["carat"].between(valChartSecondTest["carat"].quantile(.05), valChartSecondTest["carat"].quantile(.95))]


# In[ ]:


alt.data_transformers.disable_max_rows()
alt.Chart(valChartSecondTest).mark_bar().encode(
    x="width",
    y="depth",
    color = "predicted carat"

).properties(
    title=f"Second Test Predicted Carat with {valChart.size} Bars",
    #width=alt.Step(100)
    width=1200,
    height = 1000
    
)


# In[ ]:


alt.Chart(valChartSecondTest).mark_bar().encode(
    x="width",
    y="depth",
    color = "carat"

).properties(
    title=f"Second Test True Carat with {valChart.size} Bars",
    width=1200,
    height = 1000
    
)


# As you can see from these graphs, the Predicted Carat graph shows a similar gradient to that of the True Carats, which indicates we were successful in showing the machine a corrolation between weight and carats.  

# Also, the width was probably the more important factor when predicting carat, as I can't see as much change in color (which relates to carat) as I do for width. Due to this, I looked at the co-efficients the regression curve took:

# In[ ]:


reg.coef_


# As you can see, the first coefficient (relating to depth) was about 100 times smaller than that relating to the second coefficient (width), so width played a much bigger part in determining weight of a diamond. 

# ## Summary

# We showed that we can somewhat reliably determine the weight of a diamond given its dimensions by visualizing our performance through graphs and the mean-squared-error test. We noticed that we can get an even better estimate of the weights when we narrow down the tests to closer to the median of true carats. Naturally, volume of the diamond would relate to its weight, however since the diamond isn't a perfect rectangular prism, the weight won't perfectly relate to the weight, but it's good we could make close estimates.

# ## References

# The dataset is from seaborn
# 
# Most of the code is from the course
# 
# Code relating to the variable "stylingChart" was sourced from Pandas documentation on Styles and Apply:
# https://pandas.pydata.org/docs/reference/api/pandas.Series.apply.html#pandas.Series.apply
# 
# https://pandas.pydata.org/pandas-docs/stable/user_guide/style.html#Table-Styles

# <a style='text-decoration:none;line-height:16px;display:flex;color:#5B5B62;padding:10px;justify-content:end;' href='https://deepnote.com?utm_source=created-in-deepnote-cell&projectId=c67a1c31-b1d6-458c-bfa9-cec1ae3cd683' target="_blank">
# <img alt='Created in deepnote.com' style='display:inline;max-height:16px;margin:0px;margin-right:7.5px;' src='data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iODBweCIgaGVpZ2h0PSI4MHB4IiB2aWV3Qm94PSIwIDAgODAgODAiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8IS0tIEdlbmVyYXRvcjogU2tldGNoIDU0LjEgKDc2NDkwKSAtIGh0dHBzOi8vc2tldGNoYXBwLmNvbSAtLT4KICAgIDx0aXRsZT5Hcm91cCAzPC90aXRsZT4KICAgIDxkZXNjPkNyZWF0ZWQgd2l0aCBTa2V0Y2guPC9kZXNjPgogICAgPGcgaWQ9IkxhbmRpbmciIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJBcnRib2FyZCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTEyMzUuMDAwMDAwLCAtNzkuMDAwMDAwKSI+CiAgICAgICAgICAgIDxnIGlkPSJHcm91cC0zIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxMjM1LjAwMDAwMCwgNzkuMDAwMDAwKSI+CiAgICAgICAgICAgICAgICA8cG9seWdvbiBpZD0iUGF0aC0yMCIgZmlsbD0iIzAyNjVCNCIgcG9pbnRzPSIyLjM3NjIzNzYyIDgwIDM4LjA0NzY2NjcgODAgNTcuODIxNzgyMiA3My44MDU3NTkyIDU3LjgyMTc4MjIgMzIuNzU5MjczOSAzOS4xNDAyMjc4IDMxLjY4MzE2ODMiPjwvcG9seWdvbj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0zNS4wMDc3MTgsODAgQzQyLjkwNjIwMDcsNzYuNDU0OTM1OCA0Ny41NjQ5MTY3LDcxLjU0MjI2NzEgNDguOTgzODY2LDY1LjI2MTk5MzkgQzUxLjExMjI4OTksNTUuODQxNTg0MiA0MS42NzcxNzk1LDQ5LjIxMjIyODQgMjUuNjIzOTg0Niw0OS4yMTIyMjg0IEMyNS40ODQ5Mjg5LDQ5LjEyNjg0NDggMjkuODI2MTI5Niw0My4yODM4MjQ4IDM4LjY0NzU4NjksMzEuNjgzMTY4MyBMNzIuODcxMjg3MSwzMi41NTQ0MjUgTDY1LjI4MDk3Myw2Ny42NzYzNDIxIEw1MS4xMTIyODk5LDc3LjM3NjE0NCBMMzUuMDA3NzE4LDgwIFoiIGlkPSJQYXRoLTIyIiBmaWxsPSIjMDAyODY4Ij48L3BhdGg+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMCwzNy43MzA0NDA1IEwyNy4xMTQ1MzcsMC4yNTcxMTE0MzYgQzYyLjM3MTUxMjMsLTEuOTkwNzE3MDEgODAsMTAuNTAwMzkyNyA4MCwzNy43MzA0NDA1IEM4MCw2NC45NjA0ODgyIDY0Ljc3NjUwMzgsNzkuMDUwMzQxNCAzNC4zMjk1MTEzLDgwIEM0Ny4wNTUzNDg5LDc3LjU2NzA4MDggNTMuNDE4MjY3Nyw3MC4zMTM2MTAzIDUzLjQxODI2NzcsNTguMjM5NTg4NSBDNTMuNDE4MjY3Nyw0MC4xMjg1NTU3IDM2LjMwMzk1NDQsMzcuNzMwNDQwNSAyNS4yMjc0MTcsMzcuNzMwNDQwNSBDMTcuODQzMDU4NiwzNy43MzA0NDA1IDkuNDMzOTE5NjYsMzcuNzMwNDQwNSAwLDM3LjczMDQ0MDUgWiIgaWQ9IlBhdGgtMTkiIGZpbGw9IiMzNzkzRUYiPjwvcGF0aD4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+' > </img>
# Created in <span style='font-weight:600;margin-left:4px;'>Deepnote</span></a>
