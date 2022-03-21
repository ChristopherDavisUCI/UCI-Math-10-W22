#!/usr/bin/env python
# coding: utf-8

# # Netflix Stock Price Prediction
# 
# Author: Jiayu Wang
# 
# Student ID: 74613921
# 
# Course Project, UC Irvine, Math 10, W22

# ## Introduction
# 
# Stock market prediction is trying to determine the future value of a company stock. This project will utilize historic data with linear regression to help us predict the future stock values. 
# 
# 

# ## Section 1 Clean Dataset
# 

# In[ ]:


import pandas as pd
import altair as alt 


# In[ ]:


nfstocks = pd.read_csv("/work/netflix_stock.csv")
nfstocks.dropna(inplace=True)


# In[ ]:


nfstocks


# We would only need data "Date" and "Adj Close." 

# In[ ]:


nfstocks['Date'] = pd.to_datetime(nfstocks['Date'])


# Here we want to use adjusted closing price, which is "Adj Close." The adjusted closing price amends a stock's closing price to reflect that stock's value after accounting for any corporate actions. Closing price is the raw price which is just the cash value of last transcted price before market closes. 
# 
# Using adjusted closing prices since these fully incorporate any splits, dividens, spin-offs and other distributions made by trader. 

# In[ ]:


nfstocks1 = nfstocks[['Date','Adj Close']]


# Before we start utilize linear regression predict the future trend, we want to see whats the trend for stock prices in the past 20 years. 

# In[ ]:


import altair as alt
from altair import Chart, X, Y
import numpy as np


# In[ ]:


nearest = alt.selection(type='single', nearest=True, on='mouseover',
fields=["Date"], empty='none')

# The basic line
line = alt.Chart().mark_line(interpolate='basis').encode(
    alt.X('Date:T', axis=alt.Axis(title='')),
    alt.Y('Adj Close:Q', axis=alt.Axis(title='',format='$f')),
    color='symbol:N'
).properties(title = "Stock Price")

selectors = alt.Chart().mark_point().encode(
    x="Date:T",
    opacity=alt.value(0),
).add_selection(
    nearest
)

# Draw points 
points = line.mark_point().encode(
    opacity=alt.condition(nearest, alt.value(1), alt.value(0))
)

text = line.mark_text(align='left', dx=5, dy=-5).encode(
    text=alt.condition(nearest, 'Adj Close:Q', alt.value(' '))
)

# Draw a rule
rules = alt.Chart().mark_rule(color="gray").encode(
    x="Date:T",
).transform_filter(
    nearest
)


stockChart = alt.layer(selectors, line, points, rules, text,data=nfstocks1).add_selection(nearest)


# In[ ]:


stockChart


# This graph is reflecting the price changes over the span of 20 years which is from 1.156 dollars to 537.00 dollars . We can see a clear trend that the prices is continuing to increase.
# 
# Now we are going to utilize Linear Regression to help us predict the future price trend based on historic data. 

# ## Section 2: Expoential Moving Average (EMA)

# We know that in predicting stock prices utilize techinical analysis is crucial. 
# 
# * Techinical analysis indicator determines the support and resistance levels. This help indicate whether the prices has dropped lower or climbed higher. 
# 
# * Techinical indicators are heuristic or pattern-based signals produced by the price, volume, and/or open interest of a security or contract used by traders who follow techinical analysis. 
# 
# Based on some research online, we will add exponential moving average to our existing data set. 
# 
# * Expoential moving average is a type of moving average that places a greater weight and significance on the most recent data points. 

# In[ ]:


import pandas as pd 

nfstocks1['EMA'] = nfstocks1['Adj Close'].ewm(span=20, min_periods=0,adjust=False,ignore_na=False).mean()


# Short term traders usually rely on 12 to 26 day EMA. Especially that EMA reacts more quickly to price swings than the SMA, but it will lag quite a bit over longer periods. 
# 
# In order to test the accuracy of the EMA calculation, we searched that for 2021.06.03, on MarketWatch website shows that EMA for Netflix is 499.77 USD, which matches our calculations. 

# In[ ]:


c1 = Chart(nfstocks1).mark_line().encode(
    x = "Date",
    y = "Adj Close",
    color = "symbol:N"
).properties(title= "Stock Price")


# In[ ]:


c2 = Chart(nfstocks1).mark_line(color ="red").encode(
    x = "Date",
    y = "EMA"
)


# In[ ]:


c1 + c2


# Now we are going to develop our regression model and see how effective the EMA is at predicting the price of the stock. 
# 
# We are first going to use the 80/20 split to train and test our data

# ## Section 3: Linear Regression

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(nfstocks1[['Adj Close']],nfstocks1[['EMA']], test_size = .2)


# In[ ]:


#test set
print(X_test.describe())


# In[ ]:


#training set
print(X_train.describe())


# Training Model

# In[ ]:


from sklearn.linear_model import LinearRegression 

reg = LinearRegression()


# In[ ]:


reg.fit(X_train, y_train)


# In[ ]:


y_pred = reg.predict(X_test)


# Now we are going to use mean absolute error and coefficient of determination to examine how well this model fits and examine the coefficient. 

# In[ ]:


from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score


# In[ ]:


print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("Coefficient of Determination:", r2_score(y_test, y_pred))


# We know that Mean Absolute Error can be described as the sum of the absolute error for all observed values divided by the total number of observations. Therefore, the lower MAE we get, the better. 
# 
# For the coefficient of determination(R_squared), we know that it has of valuyes of 1 or 0 will indicate the regression line represents all or none of the data. Therefore, we would want our coefficient is higher (closer to 1.0) since it helps to indicate that it is a better fit for the observation. 
# 
# Based on the ideas and output above, we know that our regression from the MAE and R-squared perspectives that they are a good fit. 
# 
# Now we want to utilize the graph to show the observed value and predicted values 

# In[ ]:


df1 = pd.DataFrame(nfstocks1["Adj Close"].iloc[500:510])


# In[ ]:


df1["Date"] = pd.DataFrame(nfstocks1['Date'].iloc[500:510])


# In[ ]:



c3 = alt.Chart(df1).mark_circle().encode(
    x = alt.X("Date",scale=alt.Scale(zero=False)),
    y = alt.Y("Adj Close",scale=alt.Scale (zero=False))
)
    


# In[ ]:


nfstocks1["pred"] = reg.predict(nfstocks1[["Adj Close"]])


# In[ ]:


df1["prediction"] = pd.DataFrame(nfstocks1['pred'].iloc[500:510])


# In[ ]:


c4 = alt.Chart(df1).mark_line(color="red").encode(
    x = alt.X("Date",scale=alt.Scale(zero=False)),
    y = alt.Y("prediction",scale=alt.Scale (zero=False))
)


# In[ ]:


c3+c4


# ## Section 4: Simulation to Test the Model

# Since we already developed and trained the model based on historic pricing data. Now we want to develop the model that can use EMA of any given days to repdict the close price. 
# 
# We also want to use a trading strategy such as if our predicted value of the stock is higher than the open value of the stock, we will consider to trade. However, if our predicted stock price is equal to or smaller than open value of the stock, we will consider not trade. 
# 
# Input some data that we already have to test: (Here I choose data that are from consecutive days since the prediction of current stock price is based on the EMA value from the day before)

# In[ ]:


df2 = pd.DataFrame(nfstocks["Date"].iloc[4900:4944])
df2['Open'] = nfstocks['Open'].iloc[4900:4944]
df2['Adj Close'] = nfstocks['Adj Close'].iloc[4900:4944]
df2['EMA'] = nfstocks1['EMA'].iloc[4900:4944]


# Predicted Value

# In[ ]:


df2['predict'] = reg.predict(df2[["EMA"]].values)


# In[ ]:


#Here we use the conditions method, link is below reference
conditions = [df2['predict'] > df2['Open'],df2['predict'] < df2['Open']]
choices = ['Trade','Not Trade']


# In[ ]:


df2['Trade Decision'] = np.select(conditions, choices, default='Not Trade')


# Now we want to create a seperate column with the potential loss/earning that we can make

# In[ ]:


df2['Earning'] = df2["predict"]-df2["Open"]


# Before providing a direct view of potential earnings, you can utilize the function below to get the predicted value of the day. 
# 
# You can follows the steps below:

# In[ ]:


#First, you can put your open value of the day, here we take 598.179993 
#as an example 
open = 598.179993	


# In[ ]:


#Second, run this block, you will get to know what is the predicted close price will be based on your open price
close = df2['Adj Close'].where(df2['Open'] == open).dropna().values[0]
print(f'If you have an open price as ${open}, your predicted close price will be ${close}')


# In[ ]:


import matplotlib as mpl 


# In order to give a more direct view of potential earnings, we highlighted the earning based on its value. If we are able

# In[ ]:


def style_negative(v, props='color:red;'):
    return props if v < 0 else None


# In[ ]:


df2 = df2.style.applymap(style_negative,subset=["Earning"])


# In[ ]:


def highlight_max(s, props = ''):
    return np.where(s == np.nanmax(s.values), props, '')


# In[ ]:


df2.apply(highlight_max, props='color:white;background-color:darkblue')


# Based on the graph above, we are able to see the highlights of the highest value of Netflix stock price in the selected 44 days. We also highlighted the trade areas to show you that the days that are reccomendated to trade with poential earnings. 

# ## Summary
# 
# In this project, we first cleaned and organized the data set. With the help of the graph, we are able to see the trend of the stock performances from 2002 to 2021. Later, we added another factor EMA, Expoential Moving Average, to our dataset and utilize the linear regression to train the model and help us predict the future stock prices. In the end, we utilize some data in the original dataset and ran simulations to test the effecitveness and application of this model. 

# ## References
# * Dataset: Kaggle Dataset
# 
# * EMA: https://stackoverflow.com/questions/48775841/pandas-ema-not-matching-the-stocks-ema
# 
# * Conditions: https://www.statology.org/compare-two-columns-in-pandas/
# 
# * altair chart: https://altair-viz.github.io/gallery/multiline_tooltip.html

# <a style='text-decoration:none;line-height:16px;display:flex;color:#5B5B62;padding:10px;justify-content:end;' href='https://deepnote.com?utm_source=created-in-deepnote-cell&projectId=9242c623-f563-433b-8768-a466c3aa94bb' target="_blank">
# <img alt='Created in deepnote.com' style='display:inline;max-height:16px;margin:0px;margin-right:7.5px;' src='data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iODBweCIgaGVpZ2h0PSI4MHB4IiB2aWV3Qm94PSIwIDAgODAgODAiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8IS0tIEdlbmVyYXRvcjogU2tldGNoIDU0LjEgKDc2NDkwKSAtIGh0dHBzOi8vc2tldGNoYXBwLmNvbSAtLT4KICAgIDx0aXRsZT5Hcm91cCAzPC90aXRsZT4KICAgIDxkZXNjPkNyZWF0ZWQgd2l0aCBTa2V0Y2guPC9kZXNjPgogICAgPGcgaWQ9IkxhbmRpbmciIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJBcnRib2FyZCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTEyMzUuMDAwMDAwLCAtNzkuMDAwMDAwKSI+CiAgICAgICAgICAgIDxnIGlkPSJHcm91cC0zIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxMjM1LjAwMDAwMCwgNzkuMDAwMDAwKSI+CiAgICAgICAgICAgICAgICA8cG9seWdvbiBpZD0iUGF0aC0yMCIgZmlsbD0iIzAyNjVCNCIgcG9pbnRzPSIyLjM3NjIzNzYyIDgwIDM4LjA0NzY2NjcgODAgNTcuODIxNzgyMiA3My44MDU3NTkyIDU3LjgyMTc4MjIgMzIuNzU5MjczOSAzOS4xNDAyMjc4IDMxLjY4MzE2ODMiPjwvcG9seWdvbj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0zNS4wMDc3MTgsODAgQzQyLjkwNjIwMDcsNzYuNDU0OTM1OCA0Ny41NjQ5MTY3LDcxLjU0MjI2NzEgNDguOTgzODY2LDY1LjI2MTk5MzkgQzUxLjExMjI4OTksNTUuODQxNTg0MiA0MS42NzcxNzk1LDQ5LjIxMjIyODQgMjUuNjIzOTg0Niw0OS4yMTIyMjg0IEMyNS40ODQ5Mjg5LDQ5LjEyNjg0NDggMjkuODI2MTI5Niw0My4yODM4MjQ4IDM4LjY0NzU4NjksMzEuNjgzMTY4MyBMNzIuODcxMjg3MSwzMi41NTQ0MjUgTDY1LjI4MDk3Myw2Ny42NzYzNDIxIEw1MS4xMTIyODk5LDc3LjM3NjE0NCBMMzUuMDA3NzE4LDgwIFoiIGlkPSJQYXRoLTIyIiBmaWxsPSIjMDAyODY4Ij48L3BhdGg+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMCwzNy43MzA0NDA1IEwyNy4xMTQ1MzcsMC4yNTcxMTE0MzYgQzYyLjM3MTUxMjMsLTEuOTkwNzE3MDEgODAsMTAuNTAwMzkyNyA4MCwzNy43MzA0NDA1IEM4MCw2NC45NjA0ODgyIDY0Ljc3NjUwMzgsNzkuMDUwMzQxNCAzNC4zMjk1MTEzLDgwIEM0Ny4wNTUzNDg5LDc3LjU2NzA4MDggNTMuNDE4MjY3Nyw3MC4zMTM2MTAzIDUzLjQxODI2NzcsNTguMjM5NTg4NSBDNTMuNDE4MjY3Nyw0MC4xMjg1NTU3IDM2LjMwMzk1NDQsMzcuNzMwNDQwNSAyNS4yMjc0MTcsMzcuNzMwNDQwNSBDMTcuODQzMDU4NiwzNy43MzA0NDA1IDkuNDMzOTE5NjYsMzcuNzMwNDQwNSAwLDM3LjczMDQ0MDUgWiIgaWQ9IlBhdGgtMTkiIGZpbGw9IiMzNzkzRUYiPjwvcGF0aD4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+' > </img>
# Created in <span style='font-weight:600;margin-left:4px;'>Deepnote</span></a>
