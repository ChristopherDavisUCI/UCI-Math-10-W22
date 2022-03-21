#!/usr/bin/env python
# coding: utf-8

# # "Strength is Strength"
# 
# Author: Alessandro Tozcano
# 
# Course Project, UC Irvine, Math 10, W22

# ## Introduction
# 
# This notebook takes data recorded over a large period of time from the Open Powerlifting competition and analyzes how performance and other factors, primarily bodyweight,but also including age, are used to score competitors. I attempt to show how a famous statement in the powerlifting and athletics community holds true upon analyzing and visualizing the data.
# 
# ## "Strength is Strength"
# 

# ## Data Cleaning and Preparation
# This section focuses on:
# - Importing the necessary libraries to clean our data
# - Loading and cleaning our DataFrame to meet our needs
# 
# Link to Dataset : https://www.kaggle.com/open-powerlifting/powerlifting-database

# In[1]:


## Import necessary libraries

### Data Visualization and Cleaning
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import altair as alt

### Machine Learning
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

### Extra Component
from mpl_toolkits import mplot3d
import matplotlib.style
from matplotlib.pyplot import figure


# In[ ]:


## Loading the data
df = pd.read_csv('openpowerlifting.csv')


# In[ ]:


## We will split the dataset into Male and Female data frames
### I later decided to focus on the data from Male competitors since that is the community I would fall into
df_male = df[df['Sex'] == 'M' ].copy()
df_female = df[df['Sex']== 'F'].copy()


# In[ ]:


df_male


# In[ ]:


## Create a Sub DataFrame with Bodyweight and Best bench max

### In our data, null values signify that a competitor did not participate in that lift.
### Negative values signify the weight that was attempted but failed
### I want to only analyze data from competitors that were able to complete all 3 lifts, specifically their best performing attempts.

df_male_clean = df_male.loc[:,['BodyweightKg','WeightClassKg','Age','Best3BenchKg','Best3SquatKg','Best3DeadliftKg','TotalKg','IPFPoints','Wilks','McCulloch','Glossbrenner']] # I wanted to include columns that I may decide to analyze
df_male_clean = df_male_clean.dropna().reset_index() # I wanted to reset the indices in this data as well as drop null values
df_male_clean = df_male_clean[df_male_clean['Best3BenchKg'] > 0].copy() # Removes rows with negative values
df_male_clean = df_male_clean[df_male_clean['Best3SquatKg'] > 0].copy() # Removes rows with negative values
df_male_clean = df_male_clean[df_male_clean['Best3DeadliftKg'] > 0].copy() # Removes rows with negative values
df_male_clean = df_male_clean.drop('index',axis = 1) # When I reset the indices 
df_male_clean = df_male_clean.head(5000) # forsake of efficiency and not running out of RAM, I trimmed the values down to the first 5000 since originally our data consisted of close to a million rows.


# In[ ]:


## Here is the finalized clean data in a Pandas DataFrame
df_male_clean


# ## Training and Evaluating of Linear Regression Models
# This Section focuses on:
# - Fitting two Linear Regression models, using data split into training and test sets
# - Analyzing their coefficients, intercepts and score to see how the models fit

# ### Why we train two models: 
# During my research for the project, most of the information found on the web leads one to believe IPFPoints are calculated taking ones total weight lifted and performing some sort of standardization based on ones bodyweight/age or other factors that could contribute to disparities in strength among individuals aside from genetic differences.

# In[ ]:


### We train a Linear Model for IPFPoints with TotalKg and BodyweightKg Respectively
reg_IPFPoints_TotalKg = LinearRegression()
X_a = df_male_clean[['TotalKg']]
y_a = df_male_clean[['IPFPoints']]
X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(X_a,y_a,test_size=0.2)
reg_IPFPoints_TotalKg.fit(X_train_a,y_train_a)

reg_IPFPoints_BodyweightKg = LinearRegression()
X_b = df_male_clean[['BodyweightKg']]
y_b = df_male_clean[['IPFPoints']]
X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(X_b,y_b,test_size=0.2)
reg_IPFPoints_BodyweightKg.fit(X_train_b,y_train_b)


# In[ ]:


### Checking the performance of our Regression Models
## We check the performance, intercept and coefficient of TotalKg vs IPFPoints

IPFPoints_TotalKg_score = reg_IPFPoints_TotalKg.score(reg_IPFPoints_TotalKg.predict(X_test_a),y_test_a)
IPFPoints_TotalKg_intercept = reg_IPFPoints_TotalKg.intercept_
IPFPoints_TotalKg_coeff = reg_IPFPoints_TotalKg.coef_

## We check the performance, intercept and coefficient of BodyweightKg vs IPFPoints

IPFPoints_BodyweightKg_score = reg_IPFPoints_BodyweightKg.score(reg_IPFPoints_TotalKg.predict(X_test_b),y_test_b)
IPFPoints_BodyweightKg_intercept = reg_IPFPoints_BodyweightKg.intercept_
IPFPoints_BodyweightKg_coeff = reg_IPFPoints_BodyweightKg.coef_


# In[ ]:


print("IPFP vs TotalKg Score:" ,IPFPoints_TotalKg_score)
print("IPFP vs TotalKg Intercept:" ,IPFPoints_TotalKg_intercept)
print("IPFP vs TotalKg Coefficient:" ,IPFPoints_TotalKg_coeff)

print("IPFP vs Bodyweight Score:" ,IPFPoints_BodyweightKg_score)
print("IPFP vs Bodyweight Intercept:" ,IPFPoints_BodyweightKg_intercept)
print("IPFP vs Bodyweight Coefficient:" ,IPFPoints_BodyweightKg_coeff)


# ## The analysis and interpretation of our data:
# ### IPF Points vs Total Weight Lifted:
# - IPFP vs TotalKg Score: 0.41103225603882965
# - IPFP vs TotalKg Intercept: 124.81083149
# - IPFP vs TotalKg Coefficient: 0.6824507
# 
# ### IPF Points vs Bodyweight:
# - IPFP vs Bodyweight Score: -0.08219358849842218
# - IPFP vs Bodyweight Intercept: 506.92619584
# - IPFP vs Bodyweight Coefficient: 0.28317706
# 
# ### Off First Glance: 
# We see that neither model fits the data to a great degree. However, using total weight lifted is the superior factor when it comes to IPF Points scored. Going deeper into the numbers we do see positive correlations for both models, with bodyweight providing a higher bias than using total weight lifted as a metric.
# 

# ## Visualization of Our Two Models:
# This section focuses on:
# - Adding Predicted values to the cleaned dataframe
# - Preparing Altair charts to represent true values of both comparisons
# - Preparing Altair charts to display the linear models layered over true values 

# In[ ]:


### Adding the predicted values from our two models to our clean data frame of male data
df_male_clean['Pred_IPFP_TotalKg'] = reg_IPFPoints_TotalKg.predict(df_male_clean[['TotalKg']]).copy()
df_male_clean['PredIPFP_BodyweightKg'] = reg_IPFPoints_BodyweightKg.predict(df_male_clean[['BodyweightKg']]).copy()


# In[ ]:


TotalKg_IPFPoints = alt.Chart(df_male_clean).mark_circle().encode(
    x = alt.X('TotalKg' ,
    scale=alt.Scale(zero=False)),
    y = alt.Y('IPFPoints',
    scale=alt.Scale(zero=False)),
    color = 'WeightClassKg',
    opacity = 'Age'
   
)

Pred_TotalKg_IPFP_Line = alt.Chart(df_male_clean).mark_line(color = 'black').encode(
    x = 'TotalKg',
    y = 'Pred_IPFP_TotalKg',
    
)


Bodyweight_IPFPoints = alt.Chart(df_male_clean).mark_circle().encode(
     x = alt.X('BodyweightKg' ,
    scale=alt.Scale(zero=False)),
    y = alt.Y('IPFPoints',
    scale=alt.Scale(zero=False)),
    color = 'Age'
)

Pred_BodyweightKg_IPFP_Point = alt.Chart(df_male_clean).mark_point(color = 'red',opacity = 0.1).encode(
    x = 'BodyweightKg',
    y = 'Pred_IPFP_TotalKg', 
)


# In[ ]:


### This is a Display of our two linear regression models with their respective charts.
### We can arrive at the following conclusions from this display
### Firstly, we see that TotalKg and IPFPoints retain a strong correlation
### Secondly, we see that our model does a good job of "idealizing" what an ideal system of deriving IPFPoints from bodyweight would provide, nonetheless
### we still arrive at the conclusion that Bodyweight does not have as significant of a weight in assigning a IPFPoint Score as TotalKg lifted.

TotalKg_IPFPoints + Pred_TotalKg_IPFP_Line | Bodyweight_IPFPoints + Pred_BodyweightKg_IPFP_Point


# ## The analysis and interpretation of our data visualization:
# As the titles and axes suggest, the graph representing Total Weight vs IPF Points is depictied on the left and Bodyweight vs IPF Points is depicted on the right. These charts. On the right, our visualization can be interpreted as what IPF scored could be expected from an average candidate at that respective bodyweight. This becomes more clear from our color coding by weight class, we can almost see how members of the same weight class follow a very linear trend between total kilograms lifted and IPF score. Taking this into consideration we can take our linear model as a representation of the trend an average of all weightclasses would follow.
# 
# From a standpoint meant purely to critisize the performance and numerical accuracy of our linear model, both do a poor job. Hollistically our models tell us a lot more about the data. Expectedly, we see that our linear model on the left seems like a simplified interpretation of our data, we can extrapolate that this line represents the average expected performance for an average participant.  This leads one to believe that there are a multitude of other factors outside bodyweight that contribute to an IPF score in this competition.

# ## 3D Visualization
# This section focuses on:
# - Visualizing weight lifted, bodyweight, IPF score on a 3D model

# ### The following figure:
# This figure is essentially a representation of IPFP score based on both total kilograms lifted and the contestants respective bodyweight. Our results on the display confirm how we believe that Total Kilograms lifted has a far greater weight in calculating IPF Score than bodyweight.

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


### Visualization of our true data in 3D
fig = plt.figure()
ax = plt.axes(projection='3d')

# We will draw data from df_male_clean
x = df_male_clean['TotalKg']
y = df_male_clean['BodyweightKg']
z = df_male_clean[['IPFPoints']]
bodyweight_class = df_male_clean['WeightClassKg']
ax.scatter3D(x,y,z,c=z,cmap='viridis')
ax.set_xlabel('TotalKg')
plt.axis([max(x),min(x),min(y),max(y)])
ax.set_ylabel('BodyweightKg')
ax.set_zlabel('IPF Points')
ax.set_title('IPF Points')
ax.grid(False)
ax.view_init(35,-49)

fig.set_size_inches(10,10)


# In[ ]:





# ## Summary
# 
# In this project, a combination of data cleaning, linear regression and both 2-Dimensional and 3-Dimensional data visualization was used. The project narrowed in on an analysis between bodyweight and strength measured by total kilograms lifted across 3 different powerlifts, the bench press, squat and deadlift. This analysis was able to conclude that strength measured by total kilograms lifted was a crucial component in the ranking of strength.

# ## References
# 
# Links I used to help me with this project:
# 
# - https://www.kaggle.com/open-powerlifting/powerlifting-database
# - https://www.geeksforgeeks.org/how-to-change-angle-of-3d-plot-in-python/ 
# - https://stackoverflow.com/questions/17682216/scatter-plot-and-color-mapping-in-python 
# - https://www.geeksforgeeks.org/how-to-reverse-axes-in-matplotlib/#:~:text=In%20Matplotlib%20we%20can%20reverse,methods%20for%20the%20pyplot%20object.
# - https://likegeeks.com/3d-plotting-in-python/#Set_3D_plot_colors_based_on_class 
# - https://jakevdp.github.io/PythonDataScienceHandbook/04.12-three-dimensional-plotting.html

# 

# <a style='text-decoration:none;line-height:16px;display:flex;color:#5B5B62;padding:10px;justify-content:end;' href='https://deepnote.com?utm_source=created-in-deepnote-cell&projectId=0c46f921-7d0c-4a25-8d92-4848e84eb693' target="_blank">
# <img alt='Created in deepnote.com' style='display:inline;max-height:16px;margin:0px;margin-right:7.5px;' src='data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iODBweCIgaGVpZ2h0PSI4MHB4IiB2aWV3Qm94PSIwIDAgODAgODAiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8IS0tIEdlbmVyYXRvcjogU2tldGNoIDU0LjEgKDc2NDkwKSAtIGh0dHBzOi8vc2tldGNoYXBwLmNvbSAtLT4KICAgIDx0aXRsZT5Hcm91cCAzPC90aXRsZT4KICAgIDxkZXNjPkNyZWF0ZWQgd2l0aCBTa2V0Y2guPC9kZXNjPgogICAgPGcgaWQ9IkxhbmRpbmciIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJBcnRib2FyZCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTEyMzUuMDAwMDAwLCAtNzkuMDAwMDAwKSI+CiAgICAgICAgICAgIDxnIGlkPSJHcm91cC0zIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxMjM1LjAwMDAwMCwgNzkuMDAwMDAwKSI+CiAgICAgICAgICAgICAgICA8cG9seWdvbiBpZD0iUGF0aC0yMCIgZmlsbD0iIzAyNjVCNCIgcG9pbnRzPSIyLjM3NjIzNzYyIDgwIDM4LjA0NzY2NjcgODAgNTcuODIxNzgyMiA3My44MDU3NTkyIDU3LjgyMTc4MjIgMzIuNzU5MjczOSAzOS4xNDAyMjc4IDMxLjY4MzE2ODMiPjwvcG9seWdvbj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0zNS4wMDc3MTgsODAgQzQyLjkwNjIwMDcsNzYuNDU0OTM1OCA0Ny41NjQ5MTY3LDcxLjU0MjI2NzEgNDguOTgzODY2LDY1LjI2MTk5MzkgQzUxLjExMjI4OTksNTUuODQxNTg0MiA0MS42NzcxNzk1LDQ5LjIxMjIyODQgMjUuNjIzOTg0Niw0OS4yMTIyMjg0IEMyNS40ODQ5Mjg5LDQ5LjEyNjg0NDggMjkuODI2MTI5Niw0My4yODM4MjQ4IDM4LjY0NzU4NjksMzEuNjgzMTY4MyBMNzIuODcxMjg3MSwzMi41NTQ0MjUgTDY1LjI4MDk3Myw2Ny42NzYzNDIxIEw1MS4xMTIyODk5LDc3LjM3NjE0NCBMMzUuMDA3NzE4LDgwIFoiIGlkPSJQYXRoLTIyIiBmaWxsPSIjMDAyODY4Ij48L3BhdGg+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMCwzNy43MzA0NDA1IEwyNy4xMTQ1MzcsMC4yNTcxMTE0MzYgQzYyLjM3MTUxMjMsLTEuOTkwNzE3MDEgODAsMTAuNTAwMzkyNyA4MCwzNy43MzA0NDA1IEM4MCw2NC45NjA0ODgyIDY0Ljc3NjUwMzgsNzkuMDUwMzQxNCAzNC4zMjk1MTEzLDgwIEM0Ny4wNTUzNDg5LDc3LjU2NzA4MDggNTMuNDE4MjY3Nyw3MC4zMTM2MTAzIDUzLjQxODI2NzcsNTguMjM5NTg4NSBDNTMuNDE4MjY3Nyw0MC4xMjg1NTU3IDM2LjMwMzk1NDQsMzcuNzMwNDQwNSAyNS4yMjc0MTcsMzcuNzMwNDQwNSBDMTcuODQzMDU4NiwzNy43MzA0NDA1IDkuNDMzOTE5NjYsMzcuNzMwNDQwNSAwLDM3LjczMDQ0MDUgWiIgaWQ9IlBhdGgtMTkiIGZpbGw9IiMzNzkzRUYiPjwvcGF0aD4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+' > </img>
# Created in <span style='font-weight:600;margin-left:4px;'>Deepnote</span></a>
