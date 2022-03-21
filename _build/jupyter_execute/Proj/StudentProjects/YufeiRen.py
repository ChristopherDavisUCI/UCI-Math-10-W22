#!/usr/bin/env python
# coding: utf-8

# # Star Type
# 
# Author: Yufei Ren
# 
# Course Project, UC Irvine, Math 10, W22

# ## Introduction
# 
# The dataset "Star dataset to predict star types" consists several features of planets in 6 category: Brown Dwarf, Red Dwarf, White Dwarf, Main Sequence , SuperGiants, HyperGiants, and they are respectivley assigned with numbers 0, 1, 2, 3, 4, 5. 
# >
# In this project, the temperature, radius, 'Absolute magnitude(Mv)', and luminorsity are first used to predict the star type. After that, sklearn is used to find the relationship between temperature, radius and luminorsity.

# ## Main portion of the project
# 
# (You can either have all one section or divide into multiple sections)

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import altair as alt


# In[ ]:


df = pd.read_csv("/work/6 class csv.csv")
df = df.dropna(axis=1) # clear the data
df.head()


# In[ ]:


df.describe()


# In[ ]:


df.columns


# Atair charts is used to visualize the dataset before predicting.

# In[ ]:


brush = alt.selection_interval()
c1 = alt.Chart(df).mark_point().encode(
    x='Absolute magnitude(Mv)',
    y='Radius(R/Ro):Q',
    color='Star type:N'
).add_selection(brush)

c2= alt.Chart(df).mark_bar().encode(
    x = 'Star type:N',
    y='Absolute magnitude(Mv)'
).transform_filter(brush)

c1|c2


# ## Predict the Star type
# Firstly, KNeighborsClassifier is used to predict the star type

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error


# In[ ]:


X = df.iloc[:,:4]
y = df["Star type"]


# Before using using K-Nearest Neighbors Classifier, a scaler is used to scale the input data to avoid errors.

# In[ ]:


scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)


# In[ ]:


clf = KNeighborsClassifier()
clf.fit(X_train, y_train)
loss_train = log_loss(y_train, clf.predict_proba(X_train))
loss_test = log_loss(y_test, clf.predict_proba(X_test))


# In[ ]:


print(f"The log_loss of X_train and y_train is {loss_train:.2f}")
print(f"The log_loss of X_test and y_test is {loss_test:.2f}")


# In[ ]:


df['predict_K'] = clf.predict(X_scaled)


# The logloss of testing data is not large, so there isn't a sign of overfitting

# In[ ]:


(df["Star type"] == df["predict_K"]).value_counts()


# Here we can see that the predicted data is very close to the real data, and there isn't a sign me over-fitting.

# ## Predict the Luminosity

# After using K Neraerst Neighbors to predict the type of a star, I am interested in finding how does radius and temperature are related to the luminorsity of a star. 

# I first try the LinearRegressor

# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
X2 = df[['Radius(R/Ro)','Temperature (K)']]
y2 = df['Luminosity(L/Lo)']
reg1 = LinearRegression()
reg1.fit(X2,y2)
MSE1 = mean_squared_error(y2,reg1.predict(X2))
MAE1 = mean_absolute_error(y2,reg1.predict(X2))
print(f"the coefficients of reg are {reg1.coef_}")
print(f"the intersept of reg is {reg1.intercept_}.")
print(f'The Mean square error is {MSE1:.3f}')
print(f'The Mean absolute error is {MAE1:.3f}')


# The MSE is too high at this case, then I choose to try the KneighborRegressor, and again, the input should be scaled first because they are not in the same unit. 

# In[ ]:


from sklearn.neighbors import KNeighborsRegressor
scaler = StandardScaler()
scaler.fit(X2)
X2_scaled = scaler.transform(X2)
reg2 = KNeighborsRegressor(n_neighbors=4)


# In[ ]:


reg2.fit(X2_scaled, y2)
df['predict_l'] = reg2.predict(X2_scaled)
MSE2 = mean_squared_error(reg2.predict(X2_scaled),y2)
MAE2 = mean_absolute_error(reg2.predict(X2_scaled),y2)
print(f'The Mean square error is {MSE2:.3f}')
print(f'The Mean absolute error is {MAE2:.3f}')


# The number is still large, but smaller than the prediced error in linear regression. The reason for it might be that it is not a linear relationship, but a polynomial relationship.
# 
# To check if it is a polynomial regression, the polynomialfeatures is used. 

# In[ ]:


df3 = df.iloc[:,:3]
df3.columns


# In[ ]:


y_ply = df['Luminosity(L/Lo)']
X_ply = df[['Temperature (K)', 'Radius(R/Ro)']]


# In[ ]:


from sklearn.preprocessing import PolynomialFeatures


# Here I first created a dataframe that contains all posibilities of combination of temperature and radius within 9 degree.

# In[ ]:


poly = PolynomialFeatures(degree=9)
df_ply = pd.DataFrame(poly.fit_transform(X_ply))
df_ply.columns = poly.get_feature_names_out()


# In[ ]:


df_ply


# Then I apply linear regression on luminorsity and each predited polynomial combination, and caculate the error. In the end, I printed out the smallest error and its combination.

# In[ ]:


error_dict = {}
for column in df_ply:
    reg = LinearRegression()
    reg.fit(df_ply[[column]], y_ply)
    error = mean_squared_error(reg.predict(df_ply[[column]]), y_ply)
    error_dict[error] = column
print("the smallest mean squared error is", min(error_dict), 'from column', error_dict[min(error_dict)])


# Here we can see the lowest mean squred error is around 2.3 * 10^10, and the linear combinaiton is Radius^1 * Temperature^0
# 
# The error is very large and a possible reason for that is that all star types are evaluated together and their ranges are in very different scales. As a result, different star types are evaluated separated below.

# In[ ]:


alt.Chart(df).mark_boxplot(extent='min-max').encode(
    x='Star type:N',
    y='Luminosity(L/Lo):Q'
)


# In the plotbox above, it is apparent that the ranges of luminosity of different star types are in very different scale 

# In[ ]:


def find_combination(star_type):
    df_star = df[df['Star type'] == star_type].iloc[:,:3]
    X = df_star[['Temperature (K)', 'Radius(R/Ro)']]
    y = df_star['Luminosity(L/Lo)']
    poly = PolynomialFeatures(degree=9)
    df_ply = pd.DataFrame(poly.fit_transform(X))
    df_ply.columns = poly.get_feature_names_out()
    error_dict = {}
    for column in df_ply:
        reg = LinearRegression()
        reg.fit(df_ply[[column]], y)
        error = mean_squared_error(reg.predict(df_ply[[column]]), y)
        error_dict[error] = column
    print(f"For the star type {star_type}, the smallest error is {min(error_dict)}, which is generagted form {error_dict[min(error_dict)]}")


# In[ ]:


for i in range(5):
    find_combination(i)


# After applying polynomialfeatures to different star type separated, the mean squared error reduced apparently. However, different star type has lowest error with different polynomial combination. As a result, it is not safe to claim any polynomial combination of temperature and radius is the best to predict the Luminosity.  

# ## Summary
# 
# In this project, I am able to predic the star's type by using KneighborClasifier with comparatively high acurracy. However, a best polynomial combination of temperature and radius to predic the luminorsity is not find, because the best structures of different star types differ. As a result, a larger dataset is needed to get a more accurate result.

# ## References
# The dataset “6 class csv.csv” was adapted from [Star dataset to predict star types](https://www.kaggle.com/deepu1109/star-dataset)
# >
# The mthods and application of polynomialfeature was adapted from [sklearn.preprocessing.PolynomialFeatures](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html)
# >
# The idea of polynomialfeature is adapted from [Introduction to Polynomial Regression (with Python Implementation)](https://www.analyticsvidhya.com/blog/2020/03/polynomial-regression-python/)
# >
# The code of drawing altair histogram is adapted from [Simple Histogram](https://altair-viz.github.io/gallery/simple_histogram.html)
# >
# The code of drawing boxplot is adapted from [Boxplot with Min/Max Whiskers](https://altair-viz.github.io/gallery/boxplot.html#)

# <a style='text-decoration:none;line-height:16px;display:flex;color:#5B5B62;padding:10px;justify-content:end;' href='https://deepnote.com?utm_source=created-in-deepnote-cell&projectId=0fb54ba1-bdfd-468e-b41a-ed6482907af2' target="_blank">
# <img alt='Created in deepnote.com' style='display:inline;max-height:16px;margin:0px;margin-right:7.5px;' src='data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iODBweCIgaGVpZ2h0PSI4MHB4IiB2aWV3Qm94PSIwIDAgODAgODAiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8IS0tIEdlbmVyYXRvcjogU2tldGNoIDU0LjEgKDc2NDkwKSAtIGh0dHBzOi8vc2tldGNoYXBwLmNvbSAtLT4KICAgIDx0aXRsZT5Hcm91cCAzPC90aXRsZT4KICAgIDxkZXNjPkNyZWF0ZWQgd2l0aCBTa2V0Y2guPC9kZXNjPgogICAgPGcgaWQ9IkxhbmRpbmciIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJBcnRib2FyZCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTEyMzUuMDAwMDAwLCAtNzkuMDAwMDAwKSI+CiAgICAgICAgICAgIDxnIGlkPSJHcm91cC0zIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxMjM1LjAwMDAwMCwgNzkuMDAwMDAwKSI+CiAgICAgICAgICAgICAgICA8cG9seWdvbiBpZD0iUGF0aC0yMCIgZmlsbD0iIzAyNjVCNCIgcG9pbnRzPSIyLjM3NjIzNzYyIDgwIDM4LjA0NzY2NjcgODAgNTcuODIxNzgyMiA3My44MDU3NTkyIDU3LjgyMTc4MjIgMzIuNzU5MjczOSAzOS4xNDAyMjc4IDMxLjY4MzE2ODMiPjwvcG9seWdvbj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0zNS4wMDc3MTgsODAgQzQyLjkwNjIwMDcsNzYuNDU0OTM1OCA0Ny41NjQ5MTY3LDcxLjU0MjI2NzEgNDguOTgzODY2LDY1LjI2MTk5MzkgQzUxLjExMjI4OTksNTUuODQxNTg0MiA0MS42NzcxNzk1LDQ5LjIxMjIyODQgMjUuNjIzOTg0Niw0OS4yMTIyMjg0IEMyNS40ODQ5Mjg5LDQ5LjEyNjg0NDggMjkuODI2MTI5Niw0My4yODM4MjQ4IDM4LjY0NzU4NjksMzEuNjgzMTY4MyBMNzIuODcxMjg3MSwzMi41NTQ0MjUgTDY1LjI4MDk3Myw2Ny42NzYzNDIxIEw1MS4xMTIyODk5LDc3LjM3NjE0NCBMMzUuMDA3NzE4LDgwIFoiIGlkPSJQYXRoLTIyIiBmaWxsPSIjMDAyODY4Ij48L3BhdGg+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMCwzNy43MzA0NDA1IEwyNy4xMTQ1MzcsMC4yNTcxMTE0MzYgQzYyLjM3MTUxMjMsLTEuOTkwNzE3MDEgODAsMTAuNTAwMzkyNyA4MCwzNy43MzA0NDA1IEM4MCw2NC45NjA0ODgyIDY0Ljc3NjUwMzgsNzkuMDUwMzQxNCAzNC4zMjk1MTEzLDgwIEM0Ny4wNTUzNDg5LDc3LjU2NzA4MDggNTMuNDE4MjY3Nyw3MC4zMTM2MTAzIDUzLjQxODI2NzcsNTguMjM5NTg4NSBDNTMuNDE4MjY3Nyw0MC4xMjg1NTU3IDM2LjMwMzk1NDQsMzcuNzMwNDQwNSAyNS4yMjc0MTcsMzcuNzMwNDQwNSBDMTcuODQzMDU4NiwzNy43MzA0NDA1IDkuNDMzOTE5NjYsMzcuNzMwNDQwNSAwLDM3LjczMDQ0MDUgWiIgaWQ9IlBhdGgtMTkiIGZpbGw9IiMzNzkzRUYiPjwvcGF0aD4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+' > </img>
# Created in <span style='font-weight:600;margin-left:4px;'>Deepnote</span></a>
