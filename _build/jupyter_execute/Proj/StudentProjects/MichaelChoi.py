#!/usr/bin/env python
# coding: utf-8

# # Credit Card Approval Prediction
# 
# Author: Michael Choi
# 
# Course Project, UC Irvine, Math 10, W22

# ## Introduction
# 
# I will be analyzing the following dataset that lists the following information on credit card apporval ratings. I want to see what professions, ages, education statues get approved and at what odds that they do get approved.

# ## Main portion of the project

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import altair as alt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss


# After importing libraries, we will load two datasets. The first dataset called App is a dataset of demographics of credit users such as their age, profession, income, etc. The next dataset is Cred, a dataset with credit records and payment history.

# In[ ]:


app= pd.read_csv('application_record.csv',header=None)
cred = pd.read_csv('credit_record.csv', header=None)


# In[ ]:


cred.head()


# Notice how the dataset has the column names we want as row 0. We will fix this by first renaming the columns and then deleting the first row. Finally, we will reindex the dataset with df.reset_index(). Note that these methods are apart of the pandas library.
# 
# More information about these methods can be found in the references section.

# In[ ]:


app = app.rename(columns = app.iloc[0,:]).drop([0]).reset_index(drop = True)
cred = cred.rename(columns = cred.iloc[0,:]).drop([0]).reset_index(drop = True)


# In[ ]:


cred.head()


# Since there is no column that states if they got approved or not, I made my own based on who had a balance outstanding on their credit record. Months are listed as negative because it represents how many months ago (0 is current month, -1 is last month, etc.)
# 
# X means they didnt have a loan for the month, C means they paid it off. Everything else means they had a balance which I will record in the "Balance" column I am creating.

# In[ ]:


cred['Balance'] = cred['STATUS'].map(lambda x: 'N' if (x == 'C') or (x == 'X') else 'Y')


# I noticed that, using pd.series.value_counts(), some IDs of the two datasets did not match i.e there were some ID numbers in the app dataset not in the cred dataset and vice versa. 
# 
# Since I needed every applicant's demographics to create a prediction and their credit record to determine if they are approved or not, I used pd.df.isin() to only use IDs present in both datasets.

# In[ ]:


cred['ID'].value_counts()


# In[ ]:


app['ID'].value_counts()


# In[ ]:


app = app[app["ID"].isin(cred["ID"].value_counts()[:].index)]
cred = cred[cred["ID"].isin(app["ID"].value_counts()[:].index)]


# For a bit of more dataset cleaning up, I used the panda method pd.to_numeric() for the IDs because I wanted to sort both datasets by increasing ID numbers with pd. Reset indices again.

# In[ ]:


app['ID'] = pd.to_numeric(app['ID'])
cred['ID'] = pd.to_numeric(cred['ID'])
app = app.sort_values(by = ['ID']).reset_index(drop = True)
cred = cred.sort_values(by = ['ID']).reset_index(drop = True)


# In[ ]:


app.keys()


# For later on in the dataset, we will change more columns to numeric. In addition, for birth days, I decided to divide by 365 to get years instead of days old each applicant is. I multipled by -1 since the column is listed as negatives such as the months column as described earlier. Same for the 'DAYS_EMPLOYED' column.

# In[ ]:


app['AMT_INCOME_TOTAL'] = pd.to_numeric(app['AMT_INCOME_TOTAL'])
app['DAYS_BIRTH'] = -pd.to_numeric(app['DAYS_BIRTH'])/365
app['DAYS_EMPLOYED'] = -pd.to_numeric(app['DAYS_EMPLOYED'])/365


# This part was the trickest and longest part of my project process. Notice below how there are multiple rows of the same ID, one for each month and each month can have a different values in the 'Balance' column.

# In[ ]:


cred


# I needed to find a way to count how many 'Y' each ID row had. This will tell me how many months each applicant, identified by the ID number, had a balance on the credit record; that is, they paid late. 
# 
# After trying many for loops, list comprehensions, creating new datasets, I decided to use the groupby method we learned in week 10 of class. This wil also be a part of my "Extra" section.

# In[ ]:


a = cred.groupby("ID")['Balance'].value_counts().sort_index(ascending=True)
print(a)


# At first, I tried using a for x,y loop as used in class; however, I had issues with breaking out of the loop and the datatype of y was a multiindex which was confusing to call from. 
# Instead, I created a list for ID, Labels (Y or N), and the values themselves (months with or without a balance)

# In[ ]:


ids = a.keys().get_level_values('ID')
values = a.values
labels = [(a.keys()[i][1] == 'Y') for i in range(len(a))]


# I created a new dataframe with a breakdown values grouped by ID known as Breakdown. I created a new dataframe called Denies. This is a dataset with rows that only carry a balance so I can the number of months each ID carried a credit card balance.

# In[ ]:


Breakdown = pd.DataFrame({'ID': ids, 'Labels':labels, 'Months': values})
Breakdown = Breakdown.sort_values(by = ['ID'])
Denies = Breakdown[Breakdown["Labels"] == True].reset_index(drop = True)
Denies.head()


# In[ ]:


print(f"The number of months with an outstanding balance for the first ID is {Breakdown['Months'][0]}")


# Now we will create a new column in Denies that will deny applicants if they have more than 8 months were they carried a balance. We will add all these IDs to a list called denies.

# In[ ]:


Denies['Approval']= Breakdown['Months'].map(lambda x: 'N' if x > 8 else 'Y')
denies = list(Denies[Denies['Approval'] == 'N']['ID'].values)


# From here, we will create a new column in the app dataset where applicants with IDs in the denies list are denied and the rest are given an approval status for now.

# In[ ]:


app['Approval'] = app['ID'].map(lambda x: 'N' if denies.count(x) else 'Y')


# We will also deny applicants with an income less than 40000

# In[ ]:


for i in range(len(app)):
    if ((app['AMT_INCOME_TOTAL'][i] < 40000)):
        app['Approval'][i] = 'N'


# In[ ]:


app


# Now, onto Classification and graphing. We will use the following columns to predict if applicants get approved or not.

# In[ ]:


X = app[['AMT_INCOME_TOTAL','DAYS_EMPLOYED','DAYS_BIRTH']]
y = app['Approval']


# We will scale the data

# In[ ]:


scalar = StandardScaler()
scalar.fit(X)
X_scaled = scalar.transform(X)


# I want to determine which value of k for KNeighborsClassifier() will give me the lowest log_loss value. This is original code. Start by making a function loss(k).

# In[ ]:


def loss(k):
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(X_scaled,y)
    return log_loss(y,clf.predict_proba(X_scaled))


# Create a pandas dataset with two columns: one with a k-value and the other with the associated log_loss value from the function we created above. Will take a while to load, 30 to 50 seconds. Afterwards, graph this dataset.

# In[ ]:


list(range(1,50))
log = pd.DataFrame({'k-value':list(range(1,100))})
log['loss'] = [loss(x) for x in log['k-value']]
log


# In[ ]:


alt.Chart(log).mark_circle().encode(
    x = 'k-value',
    y = 'loss',
    tooltip = ['k-value','loss']
)


# Close examination of the chart and graph with tooltip will show k = 10 gives the lowest log_loss value

# In[ ]:


clf = KNeighborsClassifier(n_neighbors=10)
clf.fit(X_scaled,y)


# It is important to split data into a training set and test set. This allow us to train the model and then have values to test. We can also look for signs of overfitting if test score is much lower than training score. This occurs when the model follows the training set too closely.

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X_scaled,y, test_size = .3)
(clf.score(X_train,y_train),clf.score(X_test,y_test))


# Since the training and test set scores are roughly the same, we do not have signs of overfitting

# In[ ]:


log_loss(y,clf.predict_proba(X_scaled))


# Now we will graph the dataset. However, we can see the range contains lots of outliers. We will create a new dataset with incomes less than 600000 only called graph

# In[ ]:


alt.data_transformers.disable_max_rows()
alt.Chart(app).mark_circle().encode(
    alt.X('DAYS_BIRTH', scale=alt.Scale(zero=False)),
    alt.Y('AMT_INCOME_TOTAL',scale=alt.Scale(zero=False)),
    color = 'Approval',
    tooltip = ['DAYS_BIRTH','AMT_INCOME_TOTAL']
).properties(
    title = "Age vs Income"
)


# In[ ]:


graph = app[app['AMT_INCOME_TOTAL'] < 600000]


# In[ ]:


alt.data_transformers.disable_max_rows()
c1 = alt.Chart(graph).mark_circle().encode(
    alt.X('DAYS_BIRTH',scale=alt.Scale(zero= False)),
    alt.Y('AMT_INCOME_TOTAL',scale=alt.Scale(domain=(0,600000))),
    color = 'Approval',
    tooltip = ['DAYS_BIRTH','AMT_INCOME_TOTAL']
).properties(
    title = "Age vs Income"
)
c1


# We will make a second Graph with predicted Approval to compare

# In[ ]:


app["pred"] = clf.predict(X_scaled)
graph = app[app['AMT_INCOME_TOTAL'] < 600000]


# In[ ]:


app


# In[ ]:


alt.data_transformers.disable_max_rows()
c2 = alt.Chart(graph).mark_circle().encode(
    alt.X('DAYS_BIRTH',scale=alt.Scale(zero= False)),
    alt.Y('AMT_INCOME_TOTAL',scale=alt.Scale(domain=(0,600000))),
    color = 'pred',
    tooltip = ['DAYS_BIRTH','AMT_INCOME_TOTAL']
).properties(
    title = "Age vs Income"
)
c1


# In[ ]:


c2


# For fun, here is an additional graph with more interactive elements such as selection and a bar chart. I got some of this code from altair-viz.github.io which will be listed in the refrences section.

# In[ ]:


interval = alt.selection_interval()
alt.data_transformers.disable_max_rows()
c3 = alt.Chart(graph).mark_circle().encode(
    alt.X('DAYS_BIRTH',scale=alt.Scale(zero=False)),
    alt.Y('AMT_INCOME_TOTAL',scale=alt.Scale(domain=(0,600000))),
    color=alt.condition(interval, 'Approval', alt.value('lightgray')),
    tooltip = ['AMT_INCOME_TOTAL','DAYS_BIRTH']
).properties(
    title = "Income vs Work Experience"
).add_selection(
    interval
)

c4 = alt.Chart(graph).mark_bar().encode(
    x ='Approval',
    y = alt.Y('count()'),
    color='Approval'
).transform_filter(
    interval
)


# In[ ]:


c3|c4


# ## Summary
# 
# We were able to create a KNeighborsClassifier() model to predict whether or not an applicant would get approved based on their net income and age. 
# 
# The most difficult and time consuming part of this project was the beginning where I had to deal with two different datasets conected only by an ID number. There were many mismatches in dataset lengths and outside methods I had to use. 
# 
# Overall, I really enjoyed this project. I love credit cards and it was fun determining which applicants got approved based on critera that I came up with. 

# ## References
# 
# Include references that you found helpful.  Also say where you found the dataset you used.

# Dataset Origin
# https://www.kaggle.com/ginaerian/credit-card-approval-prediction-gina
# 
# Delete row from dataframe with pandas method
# https://www.shanelynn.ie/pandas-drop-delete-dataframe-rows-columns/#:~:text=to%20be%20removed.-,Deleting%20rows%20using%20%E2%80%9Cdrop%E2%80%9D%20(best,for%20small%20numbers%20of%20rows)&text=To%20delete%20rows%20from%20a%20DataFrame%2C%20the%20drop%20function%20references,index%20when%20you%20run%20%E2%80%9Cdata.
# 
# Reset Index of dataframe with pandas method
# https://www.machinelearningplus.com/pandas/pandas-reset-index/#:~:text=To%20reset%20the%20index%20in,()%20with%20the%20dataframe%20object.&text=On%20applying%20the%20.,dataframe%20as%20a%20separate%20column.
# 
# Sort dataset by specific row with pandas method
# https://pandas.pydata.org/docs/reference/api/pandas.Series.sort_values.html
# 
# Altair Interactivity Graph: https://altair-viz.github.io/altair-tutorial/notebooks/06-Selections.html

# <a style='text-decoration:none;line-height:16px;display:flex;color:#5B5B62;padding:10px;justify-content:end;' href='https://deepnote.com?utm_source=created-in-deepnote-cell&projectId=5c15bbd8-4087-42d4-b2c7-c3609f26763d' target="_blank">
# <img alt='Created in deepnote.com' style='display:inline;max-height:16px;margin:0px;margin-right:7.5px;' src='data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iODBweCIgaGVpZ2h0PSI4MHB4IiB2aWV3Qm94PSIwIDAgODAgODAiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8IS0tIEdlbmVyYXRvcjogU2tldGNoIDU0LjEgKDc2NDkwKSAtIGh0dHBzOi8vc2tldGNoYXBwLmNvbSAtLT4KICAgIDx0aXRsZT5Hcm91cCAzPC90aXRsZT4KICAgIDxkZXNjPkNyZWF0ZWQgd2l0aCBTa2V0Y2guPC9kZXNjPgogICAgPGcgaWQ9IkxhbmRpbmciIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJBcnRib2FyZCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTEyMzUuMDAwMDAwLCAtNzkuMDAwMDAwKSI+CiAgICAgICAgICAgIDxnIGlkPSJHcm91cC0zIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxMjM1LjAwMDAwMCwgNzkuMDAwMDAwKSI+CiAgICAgICAgICAgICAgICA8cG9seWdvbiBpZD0iUGF0aC0yMCIgZmlsbD0iIzAyNjVCNCIgcG9pbnRzPSIyLjM3NjIzNzYyIDgwIDM4LjA0NzY2NjcgODAgNTcuODIxNzgyMiA3My44MDU3NTkyIDU3LjgyMTc4MjIgMzIuNzU5MjczOSAzOS4xNDAyMjc4IDMxLjY4MzE2ODMiPjwvcG9seWdvbj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0zNS4wMDc3MTgsODAgQzQyLjkwNjIwMDcsNzYuNDU0OTM1OCA0Ny41NjQ5MTY3LDcxLjU0MjI2NzEgNDguOTgzODY2LDY1LjI2MTk5MzkgQzUxLjExMjI4OTksNTUuODQxNTg0MiA0MS42NzcxNzk1LDQ5LjIxMjIyODQgMjUuNjIzOTg0Niw0OS4yMTIyMjg0IEMyNS40ODQ5Mjg5LDQ5LjEyNjg0NDggMjkuODI2MTI5Niw0My4yODM4MjQ4IDM4LjY0NzU4NjksMzEuNjgzMTY4MyBMNzIuODcxMjg3MSwzMi41NTQ0MjUgTDY1LjI4MDk3Myw2Ny42NzYzNDIxIEw1MS4xMTIyODk5LDc3LjM3NjE0NCBMMzUuMDA3NzE4LDgwIFoiIGlkPSJQYXRoLTIyIiBmaWxsPSIjMDAyODY4Ij48L3BhdGg+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMCwzNy43MzA0NDA1IEwyNy4xMTQ1MzcsMC4yNTcxMTE0MzYgQzYyLjM3MTUxMjMsLTEuOTkwNzE3MDEgODAsMTAuNTAwMzkyNyA4MCwzNy43MzA0NDA1IEM4MCw2NC45NjA0ODgyIDY0Ljc3NjUwMzgsNzkuMDUwMzQxNCAzNC4zMjk1MTEzLDgwIEM0Ny4wNTUzNDg5LDc3LjU2NzA4MDggNTMuNDE4MjY3Nyw3MC4zMTM2MTAzIDUzLjQxODI2NzcsNTguMjM5NTg4NSBDNTMuNDE4MjY3Nyw0MC4xMjg1NTU3IDM2LjMwMzk1NDQsMzcuNzMwNDQwNSAyNS4yMjc0MTcsMzcuNzMwNDQwNSBDMTcuODQzMDU4NiwzNy43MzA0NDA1IDkuNDMzOTE5NjYsMzcuNzMwNDQwNSAwLDM3LjczMDQ0MDUgWiIgaWQ9IlBhdGgtMTkiIGZpbGw9IiMzNzkzRUYiPjwvcGF0aD4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+' > </img>
# Created in <span style='font-weight:600;margin-left:4px;'>Deepnote</span></a>
