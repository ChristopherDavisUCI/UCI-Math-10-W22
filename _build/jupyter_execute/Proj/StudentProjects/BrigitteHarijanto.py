#!/usr/bin/env python
# coding: utf-8

# # Titanic Survival
# 
# Author: Brigitte Harijanto
# 
# Email: brigith@uci.edu
# 
# Course Project, UC Irvine, Math 10, W22

# ## Introduction
# 
# In this project, I will be using the titanic dataset, imported from kaggle as train and test, which has never been used in class before. Here, we will explore if the fare a person paid and the cabin they were on can predict their survival rate by using scikit learn.Furthermore, we will explore the reliability of several machine learning models and get the mean of them to check the machine's confidence on this matter. Last, we will prove by graph why Linear Regression is not the way to go.

# ## Main portion of the project

# ### Importing Files

# In[1]:


import seaborn as sns
import numpy as np
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import log_loss, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LinearRegression
from torch import nn


# In[2]:


training = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

training['train_test'] = 1
test['train_test'] = 0
test['Survived'] = np.NaN
df = pd.concat([training,test])


# In[3]:


df.head()


# In[4]:


df.dropna(inplace=True)


# In[5]:


df


# In[6]:


for k in df["Cabin"].unique():
    df[f"Cabin_{k}"] = (df["Cabin"] == k)


# Using `.unique()` to get all distinct decks, then using f strings to modify decks by the respective names and assigning true or false values(numerical values) to each deck column. Afterwards, using for loop to get all the distinct decks to assign all decks to a new numerical column.

# In[7]:


df.head() #check if each Cabin has it's own column


# In[8]:


#def order(char):
    #return ord(char) - ord('A') +1


# In[9]:


#df['deck_num'] = df['deck'].map(lambda s : order(s))


# In[10]:


X_colnames = ["Fare"] + [f"Cabin_{k}" for k in sorted(df["Cabin"].unique())]
y_colname = 'Survived'
X = df.loc[:, X_colnames].copy()
y = df.loc[:, y_colname].copy()


# In[11]:


[f"Cabin_{k}" for k in sorted(df["Cabin"].unique())]


# ### Using Sci-kit learn's StandardScaler, KNeighborsClassifier, Train_test_split

# In[12]:


scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)


# In[13]:


clf = KNeighborsClassifier(n_neighbors=10)


# In[14]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
len(X_train)


# In[15]:


len(y_train)


# In[16]:


scaler.fit(X_train)
scaler.fit(X_test)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[17]:


len(X_train_scaled)


# In[18]:


clf.fit(X_train, y_train)


# In[19]:


clf.predict_proba(X_scaled)


# In[20]:


clf.score(X_test, y_test, sample_weight = None)


# In[21]:


clf.score(X_train, y_train, sample_weight = None)


# The score on training set is better than the test set, hence overfitting. However since the difference is not extreme, the overfitting is under control.

# In[22]:


log_loss(df['Survived'], clf.predict_proba(X_scaled), labels = clf.classes_)


# The loss is very close to 0(less than 1), meaning the machine can make better predictions about the survival rate.

# In[23]:


def get_scores(k):
    reg = KNeighborsClassifier(n_neighbors=k)
    reg.fit(X_train, y_train)
    test_error = log_loss(y_test, reg.predict_proba(X_test), labels = reg.classes_)
    return (test_error)


# In[24]:


for i in range(1,11):
    print(f"when n_nearest neighbor is {i}, the test error is {get_scores(i)}")


# When I ran the code, K = 10 nearest neighbors shows the best result since the error is the lowest for test set.  

# In[25]:


c1 = alt.Chart(df).mark_circle().encode(
    x = "Cabin:O",
    y = "Fare",
    color = "Survived:N",
    tooltip = ["Fare", "Cabin", "Survived"]
).properties(
    title = "Fare paid for each deck",
    height = 550,
    width = 800
)

c1


# Here we can see that there is an outlier in the graph, now lets get rid of it to see if it makes the data more reliable. 

# In[26]:


df[df["Fare"] > 500] #check which rows has datas with fare > $500


# In[27]:


df1 = df[~(df["Fare"] > 500)] #Removing the anomaly from the dataset and setting it to df1


# In[28]:


c2 = alt.Chart(df1).mark_circle().encode(
    x = "Cabin:O",
    y = "Fare",
    color = "Survived:N",
    tooltip = ["Fare", "Cabin", "Survived"]
).properties(
    title = "Fare paid for each deck",
    height = 550,
    width = 800
)

c2


# After removing the 2 outliers, we can see from the graph that it visually looks more evenly distributed. As there are more orange (survived) dots than blue (did not survive) dots in the upper half of the graph, we can tell that the people who paid more had a higher rate of survival. Now we can recalculate our losses, errors and scores to see if there is any changes after outlier is removed.

# In[29]:


X_colnames = ["Fare"] + [f"Cabin_{k}" for k in sorted(df["Cabin"].unique())]
y_colname = 'Survived'
X1 = df1.loc[:, X_colnames].copy()
y1 = df1.loc[:, y_colname].copy()


# In[30]:


#extra 
Cabin_col = [f"Cabin_{k}" for k in sorted(df["Cabin"].unique())]


# In[31]:


clf1 = KNeighborsClassifier(n_neighbors=10)


# In[32]:


X_train1, X_test1, y_train1, y_test1 = train_test_split(X1,y1,test_size=0.2)
len(X_train1)


# In[33]:


scaler = StandardScaler()
scaler.fit(X_train1)
scaler.fit(X_test1)
X_train_scaled1 = scaler.transform(X_train1)
X_test_scaled1 = scaler.transform(X_test1)


# In[34]:


scaler.fit(X1)
X_scaled1 = scaler.transform(X1)


# The difference between `X_scaled` and `X_scaled1` is the length of it. `X_scaled1` has fewer rows because anomalies has been dropped.

# In[35]:


clf1.fit(X_train1, y_train1)


# In[36]:


clf1.predict_proba(X_scaled1)


# In[37]:


clf1.score(X_test1, y_test1, sample_weight = None)


# In[38]:


clf1.score(X_train1, y_train1, sample_weight = None)


# In[39]:


log_loss(df1['Survived'], clf1.predict_proba(X_scaled1), labels = clf.classes_)


# When I run my notebook, the results after removing the anomaly shows higher score and lower log loss in `df1`. Therefore, dataset `df`, where we do not remove, the anomaly shows a worse result. We will proceed the following code with `df1` and not `df`. But for both parts, scores on test and training set does not have a significant difference of a few times higher than the other. In fact, score on test set is higher than on training set, making this underfitting but underfitting is in control since there is not much difference.

# ### Using KNeighborsRegressor

# In[40]:


kreg = KNeighborsRegressor(n_neighbors=10)


# In[41]:


kreg.fit(X_train1, y_train1)


# In[42]:


kreg.predict(X_scaled1)


# In[43]:


#since KNeighborsRegressor does not have a predict_proba attribute, we use .predict
kreg.score(X_test1, y_test1, sample_weight = None)


# In[44]:


kreg.score(X_train1, y_train1, sample_weight = None)


# Since the score is much better on training set than test set, this is a very likely sign of overfitting.

# In[45]:


mean_squared_error(df1['Survived'], kreg.predict(X_scaled1))


# Through this comparison, we can see that using `KNeighborsClassifier` is better than using `KneighborsRegressor` since the difference between test set score and training set score is much closer to each other in Classifier than Regressor. Moreover, we always want score to be higher. The higher, the better. Here we can see that the Regressor has really low scores on both sets while the Classifier has much higher scores for both sets. Hence it is more suitable to use `KNeighborsClassifier`.

# ### Extra topics not learned in class

# In the following blocks, I will use `Linear Regression`, `KNearestNeighbors`, and `Random Forest` to compute scores of how well the machine can predict the survival rate according to the fare and deck the passengers are at, by using `Cross Validation`.

# In[46]:


lr = LogisticRegression(max_iter = 2000)
cv = cross_val_score(lr,X_train_scaled1,y_train1,cv=5)
print(cv)
print(cv.mean())


# Here we use `max_iter` to set the maximum number of iterations a solver can do.
# 
# 

# In[47]:


knn = KNeighborsClassifier()
cv = cross_val_score(knn,X_train1,y_train1,cv=5)
print(cv)
print(cv.mean())


# In[48]:


rf = RandomForestClassifier(random_state = 1)
cv = cross_val_score(rf,X_train1,y_train1,cv=5)
print(cv)
print(cv.mean())


# Here we use cv = 5 which is printed in the square brackets, meaning it takes the features df and target y, splits into k-folds (which is the cv parameter), fits on the (k-1) folds and evaluates on the last fold. 

# Now we want to use the `voting classifier`. The `voting classifier` basically takes the reliability / score of each machine learning models and take the mean of it, to see how reliable the machine is overall. If the mean of these machines are > 50%, then the passenger is predicted to have survived, vice versa.
# 

# In[49]:


voting_clf = VotingClassifier(estimators = [('lr',lr),('knn',knn),('rf',rf)], voting = 'soft') 


# Here we use soft voting because we want the classifier to classify data based on probability and weights instead of class labels and weights.

# In[50]:


cv = cross_val_score(voting_clf,X_train_scaled1,y_train1,cv=5)
print(cv)
print(cv.mean())


# Here we can see that through the cross validation score, the machine is 65% confident with it's data, as a mean from all the machine learning models teste, which are `Logistic Regression`, which has the highest score, followed by `KNearestNeighbor` and last, `Random Forest Classifier`. Now lets try test the data with `Logistic Regression` and `Linear Regression`.

# ### Using Logistic Regression

# In[51]:


clfo = LogisticRegression()


# In[52]:


clfo.fit(X_train1, y_train1)


# In[53]:


clfo.predict(X_train1)


# In[54]:


np.count_nonzero(clfo.predict(X_test1) == y_test1)/len(X_test1)


# In[55]:


np.count_nonzero(clfo.predict(X_train1) == y_train1)/len(X_train1)


# Here we can see that the test set using Logistic Regression is 62% accurate, and training set using Logistic Regression is 74% accurate.

# ### Using LinearRegression

# In[56]:


reg = LinearRegression()


# In[57]:


reg.fit(df[["Fare",]], df["Survived"])


# In[58]:


reg.coef_


# In[59]:


reg.intercept_


# In[60]:


def draw_line(m,b):
    alt.data_transformers.disable_max_rows()

    d1 = alt.Chart(df1).mark_circle().encode(
        x = "Fare",
        y = "Survived"
    )

    xmax = 40
    df_line = pd.DataFrame({"Fare":[0,xmax],"Survived":[b,xmax*m]})
    d2 = alt.Chart(df_line).mark_line(color="red").encode(
        x = "Fare",
        y = "Survived"
    )
    return d1+d2


# Although this is a `Logistic Regression` Problem, we will try using `Linear Regression`.

# In[61]:


draw_line(0.00082767, 0.6070082823070905)


# Here it is obvious why we do not use Linear Regression since `Survived` is more categorical, either yes or no instead of having some in between because people can't half-survive. The graph is very hard to read but since the intercept is above 0.5, we can tell that more people survived than not when paying 0, but other than that it is very difficult to tell as the line goes down although there is no negative coefficient. However, we would think that the more we pay, the more likely we are to survive, so this just furtheer shows how linear regression does not work.
# 
# Now let's see if we use Cabins as the x-axis it will work at all.

# In[62]:


reg1 = LinearRegression()


# In[63]:


reg1.fit(df[[f"Cabin_{k}" for k in sorted(df["Cabin"].unique())]], df["Survived"])


# In[64]:


reg.coef_


# In[65]:


reg.intercept_


# In[66]:


def draw_line(m,b):
    alt.data_transformers.disable_max_rows()

    d1 = alt.Chart(df).mark_circle().encode(
        x = "Cabin:O",
        y = "Survived"
    )

    xmax = 40
    df_line = pd.DataFrame({"Cabin":[0,xmax],"Survived":[b,xmax*m]})
    d2 = alt.Chart(df_line).mark_line(color="red").encode(
        x = "Cabin",
        y = "Survived"
    )
    return d1+d2


# In[67]:


draw_line(1.1276363e+14, -112763629749553.84)


# Seeing how the graph does not ake any sense in `Cabin`, `Linear Regression` will not work in this case. Since we multiply with the x value, but this is categorical so that won't work.

# ## Summary
# 
# In this project, I used sci-kit learn to do the machine learning process. First, I used `KNeighborClassifier` and `Regressor` to compare with each other. Looking at the scores of training and test set, It is obvious that `KneightborClassifier` is the better one to use since eventhough it is underfitting, the difference is not that significant compared to the completely overfitting `Regressor`. Next, I used `cross validation score` and `voting classifier` to get the mean of how confident the machine is in predicting the survival of passengers using machine learning models: `KNearestNeighbors`, `LinearRegression`, `Random Forest`. This shows that the machine is most confident when using the `Linear Regression` model. The mean is about 62% when I run the code so I would say that the machine is still on the better side in predicting the survival although not exactly reliable. Following that, I used `Logistic Regression` and `Linear Regression`. For `Logistic Regression`, I knew it would work because it is suitable for classification problems, so I just checked the reliability of the training set in comparison to the test set. Turns out the training set shows 74% while test shows 62% accuracy making the model overfitting but the overfitting is still under control. When using Linear Regression, I had doubts whether or not it can be used at all so I graphed it to see if it makes sense but it does not. Overall, I wouldn't recommend using the machine to predict survival rates from fare and cabin.

# ## References
# Dataset:https://www.kaggle.com/c/titanic/data
# 
# Exploration: https://www.youtube.com/watch?v=I3FBJdiExcg
# 
# Reference for new material: https://www.kaggle.com/kenjee/titanic-project-example
# 
# Understanding of cv parameter: https://stackoverflow.com/questions/52611498/need-help-understanding-cross-val-score-in-sklearn-python
# 
# KNeighborsRegressor: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html#sklearn.neighbors.KNeighborsRegressor
# 
# Referenced from Homework 6, Week 6 Video Notebook, and Week9 Monday Linear Regression Worksheet from Math10.
# 

# <a style='text-decoration:none;line-height:16px;display:flex;color:#5B5B62;padding:10px;justify-content:end;' href='https://deepnote.com?utm_source=created-in-deepnote-cell&projectId=65ef8688-7f91-4fc2-8385-2d7d30df3935' target="_blank">
# <img alt='Created in deepnote.com' style='display:inline;max-height:16px;margin:0px;margin-right:7.5px;' src='data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iODBweCIgaGVpZ2h0PSI4MHB4IiB2aWV3Qm94PSIwIDAgODAgODAiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8IS0tIEdlbmVyYXRvcjogU2tldGNoIDU0LjEgKDc2NDkwKSAtIGh0dHBzOi8vc2tldGNoYXBwLmNvbSAtLT4KICAgIDx0aXRsZT5Hcm91cCAzPC90aXRsZT4KICAgIDxkZXNjPkNyZWF0ZWQgd2l0aCBTa2V0Y2guPC9kZXNjPgogICAgPGcgaWQ9IkxhbmRpbmciIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJBcnRib2FyZCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTEyMzUuMDAwMDAwLCAtNzkuMDAwMDAwKSI+CiAgICAgICAgICAgIDxnIGlkPSJHcm91cC0zIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxMjM1LjAwMDAwMCwgNzkuMDAwMDAwKSI+CiAgICAgICAgICAgICAgICA8cG9seWdvbiBpZD0iUGF0aC0yMCIgZmlsbD0iIzAyNjVCNCIgcG9pbnRzPSIyLjM3NjIzNzYyIDgwIDM4LjA0NzY2NjcgODAgNTcuODIxNzgyMiA3My44MDU3NTkyIDU3LjgyMTc4MjIgMzIuNzU5MjczOSAzOS4xNDAyMjc4IDMxLjY4MzE2ODMiPjwvcG9seWdvbj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0zNS4wMDc3MTgsODAgQzQyLjkwNjIwMDcsNzYuNDU0OTM1OCA0Ny41NjQ5MTY3LDcxLjU0MjI2NzEgNDguOTgzODY2LDY1LjI2MTk5MzkgQzUxLjExMjI4OTksNTUuODQxNTg0MiA0MS42NzcxNzk1LDQ5LjIxMjIyODQgMjUuNjIzOTg0Niw0OS4yMTIyMjg0IEMyNS40ODQ5Mjg5LDQ5LjEyNjg0NDggMjkuODI2MTI5Niw0My4yODM4MjQ4IDM4LjY0NzU4NjksMzEuNjgzMTY4MyBMNzIuODcxMjg3MSwzMi41NTQ0MjUgTDY1LjI4MDk3Myw2Ny42NzYzNDIxIEw1MS4xMTIyODk5LDc3LjM3NjE0NCBMMzUuMDA3NzE4LDgwIFoiIGlkPSJQYXRoLTIyIiBmaWxsPSIjMDAyODY4Ij48L3BhdGg+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMCwzNy43MzA0NDA1IEwyNy4xMTQ1MzcsMC4yNTcxMTE0MzYgQzYyLjM3MTUxMjMsLTEuOTkwNzE3MDEgODAsMTAuNTAwMzkyNyA4MCwzNy43MzA0NDA1IEM4MCw2NC45NjA0ODgyIDY0Ljc3NjUwMzgsNzkuMDUwMzQxNCAzNC4zMjk1MTEzLDgwIEM0Ny4wNTUzNDg5LDc3LjU2NzA4MDggNTMuNDE4MjY3Nyw3MC4zMTM2MTAzIDUzLjQxODI2NzcsNTguMjM5NTg4NSBDNTMuNDE4MjY3Nyw0MC4xMjg1NTU3IDM2LjMwMzk1NDQsMzcuNzMwNDQwNSAyNS4yMjc0MTcsMzcuNzMwNDQwNSBDMTcuODQzMDU4NiwzNy43MzA0NDA1IDkuNDMzOTE5NjYsMzcuNzMwNDQwNSAwLDM3LjczMDQ0MDUgWiIgaWQ9IlBhdGgtMTkiIGZpbGw9IiMzNzkzRUYiPjwvcGF0aD4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+' > </img>
# Created in <span style='font-weight:600;margin-left:4px;'>Deepnote</span></a>
