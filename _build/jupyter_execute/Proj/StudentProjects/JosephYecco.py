#!/usr/bin/env python
# coding: utf-8

# # Examining Free Throw percentages in the NBA Playoffs
# 
# Author: Joseph Yecco, 61517010
# 
# Course Project, UC Irvine, Math 10, W22

# ## Introduction
# 
# When a player in the NBA is fouled during a shot attempt, and the shot attempt is unsuccessful, the player is awarded 2 or 3 free throws depending on the spot of the foul; they are awarded 1 free throw if the shot is successful. In this project, I aim to examine whether free throw percentage increases or decreases during the playoffs, and to determine if any of the variables in the data available would allow us to examine where this increase/decrease may stem from.

# In[ ]:


import pandas as pd
import numpy as np
import altair as alt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.datasets import fetch_openml
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler


# Below are the two datasets we will use. The first is a dataframe containing records of over 600k free throws attempts between 2006-2016, while the second lists all NBA all-star selections from 2000-2016

# In[ ]:


df0 = pd.read_csv('free_throws.csv')
all_stars = pd.read_csv('allstars.csv')



# # Main Portion

# Now we move on to the main portion of our project. First, lets only include players in our dataframe who have taken over 100 free throw attempts in the playoffs, so that we can have a sufficient sample size to justify their free throw percentage.(The below cell may take a few minutes to run)

# In[ ]:


total_player_list = df0.iloc[:,5].value_counts().index #Lists all of the players in our dataframe
a = np.where([np.count_nonzero((df0["player"] == (total_player_list[i])) & (df0["playoffs"] == "playoffs"))>100 for i in range(len(total_player_list))])
#where function is used to find where in the list are players with over 100 playoff free throw attempts


# In[ ]:


player_list = total_player_list[a] 
df2 = df0[df0['player'].isin(player_list)] #New dataframe only contains info for players with >100 playoff attempts


# In[ ]:


df2


# Below we create the main datatable which we will use for the rest of the project

# In[ ]:


main_df = pd.DataFrame()
main_df["Player"] = pd.Series(player_list)
main_df["Regular Season Attempts"] = pd.Series([np.count_nonzero((df2["player"] == i) & (df2["playoffs"]=='regular')) for i in player_list])
main_df["Regular Season Makes"] = pd.Series([np.count_nonzero((df2["player"] == i) & (df2["shot_made"] == 1) & (df2["playoffs"]=='regular')) for i in player_list])
main_df["Regular Percentage"] = (main_df["Regular Season Makes"]/main_df["Regular Season Attempts"]).round(2)
main_df["Playoff Attempts"] = pd.Series([np.count_nonzero((df2["player"] == i) & (df2["playoffs"]=='playoffs')) for i in player_list])
main_df["Playoff Makes"] = pd.Series([np.count_nonzero((df2["player"] == i) & (df2["shot_made"] == 1) & (df2["playoffs"]=='playoffs')) for i in player_list])
main_df["Playoff Percentage"] = (main_df["Playoff Makes"]/main_df["Playoff Attempts"]).round(2)
main_df["Change"] =  (main_df["Playoff Percentage"]-main_df["Regular Percentage"]).round(2)
main_df["All Star"]= main_df.iloc[:,0].isin(all_stars["Player"])
main_df


# Now let's take a look at the difference between regular season and playoff percentage. Note that the last column ("Change") represents the change from regular season to the playoffs

# In[ ]:


print(f'The average change in free throw percentage between the regular season and playoffs is {100*(main_df["Change"].mean().round(5))}%') #Mean value -.0093


# Now let's look at whether the regular season percentage has a direct impact on that, i.e. does a higher percent yield more stability in the playoffs or not. We will use linear regression to check.

# In[ ]:


reg = LinearRegression()


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(main_df[["Regular Percentage"]],main_df["Change"],test_size=0.2)


# In[ ]:


reg.fit(X_train,y_train)
print(f"The coefficient of our line is {reg.coef_.round(4)} and the intercept is {reg.intercept_.round(4)}")


# These coefficients suggest a slight higher degree of loss in free throw percentage for players who shoot better in the regular season. We want to check how strong the correlation is betwen regular season free throw percentage and the change that occurs in the playoffs.

# In[ ]:


main_df["pred"] = reg.predict(main_df[["Regular Percentage"]])


# In[ ]:


c1 = alt.Chart(main_df).mark_circle().encode(
    x = alt.X("Regular Percentage", scale = alt.Scale(domain=(0.35,1.0))),
    y = 'Change',
    color = "All Star"
)


# In[ ]:


c2 = alt.Chart(main_df).mark_line(color ='red').encode(
    x = alt.X("Regular Percentage", scale = alt.Scale(domain=(0.35,1.0))),
    y = 'pred'
)


# In[ ]:


(c1+c2).properties(
    title = "Change in Percentage by Regular Season Percentage",
    width = 600
)


# From the above graph, it seems that there is not much of a linear relationship between regular season percentage and the change that occurs in the playoffs. Checking the score below confirms that linear regression is not a great predictor for change in playoff percentage.

# In[ ]:


reg.score(X_test,y_test,sample_weight=None)


# Thus while there may be some correlation between regular season percentage and the change that occurs in the playoffs, the correlation is not strong enough to suggest a causal relationship. From here, we will then try to answer 2 follow up questions:
# 
# 1.) Are players who were at some point selected as all-stars(and thus are more likely to have their team depend on them to perform well in the playoffs) more or less consistent with their regular season percentage in the playoffs?
# 
# 2.) Can we accurately predict the change that will occur in playoff percentage based on a player's regular season percentage and all-star status?

# In[ ]:


change_by_starstatus = main_df.groupby('All Star').mean()["Change"] 
change_by_starstatus #Average change from regular season to playoffs for non-all-stars and all-stars


# It appears that on average, non-all-stars face a decrease of approximately 1.02% in their playoff free throw percentage, while all-stars decrease about 0.87%.

# In[ ]:


increase = (100*(abs(change_by_starstatus[1]-change_by_starstatus[0]))/abs(change_by_starstatus[1])).round(2)
print(f"The free throw percentage of All-Stars changes {increase}% less than that of non-All-Stars when going from the regular season to the playoffs")


# From the above, it appears that all-stars are slightly more stable in terms of maintaining their free throw percentage than non-all-stars, with about 18% less variation. Now let's look at the second question. We will use a K-nearest neighbors regressor to try to predict the change. First, we need to scale the input variables so that one is not considered more severely than the other.

# In[ ]:


var_cols = ["Regular Percentage", "All Star"]


# In[ ]:


scaler = StandardScaler()


# In[ ]:


scaler.fit(main_df[var_cols])


# In[ ]:


X_scaled = scaler.transform(main_df[var_cols])


# Below we set up a test, then try to determine the optimal number of neighbors to use to obtain the best possible predictions without overfitting.

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X_scaled, main_df["Change"], test_size = 0.2)


# In[ ]:


def get_scores(k):
    clf = KNeighborsRegressor(n_neighbors=k)
    clf.fit(X_train, y_train)
    train_error = mean_absolute_error(clf.predict(X_train), y_train)
    test_error = mean_absolute_error(clf.predict(X_test), y_test)
    return (train_error, test_error)


# In[ ]:


error_df = pd.DataFrame({"K-inverse":[1/k for k in range(1,20)],"Training Error":[get_scores(k)[0] for k in range(1,20)],"Test Error":[get_scores(k)[1] for k in range(1,20)]})


# In[ ]:


e1 = alt.Chart(error_df).mark_line(color="Blue").encode(
    x = "K-inverse",
    y = "Training Error"
)


# In[ ]:


e2 = alt.Chart(error_df).mark_line(color="Orange").encode(
    x = "K-inverse",
    y = "Test Error"
)


# In[ ]:


(e1+e2).properties(
    title = "Training and test error by number of Neighbors",
    width = 500
)


# Based on the above, we want to avoid lower k values for our classifer as these tend to cause overfitting. This is seen on the right-hand-side of the graph, where a low k-value will lead to a high value for 1/k. Here, we see that the training error(blue line) is low, while the test error(orange line) is high, suggesting overfitting. Thus we will want a k-value of at least 10(where K-inverse=0.1) to avoid this problem and minimize overfitting.

# In[ ]:


def score(n):
    clf2 = KNeighborsRegressor(n_neighbors=n)
    clf2.fit(X_train, y_train)
    print(clf2.score(X_scaled,main_df["Change"]))


# In[ ]:


[score(i) for i in range(1,25)]


# Looking at this list, we see that all of the scores for k>9 are less than .2, indicating that either k-nearest neighbors may not be the best predictor for change in free throw percentages due to the playoffs, or that the combination of independent variables used(Regular Percentage and All-Star Status) are not sufficient to make prediction for our dependent variabel(Change in free throw percentage).

# ## Summary
# 
# The results from this examination of NBA free throw data demonstrated a minor dropoff in free throw percentage during the playoffs. The dropoff was slightly less for All-Star caliber players than for non-all-stars. There did not seem to be any strong linear correlation between regular season free throw percentage and the amount that the player's percentage would drop in the playoffs, although there was a slightly higher dropoff corresponding to players with higher regular season percentages. Finally, we found that k-nearest neighbors could not be successfully used to predict the change that would occur based on whether a player's regular season percentage and their all-star status.
# 
# While the results of this investigation did not yield any major insights, it did reflect the idea that there are many, many factors affecting a player's performance in game. While the datasets we used were thorough, it would be hard to fully account for any particular variable having a major impact on so many players

# ## References
# 
# Free throw Dataset was found on Kaggle at https://www.kaggle.com/sebastianmantey/nba-free-throws/code
# 
# All star dataset was found at https://www.kaggle.com/fmejia21/nba-all-star-game-20002016

# <a style='text-decoration:none;line-height:16px;display:flex;color:#5B5B62;padding:10px;justify-content:end;' href='https://deepnote.com?utm_source=created-in-deepnote-cell&projectId=85b93f6f-d194-4f67-a6f1-36edeb0b9a75' target="_blank">
# <img alt='Created in deepnote.com' style='display:inline;max-height:16px;margin:0px;margin-right:7.5px;' src='data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iODBweCIgaGVpZ2h0PSI4MHB4IiB2aWV3Qm94PSIwIDAgODAgODAiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8IS0tIEdlbmVyYXRvcjogU2tldGNoIDU0LjEgKDc2NDkwKSAtIGh0dHBzOi8vc2tldGNoYXBwLmNvbSAtLT4KICAgIDx0aXRsZT5Hcm91cCAzPC90aXRsZT4KICAgIDxkZXNjPkNyZWF0ZWQgd2l0aCBTa2V0Y2guPC9kZXNjPgogICAgPGcgaWQ9IkxhbmRpbmciIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJBcnRib2FyZCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTEyMzUuMDAwMDAwLCAtNzkuMDAwMDAwKSI+CiAgICAgICAgICAgIDxnIGlkPSJHcm91cC0zIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxMjM1LjAwMDAwMCwgNzkuMDAwMDAwKSI+CiAgICAgICAgICAgICAgICA8cG9seWdvbiBpZD0iUGF0aC0yMCIgZmlsbD0iIzAyNjVCNCIgcG9pbnRzPSIyLjM3NjIzNzYyIDgwIDM4LjA0NzY2NjcgODAgNTcuODIxNzgyMiA3My44MDU3NTkyIDU3LjgyMTc4MjIgMzIuNzU5MjczOSAzOS4xNDAyMjc4IDMxLjY4MzE2ODMiPjwvcG9seWdvbj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0zNS4wMDc3MTgsODAgQzQyLjkwNjIwMDcsNzYuNDU0OTM1OCA0Ny41NjQ5MTY3LDcxLjU0MjI2NzEgNDguOTgzODY2LDY1LjI2MTk5MzkgQzUxLjExMjI4OTksNTUuODQxNTg0MiA0MS42NzcxNzk1LDQ5LjIxMjIyODQgMjUuNjIzOTg0Niw0OS4yMTIyMjg0IEMyNS40ODQ5Mjg5LDQ5LjEyNjg0NDggMjkuODI2MTI5Niw0My4yODM4MjQ4IDM4LjY0NzU4NjksMzEuNjgzMTY4MyBMNzIuODcxMjg3MSwzMi41NTQ0MjUgTDY1LjI4MDk3Myw2Ny42NzYzNDIxIEw1MS4xMTIyODk5LDc3LjM3NjE0NCBMMzUuMDA3NzE4LDgwIFoiIGlkPSJQYXRoLTIyIiBmaWxsPSIjMDAyODY4Ij48L3BhdGg+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMCwzNy43MzA0NDA1IEwyNy4xMTQ1MzcsMC4yNTcxMTE0MzYgQzYyLjM3MTUxMjMsLTEuOTkwNzE3MDEgODAsMTAuNTAwMzkyNyA4MCwzNy43MzA0NDA1IEM4MCw2NC45NjA0ODgyIDY0Ljc3NjUwMzgsNzkuMDUwMzQxNCAzNC4zMjk1MTEzLDgwIEM0Ny4wNTUzNDg5LDc3LjU2NzA4MDggNTMuNDE4MjY3Nyw3MC4zMTM2MTAzIDUzLjQxODI2NzcsNTguMjM5NTg4NSBDNTMuNDE4MjY3Nyw0MC4xMjg1NTU3IDM2LjMwMzk1NDQsMzcuNzMwNDQwNSAyNS4yMjc0MTcsMzcuNzMwNDQwNSBDMTcuODQzMDU4NiwzNy43MzA0NDA1IDkuNDMzOTE5NjYsMzcuNzMwNDQwNSAwLDM3LjczMDQ0MDUgWiIgaWQ9IlBhdGgtMTkiIGZpbGw9IiMzNzkzRUYiPjwvcGF0aD4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+' > </img>
# Created in <span style='font-weight:600;margin-left:4px;'>Deepnote</span></a>
