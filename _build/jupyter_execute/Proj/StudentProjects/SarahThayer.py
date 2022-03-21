#!/usr/bin/env python
# coding: utf-8

# # Predict Position of Player with Seasonal Stats
# Author: <b>Sarah Thayer</b>
# 
# Course Project, UC Irvine, Math 10, W22<br>

# ## Introduction
# 
# My project is exploring two Kaggle Datasets of NBA stats to predict their listed position. 
# Two datasets are needed. One contains the players and their listed positions. The second contains season stats, height, and weight. We. merge the two for a complete dataset.

# ## Main portion of the project
# 
# (You can either have all one section or divide into multiple sections)

# In[ ]:


import numpy as np
import pandas as pd
import altair as alt


# ### Load First Dataset 
# 
# NBA Stats [found here](https://www.kaggle.com/mcamli/nba17-18?select=nba.csv). 
# The columns we are interested in extracting are `Player`, `Team`, `Pos`, `Age`.
# 
# Display shape of the dataframe to confirm we have enough data points.

# In[ ]:


df_position = pd.read_csv("nba.csv", sep = ',' ,encoding = 'latin-1')

df_position = df_position.loc[:,df_position.columns.isin(['Player', 'Tm', 'Pos', 'Age']) ]
df_position.head()


# In[ ]:


df_position.shape


# ### Clean First Dataset
# Rename column `TM` to `Team`.
# 
# NBA Players have unique player id in `Player` column. Remove player ID to view names. ( i.e."Alex Abrines\abrinal01" remove the unique player id after the "\\")
# 
# NBA players that have been traded mid-season appear twice. Drop `Player` duplicates from the dataframe and check shape. 
# 
#   

# In[ ]:


df_position = df_position.rename(columns={'Tm' : 'Team'}) 

df_position['Player'] = df_position['Player'].map(lambda x: x.split('\\')[0])

df_pos_unique = df_position[~df_position.duplicated(subset=['Player'])]
df_pos_unique.shape

df_pos_unique.head()


# ### Load Second Dataset
# 
# NBA stats [found here](https://www.kaggle.com/jacobbaruch/basketball-players-stats-per-season-49-leagues).
# 
# Large dataset of 20 years of stats form 49 differrent leagues. Parse dataframe for the the relevant data in the NBA league during the 2017 - 2018 regular season. Then our new dataframe contains height and weight.

# In[ ]:


df = pd.read_csv("players_stats_by_season_full_details.csv",  encoding='latin-1' ) 

df= df[(df["League"] == 'NBA') &  (df["Season"] == '2017 - 2018') & (df["Stage"] == 'Regular_Season')] 

df_hw = df.loc[:,~df.columns.isin(['Rk', 'League', 'Season', 'Stage', 'birth_month', 'birth_date', 'height', 'weight',
                                  'nationality', 'high_school', 'draft_round', 'draft_pick', 'draft_team'])]

df_hw.shape


# ### Clean Second Dataset
# Drop duplicates from dataframe with `Player` from the dataframe containing height and weight.

# In[ ]:


df_hw_unique = df_hw[~df_hw.duplicated(subset=['Player'])]
df_hw_unique.shape


# ### Prepare Merged Data
# - Merge First and Second Dataset 
# - Encode the NBA listed positions 
# 
# 
# Confirm it's the same player by matching name and team.

# In[ ]:


df_merged = pd.merge(df_pos_unique,df_hw_unique, on = ['Player','Team'])


# ### One-Hot Encoding
# Encode the positions (`strings`) into numbers (`ints`).

# In[ ]:


enc = {'PG' : 1, 'SG' : 2,  'SF': 3, 'PF': 4, 'C':5}
df_merged["Pos_enc"] = df_merged["Pos"].map(enc)
df_merged


# ### Find Best Model
# 
# Feature Selection: Data has player name, team, position, height, weight, and 20+ seasonal stats. Not all  features are relevant to predicting NBA Position. Test with different varations of features. Iterate through `combinations()` of k columns in `cols`. Combinations and estimating the counts of the training trials can be found here: ["...the number of k-element subsets (or k-combinations) of an n-element set"](https://en.wikipedia.org/wiki/Binomial_coefficient).
# 
# KNN Optimization: On each combination of possible training features, iterate through range of ints for possible `n_neighbors`.
# 
# Log Results: Create results dictionary to store `features`, `n_neighbors`, `log_loss`, and `classifier` for each training iteration. Iterate through results dictionary to find smallest `log_loss` along with features used, `n_neighbors` used, and the classifier object.
# 
# ```
# results  = {
#     trial_num: {
#         'features': [],
#         'n_neighbors':,
#         'log_loss':
#         'classifier':
#     }
# }
# ```
# 
# 

# In[ ]:


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import log_loss

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import warnings
warnings.filterwarnings('ignore')

from itertools import combinations
cols = ['Pos_enc','FGM', 'FGA', '3PM','3PA','FTM', 'FTA','TOV','PF','ORB','DRB',
        'REB','AST','STL', "BLK",'PTS','height_cm','weight_kg'
]

trial_num = 0 # count of training attempts
loss_min = False # found log_loss minimum
n_search = True # searching for ideal n neighbors 
results = {} # dictionary of results per training attempt

found_clf = False
for count in range(12, 18): 
    print(f"Testing: {len(cols)} Choose {count}")
    
    for tup in combinations(cols,count):  # iterate through combination of columns
       
        for i in range(3,6): # iterate through options of n neighbors
            if n_search:
                X = df_merged[list(tup)]
                y = df_merged["Pos"]
                scaler = StandardScaler()
                scaler.fit(X)
                X_scaled = scaler.transform(X)

                X_scaled_train, X_scaled_test, y_train, y_test = train_test_split(X,y,test_size=0.2)            

                clf = KNeighborsClassifier(n_neighbors=i)
                clf.fit(X_scaled_train,y_train)
                X["pred"] = clf.predict(X_scaled)

                probs = clf.predict_proba(X_scaled_test)
                loss = log_loss( y_true=y_test ,y_pred=probs , labels= clf.classes_)

                results[trial_num] = {
                    'features': list(tup) ,
                    'n_neighbors': i,
                    'log_loss': loss,
                    'classifier': clf
                }

                trial_num+=1

                if loss < .7:
                    n_search = False
                    loss_min = True
                    found_clf = True
                    print(f"Found ideal n_neighbors")
                    break
        
        if (n_search == False) or (loss<.6): 
            loss_min = True
            print('Found combination of features')
            break
            
    if loss_min:
        print('Return classifier')
        break

if not found_clf:
    print(f"Couldn't find accurate classifier. Continue to find best results.")


# ### Return Best Results
# 
# Find the training iteration with the best `log_loss`. 
# 
# Return the classifier and print the features selected, neighbors used, and corresponding `log_loss`.

# In[ ]:


min_log_loss = results[0]['log_loss']
for key in results:
    # key = trial number
    iter_features = results[key]['features']
    iter_n_neighbors = results[key]['n_neighbors']
    iter_log_loss = results[key]['log_loss']
    
    if iter_log_loss < min_log_loss:
        min_log_loss = iter_log_loss
        min_key=key

print(f"Total Attempts: {len(results)}")
print(f"Best log_loss: {results[min_key]['log_loss']}")
print(f"Best features: {results[min_key]['features']}")
print(f"Number of features: {len(results[min_key]['features'])}")
print(f"Ideal n_neighbors: {results[min_key]['n_neighbors']}")
print(f"Best classifier: {results[min_key]['classifier']}")


# Predict position of NBA players on entire dataset.
# 
# Access best classifier in `results` dict by `min_key`.

# In[ ]:


X = df_merged[results[min_key]['features']]
scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)

clf = results[min_key]['classifier']

df_merged['Preds'] = clf.predict(X)


# ### Vizualize Results
# 
# Display all the Centers. Our predicted values show good results at identifying Centers.
# 
# Look at a true Point Gaurd. Chris Paul is a good example of a Point Gaurd.
# 
# Look at Lebron James. In 2018, for Clevland, Kaggle has him listed has a Power Foward. 

# In[ ]:


df_merged[df_merged['Pos_enc']==5]


# In[ ]:


df_merged[df_merged['Player']=='Chris Paul']


# In[ ]:


df_merged[df_merged['Player']=='LeBron James']


# ### Evaluate Performance
# Evalute the performance of classifier using the `log_loss` metric.

# In[ ]:


X_scaled_train, X_scaled_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

probs = clf.predict_proba(X_scaled_test)

log_loss(y_true=y_test ,y_pred=probs , labels= clf.classes_)
loss = log_loss(y_true=y_test ,y_pred=probs , labels= clf.classes_)
loss


# 1st row mostly Pink some Light Blue. 
# We're good at determining Point Guards, sometimes Small Forwards trigger false positive.
# 
# 5th row mostly Blue with some Yellow.
# Makes sense. Great at determining Centers. Sometimes Power Forwards trigger false positive. 

# In[ ]:


chart = alt.Chart(df_merged).mark_circle().encode(
    x=alt.X('height_cm:O', axis=alt.Axis(title='Height in cm')),
    y=alt.X('Pos_enc:O', axis=alt.Axis(title='Encoded')),
    color = alt.Color("Preds", title = "Positions"),
).properties(
    title = f"Predicted NBA Positions",
)
chart


# ## Summary
# Taking player seasonal stats, height, and weight we attempted to predict NBA positions by classification. Some NBA positions are easier to predict than others. 

# ## References
# 
# Include references that you found helpful.  Also say where you found the dataset you used.

# **Dataframes used from Kaggle**
# 
# 
# Basketball Players Stats per Season - 49 Leagues [found here](https://www.kaggle.com/jacobbaruch/basketball-players-stats-per-season-49-leagues).
# 
# NBA Player Stats 2017-2018 [found here](https://www.kaggle.com/mcamli/nba17-18?select=nba.csv). 
# 

# <a style='text-decoration:none;line-height:16px;display:flex;color:#5B5B62;padding:10px;justify-content:end;' href='https://deepnote.com?utm_source=created-in-deepnote-cell&projectId=da008a44-10fb-41c0-95d8-1e6b51ff39cf' target="_blank">
# <img alt='Created in deepnote.com' style='display:inline;max-height:16px;margin:0px;margin-right:7.5px;' src='data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iODBweCIgaGVpZ2h0PSI4MHB4IiB2aWV3Qm94PSIwIDAgODAgODAiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8IS0tIEdlbmVyYXRvcjogU2tldGNoIDU0LjEgKDc2NDkwKSAtIGh0dHBzOi8vc2tldGNoYXBwLmNvbSAtLT4KICAgIDx0aXRsZT5Hcm91cCAzPC90aXRsZT4KICAgIDxkZXNjPkNyZWF0ZWQgd2l0aCBTa2V0Y2guPC9kZXNjPgogICAgPGcgaWQ9IkxhbmRpbmciIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJBcnRib2FyZCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTEyMzUuMDAwMDAwLCAtNzkuMDAwMDAwKSI+CiAgICAgICAgICAgIDxnIGlkPSJHcm91cC0zIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxMjM1LjAwMDAwMCwgNzkuMDAwMDAwKSI+CiAgICAgICAgICAgICAgICA8cG9seWdvbiBpZD0iUGF0aC0yMCIgZmlsbD0iIzAyNjVCNCIgcG9pbnRzPSIyLjM3NjIzNzYyIDgwIDM4LjA0NzY2NjcgODAgNTcuODIxNzgyMiA3My44MDU3NTkyIDU3LjgyMTc4MjIgMzIuNzU5MjczOSAzOS4xNDAyMjc4IDMxLjY4MzE2ODMiPjwvcG9seWdvbj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0zNS4wMDc3MTgsODAgQzQyLjkwNjIwMDcsNzYuNDU0OTM1OCA0Ny41NjQ5MTY3LDcxLjU0MjI2NzEgNDguOTgzODY2LDY1LjI2MTk5MzkgQzUxLjExMjI4OTksNTUuODQxNTg0MiA0MS42NzcxNzk1LDQ5LjIxMjIyODQgMjUuNjIzOTg0Niw0OS4yMTIyMjg0IEMyNS40ODQ5Mjg5LDQ5LjEyNjg0NDggMjkuODI2MTI5Niw0My4yODM4MjQ4IDM4LjY0NzU4NjksMzEuNjgzMTY4MyBMNzIuODcxMjg3MSwzMi41NTQ0MjUgTDY1LjI4MDk3Myw2Ny42NzYzNDIxIEw1MS4xMTIyODk5LDc3LjM3NjE0NCBMMzUuMDA3NzE4LDgwIFoiIGlkPSJQYXRoLTIyIiBmaWxsPSIjMDAyODY4Ij48L3BhdGg+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMCwzNy43MzA0NDA1IEwyNy4xMTQ1MzcsMC4yNTcxMTE0MzYgQzYyLjM3MTUxMjMsLTEuOTkwNzE3MDEgODAsMTAuNTAwMzkyNyA4MCwzNy43MzA0NDA1IEM4MCw2NC45NjA0ODgyIDY0Ljc3NjUwMzgsNzkuMDUwMzQxNCAzNC4zMjk1MTEzLDgwIEM0Ny4wNTUzNDg5LDc3LjU2NzA4MDggNTMuNDE4MjY3Nyw3MC4zMTM2MTAzIDUzLjQxODI2NzcsNTguMjM5NTg4NSBDNTMuNDE4MjY3Nyw0MC4xMjg1NTU3IDM2LjMwMzk1NDQsMzcuNzMwNDQwNSAyNS4yMjc0MTcsMzcuNzMwNDQwNSBDMTcuODQzMDU4NiwzNy43MzA0NDA1IDkuNDMzOTE5NjYsMzcuNzMwNDQwNSAwLDM3LjczMDQ0MDUgWiIgaWQ9IlBhdGgtMTkiIGZpbGw9IiMzNzkzRUYiPjwvcGF0aD4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+' > </img>
# Created in <span style='font-weight:600;margin-left:4px;'>Deepnote</span></a>
