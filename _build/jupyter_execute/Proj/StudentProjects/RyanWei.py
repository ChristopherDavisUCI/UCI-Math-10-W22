#!/usr/bin/env python
# coding: utf-8

# # Predicting match outcomes in professional Couter Strike: Global Offensive
# 
# Author: Ryan Wei
# 
# Course Project, UC Irvine, Math 10, W22

# ## Introduction

# Counter Strike: Global Offensive (CSGO) is a tactical 5v5 first person shooter video game with a thriving esports scene. A series is played out in a Best of 3 series, where teams take turns picking and banning maps from a predetermined pool of maps. Maps have a T and CT side, with teams switching halfway through a match. The first team to 16 points wins that map and takes a point in the Best of 3 series. 
# 
# The goal of this project is to see I can predict the outcome of matches based on a number of variables: the global rank of a team, their preference for the maps picked and their strength on that map. 

# ## Main portion of the project

# In[ ]:


import pandas as pd
import numpy as np
import altair as alt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import log_loss


# In[ ]:


get_ipython().system('kaggle datasets download -d viniciusromanosilva/csgo-hltv --unzip')


# In[ ]:


df = pd.read_csv("cs_hltv_data.csv")


# In[ ]:


df


# I realized that the data in score_second_team_t2_M1 (through M3) are all truncated for some reason, so instead of showing a proper score range it is limited to 0-9. This data will need to either be reverse engineered.

# In[ ]:


#df for the outcomes of respective best of 3 matches
match_df = df.iloc[:,3:37].copy()


# In[ ]:


match_df


# In[ ]:


#build new df consisting of just pure game data, isolating individual matches 
map_data_cols = ["T1_name", "T2_name", "Map", "T1_score", "T2_score", "T1_side", "T2_side", "T1_score_H1", "T1_score_H2", "T2_score_H1", "T2_score_H2"]

#pick out wanted columns, then rename
map_data = match_df.iloc[:, np.r_[0:2, 7:10, 16:22]].copy()
map_data.columns = map_data_cols


# In[ ]:


#employ np.r_, and splice the desired colums. Append to existing matrix map_data after matching up column names
map_data_temp = match_df.iloc[:, np.r_[0:2, 10:13, 22:28]].copy()
map_data_temp.columns = map_data_cols
map_data = map_data.append(map_data_temp, ignore_index = True)

map_data_temp2 = match_df.iloc[:, np.r_[0:2, 13:16, 28:34]].copy()
map_data_temp2.columns = map_data_cols
map_data = map_data.append(map_data_temp, ignore_index = True)


# In[ ]:


#T1 = first team, T2 = second team, H1 = first half, H2 = second half
map_data


# In[ ]:


#convert a column to str, so entries with "-" can be fitered out
    #these are Map 3, since matches are Best of 2 sometimes the third map isn't played
map_data = map_data.convert_dtypes()
map_data.drop(map_data[map_data["T1_score"] == "-"].index, inplace = True)


# In[ ]:


#alter T2_score_H2 data based on whether match went into overtime
#if match went into overtime, then alter based on T2_score (want to eliminate extra rounds from overtime)
for x in map_data.index:
    if (map_data["T1_score"][x] + map_data["T2_score"][x] > 30):
        if (map_data["T2_score"][x] > 16):
            map_data.at[x, "T2_score_H2"] = 15 - map_data["T2_score_H1"][x]
        else:
            map_data.at[x, "T2_score_H2"] = map_data["T2_score"][x] - map_data["T2_score_H1"][x]


# In[ ]:


map_data


# After fixing that column of data, we can proceed to calculating whether each map is T or CT sided
# 

# In[ ]:


map_names = map_data.Map.unique()
map_side_data = {"Map": map_names,
    "CT_wins": 0,
    "T_wins": 0
}
map_side = pd.DataFrame(map_side_data, columns = ["Map", "CT_wins", "T_wins"])


# In[ ]:


#fill out map_side df with raw wins/losses
CT = "CT_wins"
T = "T_wins"
for x in map_data.index:
    m_name = map_data["Map"][x]
    if (map_data["T1_side"][x] == "CT"):
        map_side.loc[map_side.Map == m_name, f"{CT}"] += map_data["T1_score_H1"][x] + map_data["T2_score_H2"][x]
        map_side.loc[map_side.Map == m_name, f"{T}"] += map_data["T1_score_H2"][x] + map_data["T2_score_H1"][x]
    else:
        map_side.loc[map_side.Map == m_name, f"{CT}"] += map_data["T1_score_H2"][x] + map_data["T2_score_H1"][x]
        map_side.loc[map_side.Map == m_name, f"{T}"] += map_data["T1_score_H1"][x] + map_data["T2_score_H2"][x]


# In[ ]:


#calculate win rate percentages
map_side["CT_winrate"] = map_side["CT_wins"] / (map_side["CT_wins"] + map_side["T_wins"])
map_side["T_winrate"] = map_side["T_wins"] / (map_side["CT_wins"] + map_side["T_wins"])


# In[ ]:


#flag maps as T or CT
map_side["T_or_CT"] = ""

for x in range(9):
    if (map_side.iloc[x]["CT_winrate"] >= 0.5):
        map_side.at[x,"T_or_CT"] = "CT"
    else:
        map_side.at[x,"T_or_CT"] = "T"


# In[ ]:


map_side


# In[ ]:


alt.Chart(map_side).mark_bar().encode(
    x = "Map",
    y = alt.Y("CT_winrate:Q",
        scale = alt.Scale(domain = (.45, .55)))
).properties(
    title = "Map CT win rates"
)


# From the data above, we can see that maps like Nuke, Train and Mirage and Overpass are more heavily CT sided while maps like Vertigo, Dust2 and Cache and Cobblestone are more T-sided. 
# 
# Inferno is quite balanced, with almost a perfectly even 50% win rate. 
# 
# With the data on which side is more likely to win a map, now we want to calculate how each team performs on each side of the map. That can be used to determine if a team is proficient at a map or not, and then use that as another metric to predict the outcome of a match. 

# In[ ]:


#create an array with all unique team names
team_names = np.concatenate([map_data.T1_name.unique(), map_data.T2_name.unique()])
team_names = pd.unique(team_names)

#dictionary to convert into df for individual teams and their success rate on each map
team_map_data = {"Team": team_names}
for x in map_names:
    team_map_data[x + "_CT_wins"] = 0
    team_map_data[x + "_CT_losses"] = 0
    team_map_data[x + "_T_wins"] = 0
    team_map_data[x + "_T_losses"] = 0
    team_map_data[x + "_CT_winrate"] = 0
    team_map_data[x + "_T_winrate"] = 0
    team_map_data[x + "_game_count"] = 0

#df of teams and their map data, empty
team_map = pd.DataFrame(team_map_data)


# In[ ]:


#define a function to compute the desired raw stats
team_map_elements = ["_CT_wins", "_CT_losses", "_T_wins", "_T_losses"]

def map_data_updater(team_name, score_headers, map_title, index):
    for a, b in zip(team_map_elements, score_headers):
        team_map.loc[team_map.Team == team_name, map_title + a] += map_data[b][index]
    team_map.loc[team_map.Team == team_name, map_name + "_game_count"] += 1


# In[ ]:


for x in map_data.index:
    team1_name = map_data["T1_name"][x]
    team2_name = map_data["T2_name"][x]
    map_name = map_data["Map"][x]

    #call on map_data_updater to fill out scores as desired
    if (map_data["T1_side"][x] == "CT"):
        score_lst = ["T1_score_H1", "T2_score_H1", "T1_score_H2", "T2_score_H2"]
    else:
        score_lst = ["T1_score_H2", "T2_score_H2", "T1_score_H1", "T2_score_H1"]
    map_data_updater(team1_name, score_lst, map_name, x)
    map_data_updater(team2_name, score_lst[::-1], map_name, x)


# In[ ]:


#calculating each team's win rate on each map
var = 1
for x in range(9):
    team_map.iloc[:,var + 4] = team_map.iloc[:,var]/(team_map.iloc[:,var] + team_map.iloc[:,var + 1])
    team_map.iloc[:,var + 5] = team_map.iloc[:,var + 2]/(team_map.iloc[:,var + 2] + team_map.iloc[:,var + 3])
    var = var + 7

team_map = team_map.fillna(0)


# In[ ]:


team_map


# With this new table of data/win rates, I want a way to quantify a team's success rate on each respective map. This new quantifier will be called "map proficiency", a score defined by:
# 
#     (Map CT win rate + Map T win rate) * games played
# 
# This will give me a rough idea of how good each team is on a given map, as well as include the team's experience in some way. In a tactical game like CS:GO, map experience can be a big determining factor in how a team performs. 

# In[ ]:


#map proficiency score calculation
for x in map_names:
    team_map[x + "_aggregate_percentage"] = (team_map[x + "_CT_winrate"] + team_map[x + "_T_winrate"])

for x in map_names:
    team_map[x + "_proficiency"] = team_map[x + "_aggregate_percentage"] * team_map[x + "_game_count"]


# In[ ]:


team_map


# Now I want to chart proficiency ratings against T or CT win rates, depending on which side a map favors (note: Inferno is incredibly even, so T/CT makes almost no difference). This can give a visual indication of approximately in what range do the best teams perform, as illustrated by their higher proficiency scores. 
# 
# In these graphs I am electing to choose either using T or CT winrates along the x-axis according to each map's bias, since this gives the best indication of how a team performs when they have an advantage. I have tried doing the opposite, but since the win rates are lower the graphical representation is much more scattered and difficult to observe. 

# In[ ]:


def make_proficiency_chart(maps):
    counter = map_side.index[map_side["Map"] == maps]
    if(map_side.at[counter[0],"T_or_CT"] == "CT"):
        phrase = "_CT_winrate"
    else: 
        phrase = "_T_winrate"

    c = alt.Chart(team_map).mark_circle().encode(
        x = alt.X(maps + phrase + ":Q",
            scale = alt.Scale(domain = (0.0, 1.0))),
        y = alt.Y(maps + "_proficiency:Q",
            scale = alt.Scale(domain = (0, 250))) 

    ).properties(
        title = f"{maps} winrate vs proficiency"
    )
    return c


# In[ ]:


alt.vconcat(*[make_proficiency_chart(k) for k in map_names])


# A general trend can be observed where better teams tend to perform in accordance with each map's bias, whether that is T or CT sided. Inferno lines up right down the middle, illustrating its 50/50 win percentages. Cache and Cobblestone have a lack of data, most likely due to these maps being rotated out early on within the time frame captured by this data set. 

# With the analysis on individual maps completed, along with the performace of each team on any given map, it is now time to augment a copy of the match_df dataframe so newly constructed variables can be used to analyze the outcome of matches.
# 
#     T1_pref_score: preference score for a given match, incremented by one for each map that the first team has an aggregate rating over 1 (up to 3). 
#     T1_proficiency: proficiency score of team 1, as calculated earlier. 

# In[ ]:


#initialize new columns
match_df_final = match_df.copy()
m_df_add = ["T1_pref_score", "T2_pref_score", "T1_proficiency", "T2_proficiency"]

for x in m_df_add:
    match_df_final[x] = 0


# In[ ]:


def map_pref_proficiency(team, map_name_lst):
    preference = 0
    proficiency = 0

    for x in map_name_lst:
        if (team_map.loc[team_map["Team"] == team, x + "_aggregate_percentage"].values[0] > 1):
            preference += 1
        proficiency += team_map.loc[team_map["Team"] == team, x + "_proficiency"].values[0]
    ret_arr = [preference, proficiency]
    return ret_arr


# In[ ]:


#this entry was causing an issue, M3 wasn't filled out correctly
match_df_final.loc[3119, "M3"] = "Cache"

#pd.options.mode.chained_assignment = None

for x in match_df_final.index:
    maps = [match_df_final["M1"][x], match_df_final["M2"][x], match_df_final["M3"][x]]
    team1 = match_df_final["first_team"][x]
    team2 = match_df_final["second_team"][x]
    T1_vals = []
    T2_vals = []

    T1_vals = map_pref_proficiency(team1, maps)
    T2_vals = map_pref_proficiency(team2, maps)
    
    #for a, b in zip(m_df_add, [0, 0, 1, 1]):
    #    match_df_final.at[x, a] = T1_vals[b]
    #for a, b in zip(team_map_elements, score_headers):
    match_df_final.at[x, "T1_pref_score"] = T1_vals[0]
    match_df_final.at[x, "T1_proficiency"] = T1_vals[1]
    match_df_final.at[x, "T2_pref_score"] = T2_vals[0]
    match_df_final.at[x, "T2_proficiency"] = T2_vals[1]


# In[ ]:


match_df_final


# Below: calculating how likely a team is to win based purely on their global rank

# In[ ]:


#how likely will a team win based on their global rank?
wr_arr = ["first_team_world_rank_#", "second_team_world_rank_#", "first_team_won"]
wr_df = match_df_final[wr_arr].copy()
scaler = StandardScaler()

wr_df = wr_df[wr_df["first_team_world_rank_#"] != "Unranked"]
wr_df = wr_df[wr_df["second_team_world_rank_#"] != "Unranked"]
wr_df2 = wr_df.astype(int)

dfrnk = wr_df2[["first_team_world_rank_#", "second_team_world_rank_#"]]


# In[ ]:


scaler.fit(dfrnk)
X_scaled = scaler.transform(dfrnk[["first_team_world_rank_#", "second_team_world_rank_#"]])
y = wr_df2["first_team_won"]

clf = KNeighborsClassifier(n_neighbors = 7)
clf.fit(X_scaled, wr_df2["first_team_won"])
dfrnk["pred"] = clf.predict(X_scaled)
X_train, X_test, y_train, y_test = train_test_split(X_scaled,y,test_size=0.2)
from sklearn.linear_model import LogisticRegression
cmf = LogisticRegression()
cmf.fit(X_train, y_train)
cmf.predict(X_test) == y_test


# In[ ]:


np.count_nonzero(clf.predict(X_test) == y_test)/len(X_test)


# In[ ]:


np.count_nonzero(clf.predict(X_train) == y_train)/len(X_train)


# it appears that global ranking is a pretty good indicator of winning, as is expected since the two should go hand in hand. This result is not very interesting but it can serve as a baseline

# In[ ]:


#creating df to analyze with machine learning
analysis_df = match_df_final.iloc[:, np.r_[2:4, 34:38]].copy()
analysis_df["first_team_won"] = match_df_final["first_team_won"].copy()
aScaler = StandardScaler()

#remove rows w/ no world rank for a given team, only removes about 50 entries 
analysis_df = analysis_df[analysis_df["first_team_world_rank_#"] != "Unranked"]
analysis_df = analysis_df[analysis_df["second_team_world_rank_#"] != "Unranked"]

analysis_df_X = analysis_df.iloc[:,0:6]


# In[ ]:


analysis_df_X


# In[ ]:


from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


#scaling values of analysis_df_X
aScaler.fit(analysis_df_X)
X_scaled = aScaler.transform(analysis_df_X)
y = analysis_df["first_team_won"]
X_train, X_test, y_train, y_test = train_test_split(X_scaled,y,test_size=0.15)


# Below is mostly from code found in class, used to graph outputs in relation to different K values for KNeighborsCLassifier. I liked this code since it gives a good representation of what values of K would do best, granted the data set is good.

# In[ ]:


def score_generator(k):
    cmf = KNeighborsClassifier(n_neighbors = k)
    cmf.fit(X_scaled, y)
    train_error = mean_absolute_error(cmf.predict(X_train), y_train)
    test_error = mean_absolute_error(cmf.predict(X_test), y_test)
    return (train_error, test_error)


# In[ ]:


#define df with 200 empty cells, with 3 columns of appropriate names 
df_scores = pd.DataFrame({"k":range(1,200),"train_error":np.nan,"test_error":np.nan})


# In[ ]:


for i in df_scores.index:
    df_scores.loc[i,["train_error","test_error"]] = score_generator(df_scores.loc[i,"k"])


# In[ ]:


#scale the values of K and create graphs for the training and test sets
df_scores["1/k"] = 1/df_scores.k

KNtrain = alt.Chart(df_scores).mark_line().encode(
    x = "1/k",
    y = "train_error"
)
KNtest = alt.Chart(df_scores).mark_line(color="orange").encode(
    x = "1/k",
    y = "test_error"
)


# In[ ]:


#combine the graphs
KNtrain + KNtest


# Looking at the graph, it doesn't seem to bode well for this data set. It appears that test error and training error start at near zero, which is strange, I'm not sure what is causing this. What is more concerning is that both sets seem to increase in error as K increases.

# Now, I want to try using a different machine learning method, this time with random forests. 

# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# Defining another score generator so I can graph the outputs of this with different K values as well. 

# In[ ]:


def score_generator_forest(k):
    rfc = RandomForestClassifier(n_estimators = k, max_features=None, max_depth=None, min_samples_split=2)
    rfc.fit(X_train, y_train)
    train_error = mean_absolute_error(rfc.predict(X_train), y_train)
    test_error = mean_absolute_error(rfc.predict(X_test), y_test)
    return (train_error, test_error)


# In[ ]:


df_scores_forest = pd.DataFrame({"k":range(1,50),"train_error":np.nan,"test_error":np.nan})
for i in df_scores_forest.index:
    df_scores_forest.loc[i,["train_error","test_error"]] = score_generator_forest(df_scores_forest.loc[i,"k"])


# This time I'm opting not to scale the values of K

# In[ ]:


RFtrain = alt.Chart(df_scores_forest).mark_line().encode(
    x = "k",
    y = "train_error"
)
RFtest = alt.Chart(df_scores_forest).mark_line(color="orange").encode(
    x = "k",
    y = "test_error"
)


# In[ ]:


RFtrain + RFtest


# This graph more closely resembles what results I would be expecting, as the test error is always higher than the training set's error. The random trees model seems to maximize at around 20 trees, with the test sets plateauing around 35% error (so ~65% error on average at 20+ trees). It appears that neither method of machine learning fared especially well for the match analysis I wanted to do, as the featured engineered columns seem to introduce more chaos than clarity. 

# ## Summary
# 
# I set out to try and predict the match outcomes of professional CS:GO matches, and I started by feature engineering columns of data to analyze. 
# 
# First was calculating the raw match data, so I could then compare that to each respective team's performance. With that comparison, I could determine if a team was "proficient" or not to some degree on a given map, and that was assigned a score based on their combined win/loss percentage multiplied by their number of games on that given map.
# 
# I thought multiplying those values would give a more accurate reflection of how well a team would perform on that map. I finally funneled all of that data back into a copy of the original dataframe. 
# 
# The machine learning was not as great as expected, seeing as the use of random trees was around 65% accurate while just depending on a team's world rank was around 70% accurate. I think that the training set size wasn't an issue, but rather the data itself. The metrics I chose could be good predictors of how teams would perform if they were just playing a best of one series, but in the highest echelons of professional play, CS:GO is played in best of three series. 
# 
# This means that one of my feature engineered columns, the match preference value, grossly oversimplifies the complexity involved in how teams pick/ban maps, and there may be a better way to find a total aggregate map proficiency value than what I did. Finding a way to incorporate the teams themselves into the machine learning could have also been better than just using a world ranking, since world rank on its own doesn't account for each team's strengths and weaknesses. 

# ## References
# 
# Dataset for CS:GO found on Kaggle, which was imported from HLTV

# Kaggle integration on DeepNote: https://deepnote.com/@dhruvildave/Kaggle-heouwNORROiS3aTQFvbklg
# 
# Map pick/ban order in competitive CSGO (BO3s): https://help.challengermode.com/en/articles/684985-how-to-ban-and-pick-cs-go-maps
# 
# Using np.r_ indexer: https://stackoverflow.com/questions/45985877/slicing-multiple-column-ranges-from-a-dataframe-using-iloc
# 
# Graphical representation of different KNeighborsRegressor values: https://christopherdavisuci.github.io/UCI-Math-10-W22/Week6/Week6-Wednesday.html

# <a style='text-decoration:none;line-height:16px;display:flex;color:#5B5B62;padding:10px;justify-content:end;' href='https://deepnote.com?utm_source=created-in-deepnote-cell&projectId=10c9d67f-1040-464b-b5c8-4b01ec2239ca' target="_blank">
# <img alt='Created in deepnote.com' style='display:inline;max-height:16px;margin:0px;margin-right:7.5px;' src='data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iODBweCIgaGVpZ2h0PSI4MHB4IiB2aWV3Qm94PSIwIDAgODAgODAiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8IS0tIEdlbmVyYXRvcjogU2tldGNoIDU0LjEgKDc2NDkwKSAtIGh0dHBzOi8vc2tldGNoYXBwLmNvbSAtLT4KICAgIDx0aXRsZT5Hcm91cCAzPC90aXRsZT4KICAgIDxkZXNjPkNyZWF0ZWQgd2l0aCBTa2V0Y2guPC9kZXNjPgogICAgPGcgaWQ9IkxhbmRpbmciIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJBcnRib2FyZCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTEyMzUuMDAwMDAwLCAtNzkuMDAwMDAwKSI+CiAgICAgICAgICAgIDxnIGlkPSJHcm91cC0zIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxMjM1LjAwMDAwMCwgNzkuMDAwMDAwKSI+CiAgICAgICAgICAgICAgICA8cG9seWdvbiBpZD0iUGF0aC0yMCIgZmlsbD0iIzAyNjVCNCIgcG9pbnRzPSIyLjM3NjIzNzYyIDgwIDM4LjA0NzY2NjcgODAgNTcuODIxNzgyMiA3My44MDU3NTkyIDU3LjgyMTc4MjIgMzIuNzU5MjczOSAzOS4xNDAyMjc4IDMxLjY4MzE2ODMiPjwvcG9seWdvbj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0zNS4wMDc3MTgsODAgQzQyLjkwNjIwMDcsNzYuNDU0OTM1OCA0Ny41NjQ5MTY3LDcxLjU0MjI2NzEgNDguOTgzODY2LDY1LjI2MTk5MzkgQzUxLjExMjI4OTksNTUuODQxNTg0MiA0MS42NzcxNzk1LDQ5LjIxMjIyODQgMjUuNjIzOTg0Niw0OS4yMTIyMjg0IEMyNS40ODQ5Mjg5LDQ5LjEyNjg0NDggMjkuODI2MTI5Niw0My4yODM4MjQ4IDM4LjY0NzU4NjksMzEuNjgzMTY4MyBMNzIuODcxMjg3MSwzMi41NTQ0MjUgTDY1LjI4MDk3Myw2Ny42NzYzNDIxIEw1MS4xMTIyODk5LDc3LjM3NjE0NCBMMzUuMDA3NzE4LDgwIFoiIGlkPSJQYXRoLTIyIiBmaWxsPSIjMDAyODY4Ij48L3BhdGg+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMCwzNy43MzA0NDA1IEwyNy4xMTQ1MzcsMC4yNTcxMTE0MzYgQzYyLjM3MTUxMjMsLTEuOTkwNzE3MDEgODAsMTAuNTAwMzkyNyA4MCwzNy43MzA0NDA1IEM4MCw2NC45NjA0ODgyIDY0Ljc3NjUwMzgsNzkuMDUwMzQxNCAzNC4zMjk1MTEzLDgwIEM0Ny4wNTUzNDg5LDc3LjU2NzA4MDggNTMuNDE4MjY3Nyw3MC4zMTM2MTAzIDUzLjQxODI2NzcsNTguMjM5NTg4NSBDNTMuNDE4MjY3Nyw0MC4xMjg1NTU3IDM2LjMwMzk1NDQsMzcuNzMwNDQwNSAyNS4yMjc0MTcsMzcuNzMwNDQwNSBDMTcuODQzMDU4NiwzNy43MzA0NDA1IDkuNDMzOTE5NjYsMzcuNzMwNDQwNSAwLDM3LjczMDQ0MDUgWiIgaWQ9IlBhdGgtMTkiIGZpbGw9IiMzNzkzRUYiPjwvcGF0aD4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+' > </img>
# Created in <span style='font-weight:600;margin-left:4px;'>Deepnote</span></a>
