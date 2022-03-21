#!/usr/bin/env python
# coding: utf-8

# # Project Title
# 
# Author: Jia Bao Zhen
# 
# Course Project, UC Irvine, Math 10, W22

# ## Introduction
# 
# Introduce your project here.  About 3 sentences.

# The aspect of this data set that I want to explore is whether the total stats of a Pokemon can determine the type of the Pokemon. Determine whether the total stats can also predict a Pokemons' secondary typing if applicable to the Pokemon as not all Pokemon have two types. As well as trying to predict whether total stats can determine whether a Pokemon has dual typing or not.

# ## Main portion of the project
# 
# (You can either have all one section or divide into multiple sections)

# In[ ]:


import pandas as pd
import altair as alt
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss


# In[ ]:


pokemon = pd.read_csv('Pokemon.csv')
pokemon['Dual Type'] = ~pokemon['Type 2'].isna()
pokemon.head()


# Overview of the dataset, there are NA values for some Pokemon in the column `Type 2` as not every Pokemon has two typings, we will not drop the Pokemon that are missing a second typing in this dataset, but I have added a boolean column called `Dual Type` that returns true if the Pokemon has a `Type 2` and false otherwise.

# In[ ]:


choice = alt.selection_multi(fields=['Type 1'], bind='legend')

hist1 = alt.Chart(pokemon).mark_bar(size=10).encode(
    x = 'Total',
    y = 'count()',
    color = 'Type 1',
    opacity = alt.condition(choice, alt.value(1), alt.value(0.2))
).add_selection(
    choice
).properties(
    title='Pokemon Type and Total Stat Distribution'
)


# In[ ]:


hist2 = alt.Chart(pokemon).mark_bar(size=10).encode(
    x = 'Total',
    y = 'count()',
    color = 'Type 1'
).transform_filter(choice).properties(
    title='Pokemon Type and Total Stat Distribution'
)


# In[ ]:


hist1 | hist2


# Create two interactive histogram and concat them together to show the distribution of total stats by the type of Pokemon.

# In[ ]:


type1 = alt.Chart(pokemon).mark_bar(size=10).encode(
    x = alt.X('Type 1', sort='y'),
    y = 'count()',
    color = alt.Color('Type 1', legend=None)
).properties(
    title='Number of Pokemon by Type 1'
)
type1


# Graph to show the number of Pokemon based on their first type sorted from lowest to highest. From it we can see that there are a lot of Pokemon whose main typing is water.

# In[ ]:


data = pokemon.copy()
data.dropna(inplace=True)
poke_type2 = pd.DataFrame(data)
poke_type2.reset_index(inplace=True)
poke_type2.head()


# In[ ]:


print(f'The number of Pokemon in this dataset with a second type is {poke_type2.shape[0]}')


# Store the secondary typing of Pokemons into a new data frame and drop the NA values.

# In[ ]:


type2 = alt.Chart(poke_type2).mark_bar(size=10).encode(
    x = alt.X('Type 2', sort='y'),
    y = 'count()',
    color = alt.Color('Type 2', legend=None)
).properties(
    title='Number of Pokemon by Type 2'
)
type2


# Graph to show the number of Pokemon based on their second type sorted from lowest to highest. Visualization does not contain the same number of Pokemon as the previous graph as not all Pokemon have a second type. However, from it we can see that there are a lot of Pokemon whose secondary typing is flying.

# In[ ]:


corr_data = (pokemon.drop(columns=['#', 'Name', 'Type 1', 'Type 2', 'Dual Type', 'Legendary', 'Generation'])
    .corr().stack()
    .reset_index()
    .rename(columns={0: 'Correlation', 'level_0' : 'Var1', 'level_1' : 'Var2'})
    )

corr_data['Correlation'] = corr_data['Correlation'].round(2)

corr_data.head()


# Calculate the correlation between different variables with '.corr' function and using '.stack()' to be able to graph it in altair.

# In[ ]:


base = alt.Chart(corr_data).encode(
    x = 'Var1:O',
    y = 'Var2:O'
)

text = base.mark_text().encode(
    text = 'Correlation',
    color = alt.condition(
        alt.datum.correlation > 0.5,
        alt.value('white'),
        alt.value('black')
    )
)

corr_plot = base.mark_rect().encode(
    color = 'Correlation:Q'
).properties(
    title='Correlation by Pokemon Stats',
    width=350,
    height=350
)

corr_plot + text


# Create a correlation heatmap to represent the correlation between different stats and total stats.

# I tried to make the heatmap interactive by adapting the code [here](https://towardsdatascience.com/altair-plot-deconstruction-visualizing-the-correlation-structure-of-weather-data-38fb5668c5b1), but the attempt was unsuccessful. 

# In[ ]:


poke_stats = ['Total', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']
X = pokemon[poke_stats]
y = pokemon['Type 1']


# In[ ]:


scaler = StandardScaler()
scaler.fit(X)


# Rescaling the data to change the values of the numeric columns such as `Total, HP, Attack, Defense, Sp. Atk, Sp. Def, Speed` into a common scale.

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X,y)


# In[ ]:


clf = KNeighborsClassifier()
clf.fit(X_train, y_train)


# Fit the training data

# In[ ]:


pokemon['pred_type1'] = pd.Series(clf.predict(X_test))


# Predict the Pokemon types with the testing data, this produced an error in which the length were different so I made it into a pandas Series.

# In[ ]:


pokemon['pred_type1'] = pokemon['pred_type1'].fillna(method='ffill')


# Because of the error that occured in the previous line of code, there were null values that were resulted from the model being unable to predict the type of the Pokemon, so I filled the null values with the highest frequency Pokemon type.

# In[ ]:


log_loss(pokemon['Type 1'], clf.predict_proba(X))


# The log loss is considerably high and suggest that the model is indequate for predicting the Pokemon type.

# In[ ]:


pred_type1_graph = alt.Chart(pokemon).mark_circle().encode(
    x = alt.X('Type 1', title = 'Actual Type 1'),
    y = alt.Y('pred_type1', title = 'Predicted Type 1')
).properties(
    title='Predicted Pokemon Types by Pokemon Stats'
)

pred_type1_graph


# From this model, we can see that it is hard to predict the first type of a Pokemon simply based on their stats because there can be Pokemon with the same stats that are different types. Since this was a classification question, I chose to illustrate the data with a scatter plot to show what actual types the model is predicting. For example if we look at the x axis which is the actual type 1 of the Pokemon, we can see that Bug type Pokemons have been predicted to be Water, Rock, Normal, Grass, Ghost, Fire, Electric, and Dark besides their actual true type of Bug.

# In[ ]:


X2 = poke_type2[poke_stats]
y2 = poke_type2['Type 2']


# In[ ]:


scaler = StandardScaler()
scaler.fit(X2)


# Rescaling the data again to change the values of the numeric columns such as `Total, HP, Attack, Defense, Sp. Atk, Sp. Def, Speed` into a common scale.

# In[ ]:


X2_train, X2_test, y2_train, y2_test = train_test_split(X2,y2)


# In[ ]:


clf2 = KNeighborsClassifier()
clf2.fit(X2_train, y2_train)


# Fit the training data

# In[ ]:


poke_type2['pred_type2'] = pd.Series(clf2.predict(X2_test))


# Predict the type 2 of Pokemons' that have a type 2, like above with type 1 it produced a length error and so I made it a pandas Series.

# In[ ]:


poke_type2['pred_type2'] = poke_type2['pred_type2'].fillna(method='ffill')


# Again filling the null values with the highest frequency predicted type 2 for the sake of graphing.

# In[ ]:


log_loss(poke_type2['Type 2'], clf2.predict_proba(X2))


# Not really a surprised that the log loss for predicting type 2 of Pokemons' is also quite high, suggesting that this model is not suitable to predict Pokemon typing.

# In[ ]:


pred_type2_graph = alt.Chart(poke_type2).mark_circle().encode(
    x = alt.X('Type 2', title = 'Actual Type 2'),
    y = alt.Y('pred_type2', title = 'Predicted Type 2')
).properties(
    title='Predicted Pokemon Types by Pokemon Stats'
)

pred_type2_graph


# As with the previous model, it is difficult to use the stats of Pokemon to predict what their secondary type is. It seems that it has performed worse when trying to predict the type 2 of Pokemon as it has only predicted Flying as the type 2 for Bug type 2 Pokemon and if we had not filled the null values after creating out model with the test data, then there would not have been a predicted type 2 for Bug type 2 Pokemon at all.

# In[ ]:


X3 = pokemon[['Total']]
y3 = pokemon['Dual Type']


# In[ ]:


X3_train, X3_test, y3_train, y3_test = train_test_split(X3,y3)


# In[ ]:


clf3 = KNeighborsClassifier()
clf3.fit(X3_train, y3_train)


# In[ ]:


pokemon['pred_dual'] = pd.Series(clf3.predict(X3_test))


# In[ ]:


log_loss(pokemon['Dual Type'], clf3.predict_proba(X3))


# In[ ]:


pred_dual_type = (pokemon[['Dual Type', 'pred_dual']]).copy()
pdt = pred_dual_type.value_counts(normalize=True)
pdt


# Here we tried to see whether the total stat of a Pokemon can predict wheter that Pokemon has two types. From the data we can see that 29.5% of the time the model predicted that a Pokemon without double typing to have double typing, while 23.5% of the time it was correct in predicting Pokemon without double typing to not have double typing. On the other hand, it had the same probability of predicting true double type Pokemons as falsely predicting true double type Pokemons at 23.5%.

# ## Summary
# 
# Either summarize what you did, or summarize the results.  About 3 sentences.

# In this project, I created graphs to show the distribution of Pokemon types based on their typing. Some Pokemon have a second typing so I accomadated that by creating graphs illustrating the distribution of the second Pokemon type some Pokemon possess. Then I fitted the training data, to try to predict both types that Pokemons can possess with the stats of the Pokemon, however it turns out that there is not much indication that the Pokemon stats can properly predict the type of a Pokemon. I also fitted data to see whether the model can predict truly if a Pokemon has two types or only one and it turns out that it's roughly about the same odds that it will predict correctly or predict incorrectly so it seems that Pokemon stats are not a good variable for predicting Pokemon typing. Something that suprised me in my finding was when predicting the type 1 and type 2 of Pokemon, I had to fill in the null values of the model with the highest frequency predicted type and for type 1, the highest was Bug type even though the highest actual type 1 Pokemon in the data set was Water. On the other hand for type 2, it the highest frequency type 2 was Flying and Flying was also the highest frequency actual type 2 that Pokemons had. 

# ## References
# 
# Include references that you found helpful.  Also say where you found the dataset you used.

# I found the dataset on [Kaggle](https://www.kaggle.com/abcsds/pokemon).
# 
# I found some graphs that I liked and wanted to recreate from [here](https://www.kaggle.com/christinobarbosa/machinelearningmodel-pokemon)
# 
# I found code to recreate some of the aformentioned graphs in altair online [here](https://towardsdatascience.com/altair-plot-deconstruction-visualizing-the-correlation-structure-of-weather-data-38fb5668c5b1)

# <a style='text-decoration:none;line-height:16px;display:flex;color:#5B5B62;padding:10px;justify-content:end;' href='https://deepnote.com?utm_source=created-in-deepnote-cell&projectId=afdca740-6c61-4cfa-9a9e-fdd1cde9603e' target="_blank">
# <img alt='Created in deepnote.com' style='display:inline;max-height:16px;margin:0px;margin-right:7.5px;' src='data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iODBweCIgaGVpZ2h0PSI4MHB4IiB2aWV3Qm94PSIwIDAgODAgODAiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8IS0tIEdlbmVyYXRvcjogU2tldGNoIDU0LjEgKDc2NDkwKSAtIGh0dHBzOi8vc2tldGNoYXBwLmNvbSAtLT4KICAgIDx0aXRsZT5Hcm91cCAzPC90aXRsZT4KICAgIDxkZXNjPkNyZWF0ZWQgd2l0aCBTa2V0Y2guPC9kZXNjPgogICAgPGcgaWQ9IkxhbmRpbmciIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJBcnRib2FyZCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTEyMzUuMDAwMDAwLCAtNzkuMDAwMDAwKSI+CiAgICAgICAgICAgIDxnIGlkPSJHcm91cC0zIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxMjM1LjAwMDAwMCwgNzkuMDAwMDAwKSI+CiAgICAgICAgICAgICAgICA8cG9seWdvbiBpZD0iUGF0aC0yMCIgZmlsbD0iIzAyNjVCNCIgcG9pbnRzPSIyLjM3NjIzNzYyIDgwIDM4LjA0NzY2NjcgODAgNTcuODIxNzgyMiA3My44MDU3NTkyIDU3LjgyMTc4MjIgMzIuNzU5MjczOSAzOS4xNDAyMjc4IDMxLjY4MzE2ODMiPjwvcG9seWdvbj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0zNS4wMDc3MTgsODAgQzQyLjkwNjIwMDcsNzYuNDU0OTM1OCA0Ny41NjQ5MTY3LDcxLjU0MjI2NzEgNDguOTgzODY2LDY1LjI2MTk5MzkgQzUxLjExMjI4OTksNTUuODQxNTg0MiA0MS42NzcxNzk1LDQ5LjIxMjIyODQgMjUuNjIzOTg0Niw0OS4yMTIyMjg0IEMyNS40ODQ5Mjg5LDQ5LjEyNjg0NDggMjkuODI2MTI5Niw0My4yODM4MjQ4IDM4LjY0NzU4NjksMzEuNjgzMTY4MyBMNzIuODcxMjg3MSwzMi41NTQ0MjUgTDY1LjI4MDk3Myw2Ny42NzYzNDIxIEw1MS4xMTIyODk5LDc3LjM3NjE0NCBMMzUuMDA3NzE4LDgwIFoiIGlkPSJQYXRoLTIyIiBmaWxsPSIjMDAyODY4Ij48L3BhdGg+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMCwzNy43MzA0NDA1IEwyNy4xMTQ1MzcsMC4yNTcxMTE0MzYgQzYyLjM3MTUxMjMsLTEuOTkwNzE3MDEgODAsMTAuNTAwMzkyNyA4MCwzNy43MzA0NDA1IEM4MCw2NC45NjA0ODgyIDY0Ljc3NjUwMzgsNzkuMDUwMzQxNCAzNC4zMjk1MTEzLDgwIEM0Ny4wNTUzNDg5LDc3LjU2NzA4MDggNTMuNDE4MjY3Nyw3MC4zMTM2MTAzIDUzLjQxODI2NzcsNTguMjM5NTg4NSBDNTMuNDE4MjY3Nyw0MC4xMjg1NTU3IDM2LjMwMzk1NDQsMzcuNzMwNDQwNSAyNS4yMjc0MTcsMzcuNzMwNDQwNSBDMTcuODQzMDU4NiwzNy43MzA0NDA1IDkuNDMzOTE5NjYsMzcuNzMwNDQwNSAwLDM3LjczMDQ0MDUgWiIgaWQ9IlBhdGgtMTkiIGZpbGw9IiMzNzkzRUYiPjwvcGF0aD4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+' > </img>
# Created in <span style='font-weight:600;margin-left:4px;'>Deepnote</span></a>
