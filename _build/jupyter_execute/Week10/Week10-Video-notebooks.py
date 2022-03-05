#!/usr/bin/env python
# coding: utf-8

# # Week 10 Video notebooks

# ## Adding a title in Altair
# 
# A few references in the Altair documentation:
# * [Customizing visualizations](https://altair-viz.github.io/user_guide/customization.html)
# * [Top-level chart configuration](https://altair-viz.github.io/user_guide/configuration.html)

# In[1]:


import pandas as pd
import altair as alt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv("../data/spotify_dataset.csv", na_values=" ")
df.dropna(inplace=True)
df = df[df["Artist"].isin(["Taylor Swift", "Billie Eilish"])]

scaler = StandardScaler()
df[["Tempo","Energy"]] = scaler.fit_transform(df[["Tempo","Energy"]])

X_train, X_test, y_train, y_test = train_test_split(df[["Tempo","Energy"]], df["Artist"], test_size=0.2)


# In[2]:


clf = KNeighborsClassifier(n_neighbors=1)
clf.fit(X_train, y_train)
df["pred"] = clf.predict(df[["Tempo","Energy"]])
c = alt.Chart(df).mark_circle().encode(
    x="Tempo",
    y="Energy",
    color=alt.Color("pred", title="Legend title"),
).properties(
    title="Here is a title",
    width=700,
    height=100,
)

c.configure_legend(
    strokeColor='gray',
    fillColor='#EEEEEE',
    padding=10,
    cornerRadius=10,
    orient='top-right'
)


# ## f-strings

# In[3]:


k = 20
j = "hello"

clf = KNeighborsClassifier(n_neighbors=k)
clf.fit(X_train, y_train)
df["pred"] = clf.predict(df[["Tempo","Energy"]])
c = alt.Chart(df).mark_circle().encode(
    x="Tempo",
    y="Energy",
    color=alt.Color("pred", title="Predicted Artist"),
).properties(
    title=f"n_neighbors = {k}",
    width=700,
    height=100,
)

c.configure_legend(
    strokeColor='gray',
    fillColor='#EEEEEE',
    padding=10,
    cornerRadius=10,
    orient='top-right'
)


# ## DRY (Don't Repeat Yourself)

# In[4]:


def make_chart(k):
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(X_train, y_train)
    df[f"pred{k}"] = clf.predict(df[["Tempo","Energy"]])
    c = alt.Chart(df).mark_circle().encode(
        x="Tempo",
        y="Energy",
        color=alt.Color(f"pred{k}", title="Predicted Artist"),
    ).properties(
        title=f"n_neighbors = {k}",
        width=700,
        height=100,
    )
    return c


# In[5]:


alt.vconcat(*[make_chart(k) for k in range(1,31)])


# ## Alternatives to Deepnote
# 
# * [Jupyter notebook and Jupyter lab](https://jupyter.org/)
# * [Google colab](https://colab.research.google.com/)
