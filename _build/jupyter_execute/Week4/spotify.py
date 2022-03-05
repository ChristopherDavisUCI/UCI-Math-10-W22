#!/usr/bin/env python
# coding: utf-8

# # Spotify dataset
# 
# [Recording of lecture from 1/24/2022](https://uci.zoom.us/rec/share/JBsBgWKEEa9fweZFzQH2J8SiWKWcVSDiMQx0dOt56nxRQ0b8dpW-Nj6-Hrp_ijkQ._1bDpwn5N2eTJ4u-?startTime=1643039871000)
# 
# The csv file attached to this project was originally taken from this [Kaggle dataset](https://www.kaggle.com/sashankpillai/spotify-top-200-charts-20202021/version/1).
# 
# To start out we want to plot Energy vs Loudness using Altair.
# 
# To change the default colors, we can select a different [color scheme](https://vega.github.io/vega/docs/schemes/).

# In[1]:


import pandas as pd
import altair as alt


# In[2]:


df = pd.read_csv("../data/spotify_dataset.csv")


# In[3]:


df.head()


# If you try to make this into a chart directly with the following code, it does not work.
# 
# ```
# alt.Chart(df).mark_circle().encode(
#     x = "Energy",
#     y = "Loudness"
# )
# ```

# A first guess is that df is too long (Altair by default only works with DataFrames with 5000 rows or fewer).

# In[4]:


alt.Chart(df.iloc[:5]).mark_circle().encode(
    x = "Energy",
    y = "Loudness"
)


# In[5]:


alt.Chart(df[:5]).mark_circle().encode(
    x = "Energy",
    y = "Loudness"
)


# In[6]:


type(df.head())


# In[7]:


df.info()


# In[8]:


df.Artist.value_counts()


# In[9]:


df["Artist"].value_counts()


# In[10]:


df[:3]


# In[11]:


# Not too long for Altair (5000 is the cutoff for Altair)
len(df)


# In[12]:


df.shape


# In[13]:


x = df.shape[0]


# In[14]:


x


# In[15]:


df.info()


# In[16]:


df.loc[10,"Energy"]


# In[17]:


df.dtypes


# In[18]:


pd.to_numeric(df["Energy"])


# In[19]:


df.isna().any(axis=1)


# In[20]:


df.isna().any(axis=1).sum()


# In[21]:


"" == " "


# In[22]:


# Tell pandas what missing values look like
df2 = pd.read_csv("../data/spotify_dataset.csv", na_values=" ")


# In[23]:


df2.dtypes


# In[24]:


# Count the bad rows
df2.isna().any(axis=1).sum()


# In[25]:


df2.isna().any(axis=1)


# In[26]:


df[df2.isna().any(axis=1)]


# In[27]:


# Count the good rows
df2.notna().all(axis=1).sum()


# In[28]:


# Keep just the good rows
df3 = df2[df2.notna().all(axis=1)].copy()
df3


# In[29]:


df3.shape


# In[30]:


alt.Chart(df2).mark_circle().encode(
    x = "Energy",
    y = "Loudness"
)


# In[31]:


alt.Chart(df2).mark_circle().encode(
    x = alt.X("Energy", scale = alt.Scale(domain=(0.1,0.8))),
    y = "Loudness"
)


# In[32]:


alt.Chart(df2).mark_circle(clip=True, color="Red",size=100).encode(
    x = alt.X("Energy", scale = alt.Scale(domain=(0.1,0.8))),
    y = "Loudness"
)


# In[33]:


df2.dtypes


# In[34]:


alt.Chart(df2).mark_circle().encode(
    x = "Energy",
    y = "Loudness",
    color = "Danceability"
)


# In[35]:


alt.Chart(df2).mark_circle().encode(
    x = "Energy",
    y = "Loudness",
    color = alt.Color("Tempo",scale=alt.Scale(scheme="Turbo")),
    tooltip = "Artist"
)


# In[36]:


# Are these the same
# ["Artist"] vs list("Artist")
# No, list("Artist") it's the characters


# In[37]:


list("Artist")


# In[38]:


sel = alt.selection_multi(fields=["Artist"])


# In[39]:


c1 = alt.Chart(df2).mark_circle().encode(
    x = "Energy",
    y = "Loudness",
    color = alt.Color("Tempo",scale=alt.Scale(scheme="Turbo")),
    tooltip = "Artist"
).add_selection(
    sel
)


# In[40]:


c2 = alt.Chart(df2).mark_circle().encode(
    x = "Energy",
    y = "Loudness",
    color = alt.Color("Tempo",scale=alt.Scale(scheme="Turbo")),
    tooltip = "Artist"
).transform_filter(
    sel
)


# In[41]:


c1|c2


# Try clicking on one or more of the points in the following chart.  (Hold down shift while clicking to select multiple points.)

# In[42]:


c3 = alt.Chart(df2).mark_circle().encode(
    x = "Energy",
    y = "Loudness",
    color = alt.Color("Tempo",scale=alt.Scale(scheme="Turbo")),
    opacity = alt.condition(sel,alt.value(1),alt.value(0.2)),
    size = alt.condition(sel,alt.value(400),alt.value(10)),
    tooltip = "Artist"
).add_selection(
    sel
)

c3


# In[ ]:




