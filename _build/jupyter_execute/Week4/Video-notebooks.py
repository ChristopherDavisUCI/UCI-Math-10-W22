#!/usr/bin/env python
# coding: utf-8

# # Week 4 Video notebooks
# 
# This is the notebook file corresponding to the Week 4 videos.

# ## Encoding data types
# 
# Reference: [Altair documentation](https://altair-viz.github.io/user_guide/encoding.html#encoding-data-types)

# In[1]:


import pandas as pd
import altair as alt


# In[2]:


df = pd.DataFrame({"a":[3,2,1,4],"b":[4,8,3,1]})


# In[3]:


alt.Chart(df).mark_bar().encode(
    x = "a",
    y = "b"
)


# In[4]:


df


# In[5]:


alt.Chart(df).mark_bar(width=50).encode(
    x = "a",
    y = "b"
)


# In[6]:


alt.Chart(df).mark_bar().encode(
    x = "a:N",
    y = "b",
    color = "a:N"
)


# In[7]:


alt.Chart(df).mark_bar().encode(
    x = "a:O",
    y = "b",
    color = "a:O"
)


# In[8]:


alt.Chart(df).mark_bar().encode(
    x = alt.X("a:O", sort=None),
    y = "b",
    color = "a:O"
)


# In[9]:


df.a


# ## Interactive bar chart

# In[10]:


import pandas as pd
import altair as alt
import seaborn as sns


# In[11]:


penguin = sns.load_dataset("penguins")


# In[12]:


penguin


# In[13]:


penguin.columns


# In[14]:


c1 = alt.Chart(penguin).mark_circle().encode(
    x = alt.X('bill_length_mm', scale=alt.Scale(zero=False)),
    y = alt.Y('flipper_length_mm',scale=alt.Scale(domain=(160,240))),
    color = "species"
)


# In[15]:


type(c1)


# In[16]:


c1


# In[17]:


brush = alt.selection_interval()


# In[18]:


c1.add_selection(brush)


# In[19]:


c2 = alt.Chart(penguin).mark_bar().encode(
    x = "species",
    y = "count()",
    color = "species"
)


# In[20]:


c2


# In[21]:


c1 = c1.add_selection(brush)


# In[22]:


penguin.species.unique()


# In[23]:


c2 = alt.Chart(penguin).mark_bar().encode(
    x = alt.X("species", scale = alt.Scale(domain=penguin.species.unique())),
    y = alt.Y("count()", scale = alt.Scale(domain=(0,160))),
    color = "species"
).transform_filter(brush)


# In[24]:


c1|c2


# ## pd.to_datetime

# In[25]:


import pandas as pd
import altair as alt
import seaborn as sns


# In[26]:


taxis = sns.load_dataset("taxis")


# In[27]:


taxis.head(6)


# In[28]:


alt.Chart(taxis[::10]).mark_circle().encode(
    x = "pickup",
    y = "distance"
)


# In[29]:


taxis.dtypes


# In[30]:


taxis.loc[10,"pickup"]


# In[31]:


type(taxis.loc[10,"pickup"])


# In[32]:


alt.Chart(taxis[::10]).mark_circle().encode(
    x = "pickup:T",
    y = "distance",
    tooltip = "pickup:T"
)


# In[33]:


taxis["pickup"]


# In[34]:


pd.to_datetime(taxis["pickup"])


# In[35]:


pd.to_datetime(taxis["pickup"]).loc[0].day_name()


# In[36]:


taxis.iloc[:3]


# In[37]:


alt.Chart(taxis[::10]).mark_circle().encode(
    x = "pickup",
    y = "distance",
    tooltip = "pickup"
)


# In[38]:


taxis["pickup"] = pd.to_datetime(taxis["pickup"])


# In[39]:


taxis.iloc[:3]


# In[40]:


alt.Chart(taxis[::10]).mark_circle().encode(
    x = "pickup",
    y = "distance",
    tooltip = "pickup"
)


# In[41]:


alt.Chart(taxis[:5000]).mark_circle().encode(
    x = "pickup",
    y = "distance",
    tooltip = "pickup"
)


# ## map and lambda functions

# In[42]:


import pandas as pd
import seaborn as sns


# In[43]:


df = pd.DataFrame({"a":[3,2,1,4],"b":[4.3,8.1,-2.9,1.8]})
df


# In[44]:


df["b"]**2


# In[45]:


df["b"].map(round)


# In[46]:


def square(x):
    return x**2


# In[47]:


df["b"].map(square)


# In[48]:


df["b"].map(lambda x: x**2)


# In[49]:


taxis = sns.load_dataset("taxis")


# In[50]:


taxis["dropoff_zone"].isna().sum()


# In[51]:


taxis["dropoff_zone"] = taxis["dropoff_zone"].fillna("")


# In[52]:


taxis["dropoff_zone"].isna().sum()


# In[53]:


taxis["dropoff_zone"] == "Upper West Side"


# In[54]:


taxis.loc[1,"dropoff_zone"]


# In[55]:


"Upper West Side" in taxis.loc[1,"dropoff_zone"]


# In[56]:


taxis["dropoff_zone"].map(lambda s: "Upper West Side" in s)


# In[57]:


taxis["dropoff_zone"].map(lambda s: "Upper West Side" in s).sum()


# In[58]:


taxis[taxis["dropoff_zone"].map(lambda s: "Upper West Side" in s)]

