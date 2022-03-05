#!/usr/bin/env python
# coding: utf-8

# # Linear regression worksheet
# 
# [YuJa recording](https://uci.yuja.com/V/Video?v=4492538&node=15011998&a=1761550177&autoplay=1)
# 
# Here are some linear regression questions, since there weren't any linear regression questions on the sample midterm.

# In[1]:


import seaborn as sns
import pandas as pd
import altair as alt
from sklearn.linear_model import LinearRegression
taxis = sns.load_dataset('taxis')
taxis.dropna(inplace=True)


# In[2]:


taxis.columns


# **Question 1**: Fit a linear regression model to the taxis data using "distance","pickup hour","tip" as input (predictor) variables, and using "total" as the output (target) variable.  (**Note**: "pickup hour" is not in the original DataFrame, so you will have to create it yourself.)

# The first task is to create the "pickup hour" column.  We get an error if we try to use `.dt.hour` directly.

# In[3]:


taxis["pickup"].dt.hour


# The problem is these entries are strings.

# In[4]:


taxis["pickup"]


# In[5]:


type(taxis.iloc[0]["pickup"])


# In[6]:


taxis["pickup"].map(type)


# We convert the entries using `pd.to_datetime`.

# In[7]:


pd.to_datetime(taxis["pickup"])


# Now we can use `.dt.hour`.

# In[8]:


pd.to_datetime(taxis["pickup"]).dt.hour


# We put this new column into the DataFrame with the column name "pickup hour".

# In[9]:


taxis["pickup hour"] = pd.to_datetime(taxis["pickup"]).dt.hour


# The column now appears on the right side of the DataFrame.

# In[10]:


taxis.head()


# That was preliminary work (an example of "feature engineering").  Now we start on the linear regression portion.
# 
# We first create (instantiate) the linear regression object.

# In[11]:


reg = LinearRegression()


# Now we fit this linear regression object to the data.  Make sure to use double square brackets (the inner square brackets are making a list).

# In[12]:


reg.fit(taxis[["distance", "pickup hour", "tip"]], taxis["total"])


# **Question 2**: (a) How much does your model predict is the rate per mile?
# 
# (b) Does your model predict that taxi rides get more or less expensive later in the day?
# 
# (c) To me, the "tips" coefficient seems incorrect.  Do you see why I think that?  Do you agree or do you see an explanation of why it makes sense?

# In[13]:


reg.coef_


# In[14]:


reg.intercept_


# (a) This data suggests a rate of $2.79 per mile.
# 
# (b) The 0.04 coefficient suggests there is a slight increase in price as the taxi ride occurs later in the day (from midnight counting as earliest, 0, to 11pm counting as latest, 23).  Because the 0.04 is positive, that is why we say increase instead of decrease.
# 
# (c) Since the true formula for the total cost of the taxi ride includes the tip exactly once (as opposed to 1.466*tip), it seems a little surprising that the coefficient corresponding to tip is quite far from 1.

# **Question 3**: Let `m` be your calculated "distance" coefficient and let `b` be your calculated intercept.  Use the plotting function defined below to see how accurate the line seems.

# In[15]:


def draw_line(m,b):
    alt.data_transformers.disable_max_rows()

    c1 = alt.Chart(taxis).mark_circle().encode(
        x = "distance",
        y = "total"
    )

    xmax = 40
    df_line = pd.DataFrame({"distance":[0,xmax],"total":[b,xmax*m]})
    c2 = alt.Chart(df_line).mark_line(color="red").encode(
        x = "distance",
        y = "total"
    )
    return c1+c2


# This result looks okay, but it does not match the data very closely.  It looks significantly low.  (That makes sense, because this estimate of total as approximately 2.8*distance + 6.4 is missing the tip data (and the hour data, but that is less impactful).

# In[16]:


draw_line(2.8, 6.4)


# **Question 4**:  Fit a new linear regression model using only "distance" as your input and "total" as your output (no other variables).  Does the resulting plot seem more or less accurate? 

# In[17]:


reg2 = LinearRegression()
reg2.fit(taxis[["distance"]], taxis["total"])


# In[18]:


reg2.coef_


# In[19]:


reg2.intercept_


# This line looks significantly more accurate (as it should, since it is the "best" line, in the sense that it minimizes the Mean Squared Error).

# In[20]:


draw_line(3.23, 8.6)

