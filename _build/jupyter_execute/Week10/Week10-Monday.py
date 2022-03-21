#!/usr/bin/env python
# coding: utf-8

# # Week 10, Monday
# 
# [YuJa recording](https://uci.yuja.com/V/Video?v=4539948&node=15119194&a=1725214600&autoplay=1)

# In[1]:


import seaborn as sns
import numpy as np
import pandas as pd


# ## Lecture
# 
# We will practice analyzing and cleaning a dataset.  This dataset contains scaled versions of the Midterm 2 scores.
# 
# * Can you identify which problem needed to be curved differently between the two versions of the midterm?

# In[2]:


df = pd.read_csv("../data/Midterm_scaled.csv")


# In[3]:


df.head()


# In[4]:


df.dtypes


# We want to convert most of those columns to numeric values.

# In[5]:


pd.to_numeric(df["1a"])


# We can fix that error by using the `errors` keyword argument.

# In[6]:


pd.to_numeric(df["1a"], errors="coerce")


# If we want to do the same thing to all the columns from "1a" to "3", we can use `apply` and a `lambda` function.

# In[7]:


df.loc[:,"1a":"3"].apply(lambda col: pd.to_numeric(col, errors="coerce"), axis=0)


# For this particular dataset, a much easier strategy is just to specify during the import that we want to skip the top two rows (after the header row).

# In[8]:


df = pd.read_csv("../data/Midterm_scaled.csv", skiprows=[1,2])


# In[9]:


df.head()


# The original dataset only specifies who had version "a", not version "b".  Let's fill in version "b" in place of the `NaN` values.

# In[10]:


df["Version"] = df["Version"].fillna("b")


# Here are some examples using `df.groupby`.  There are more examples below in the Worksheet portion.

# This shows all possible combinations of (scaled) scores on problems "1a" and "1b".  For example, this shows that 3 students scored -0.717101719 on 1a and -0.62029917 on 1b.

# In[11]:


for x,y in df.groupby(["1a","1b"]):
    print(f"The value of x is {x}")
    print(y.shape)
    print("")


# For our question of whether one version was easier than the other version, we are interested in grouping by "Version". 

# In[12]:


for x,y in df.groupby("Version"):
    print(f"The value of x is {x}")
    print(y.shape)


# In this code, `y` is a DataFrame.  Notice for example how in the first displayed DataFrame, the exams are all Version a.

# In[13]:


for x,y in df.groupby("Version"):
    print(f"The value of x is {x}")
    display(y.head())


# Instead of iterating over the different possibilities, we can also perform what is called an aggregation operation, such as taking the `mean`.

# In[14]:


df.groupby("Version").mean()


# It's a little easier to read if we take the transpose.

# In[15]:


df.groupby("Version").mean().T


# We can also apply formatting to these strings, by saying we only want three decimal places.

# In[16]:


df.groupby("Version").mean().T.applymap(lambda x: f"{x:.3f}")


# Notice how the 1b value is significantly higher in the "a" column than in the "b" column.  This is the reason that the "b" version of the exam was curved one point more than the "a" version.

# ## Worksheet
# 
# (This worksheet contains some repetition from the portion above.)

# In[17]:


df = sns.load_dataset("taxis")
df.dropna(inplace=True)


# ## Practice with pandas groupby
# 
# We haven't covered pandas groupby in Math 10 before today. This is a possible "extra topic" for the course project.
# 
# Here is an example using `groupby`.  We also use f-strings.

# In[18]:


for a,b in df.groupby("pickup_zone"):
    print(f"a is {a}")
    print(f"The type of b is {type(b)}")
    print(f"The shape of b is {b.shape}")
    break


# If we instead wanted to get the first 5 values, we could do something like the following.  For example, this indicates that 65 rides began in the pickup zone "Astoria".

# In[19]:


counter = 0

for a,b in df.groupby("pickup_zone"):
    print(f"a is {a}")
    print(f"The type of b is {type(b)}")
    print(f"The shape of b is {b.shape}")
    print("")
    counter += 1

    if counter >= 5:
        break


# You can also group by multiple categories.  For example, the following indicates that only 4 rides in the dataset began in Bronx and finished in Brooklyn.

# In[20]:


counter = 0

for a,b in df.groupby(["pickup_borough","dropoff_borough"]):
    print(f"a is {a}")
    print(f"The type of b is {type(b)}")
    print(f"The shape of b is {b.shape}")
    print("")
    counter += 1

    if counter >= 5:
        break


# Sample exercises:
# 
# 1.  For each pickup borough, using f-strings, print the average tip for rides that begin in that borough. 
# 
# 2.  Try producing a sub-DataFrame, `df_sub`, which contains only the "distance", "fare", "tip", and "pickup_zone" columns, and which contains only rows where the "tip" amount is greater than zero.  Then execute `df_sub.groupby("pickup_zone").mean()`.  What information is this providing?
# 
# 3.  Do the same thing as in the previous exercise, but instead find what the maximum was in each category, instead of the average.

# ## Practice with pandas styler
# 
# We haven't covered pandas styler in Math 10.  This is a possible "extra topic" for the course project.
# 
# Based on the [pandas documentation](https://pandas.pydata.org/pandas-docs/stable/user_guide/style.html#Styler-Functions).
# 
# As an example, we will color the cells blue for which the "pickup_zone" or "dropoff_zone" contains the word "Midtown".

# In[21]:


def make_blue(x):
    if "Midtown" in x:
        return 'color:white;background-color:darkblue'
    else:
        return None


# You will have to scroll right to see the blue cells.  We only display the first 20 rows.

# In[22]:


df[:20].style.applymap(make_blue,subset=["pickup_zone","dropoff_zone"])


# Here is a similar example, but where we color every cell in the row a random color.  Notice the use of f-strings.

# In[23]:


rng = np.random.default_rng()
color_list = ["red","purple","orange","wheat","black","blue"]
prop_list = [f'color:white;background-color:{c}' for c in color_list]

def find_midtown(row):
    if ("Midtown" in row["dropoff_zone"]) or ("Midtown" in row["pickup_zone"]):
        return rng.choice(prop_list, size=len(row))
    else:
        return [None]*len(row)


# In[24]:


df[:20].style.apply(find_midtown,axis=1)


# pandas styler sample exercises:
# 
# 1.  Try changing the text color to red on all rides where the fare was at least 10 dollars.
# 
# 2.  For all cells where the pickup time is between 11pm and midnight, try giving those cells a black background with white text.
# 
# 3.  For how many rides was the tip amount greater than 40% of the fare?  Try coloring the entire row for those rides in red.
