#!/usr/bin/env python
# coding: utf-8

# # Quiz 2 Practice Exercises
# 
# These questions relate to the [learning objectives](https://canvas.eee.uci.edu/courses/42645/pages/learning-objectives?module_item_id=1612181) for Quiz 2.
# 
# For each of these topics, we state the learning objective and then give one or more practice exercises.

# ## loc and iloc
# 
# * Access entries (and assign values to entries) within pandas using loc, iloc.

# ### Exercise: 
# 
# Define a DataFrame `df` using the following code.  (Make sure to put all of these commands in the same cell.)
# 
# ```
# import numpy as np
# import pandas as pd
# rng = np.random.default_rng(seed=30)
# A = rng.integers(-1,10,size=(10,5))
# df = pd.DataFrame(A,columns=list("abcde"))
# ```

# Write code to replace all the even-indexed (like 0, 2, 4, ...) entries in the `b` column with NumPy's *not a number*, `nan`.

# ## Distribution of data
# 
# * Use the describe, info, and value_counts functions in pandas to learn about the distribution of data within a DataFrame or Series.

# ### Exercise
# 
# Which column in the above DataFrame has the smallest mean value?  (Can you answer this question two different ways, first using one of the methods `describe`, `info`, or `value_counts`, and second using the `mean` method together with an `axis` argument?)

# ## Count occurrences
# 
# * Count occurrences in rows or columns in a NumPy array or pandas DataFrame using sum and axis.

# ### Exercise
# 
# How often does 5 occur in each row?

# ## Select rows
# 
# * Select the rows in a pandas DataFrame where a certain condition is True, possibly using some combination of any, all, and axis.

# ### Exercise
# 
# Create a new pandas DataFrame using the below code.  (Be sure to run all of this code, especially the `rng` part, even if you already created a random number generator above.)
# 
# ```
# import numpy as np
# import pandas as pd
# rng = np.random.default_rng(seed=10)
# A = 10*rng.random(size=(10**4,3))
# df = pd.DataFrame(A,columns=["x","y","z"])
# ```

# Create a new DataFrame `df2` consisting of only the rows of `df` where the `x`-value is strictly greater than the `y`-value.  What is the maximum in the `y` column of `df2`?  Enter your answer correct to two decimal places.

# ## Count rows
# 
# * Count the number of rows in a pandas DataFrame satisfying a condition.

# ### Exercise
# 
# For the DataFrame `df2` that you created above, in how many rows is the `y`-value strictly greater than the `z`-value?  If you pick a random row from `df2`, what is the probability that its `y`-value is strictly greater than its `z`-value?

# ## Missing values
# 
# * Locate the missing values within a pandas DataFrame.

# ### Exercise
# 
# Write a function which takes as input a pandas DataFrame, and as output returns the sub-DataFrame with all rows removed which contained a null value.  Possible approach: use the `isna` method and logical negation.

# ## Data types
# 
# * Identify the different data types present within a pandas DataFrame.

# ### Exercise
# 
# Read in the cars dataset using `read_csv`.  What are the different datatypes present?  What does `object` represent?  If you look at the `Horsepower` column, all the values seem to be integers, yet pandas represents the data type as `float64`.  Do you have a guess why that is?  (Hint.  It's not that there is secretly some fractional horsepower somewhere in it.  It relates to the previous learning objective.)

# ## Logic
# 
# * Find content satisfying conditions using logical statements (`not`, `and`, `or`) in Python and in pandas/NumPy (`~`, `&`, `|`).

# ### Exercise
# 
# What is the average weight of cars in the dataset that are from Europe or Japan?  (Your answer should be a single number.)

# <a style='text-decoration:none;line-height:16px;display:flex;color:#5B5B62;padding:10px;justify-content:end;' href='https://deepnote.com?utm_source=created-in-deepnote-cell&projectId=49f11f28-f304-47f3-a40b-e0d3be20bd63' target="_blank">
# <img alt='Created in deepnote.com' style='display:inline;max-height:16px;margin:0px;margin-right:7.5px;' src='data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iODBweCIgaGVpZ2h0PSI4MHB4IiB2aWV3Qm94PSIwIDAgODAgODAiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8IS0tIEdlbmVyYXRvcjogU2tldGNoIDU0LjEgKDc2NDkwKSAtIGh0dHBzOi8vc2tldGNoYXBwLmNvbSAtLT4KICAgIDx0aXRsZT5Hcm91cCAzPC90aXRsZT4KICAgIDxkZXNjPkNyZWF0ZWQgd2l0aCBTa2V0Y2guPC9kZXNjPgogICAgPGcgaWQ9IkxhbmRpbmciIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJBcnRib2FyZCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTEyMzUuMDAwMDAwLCAtNzkuMDAwMDAwKSI+CiAgICAgICAgICAgIDxnIGlkPSJHcm91cC0zIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxMjM1LjAwMDAwMCwgNzkuMDAwMDAwKSI+CiAgICAgICAgICAgICAgICA8cG9seWdvbiBpZD0iUGF0aC0yMCIgZmlsbD0iIzAyNjVCNCIgcG9pbnRzPSIyLjM3NjIzNzYyIDgwIDM4LjA0NzY2NjcgODAgNTcuODIxNzgyMiA3My44MDU3NTkyIDU3LjgyMTc4MjIgMzIuNzU5MjczOSAzOS4xNDAyMjc4IDMxLjY4MzE2ODMiPjwvcG9seWdvbj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0zNS4wMDc3MTgsODAgQzQyLjkwNjIwMDcsNzYuNDU0OTM1OCA0Ny41NjQ5MTY3LDcxLjU0MjI2NzEgNDguOTgzODY2LDY1LjI2MTk5MzkgQzUxLjExMjI4OTksNTUuODQxNTg0MiA0MS42NzcxNzk1LDQ5LjIxMjIyODQgMjUuNjIzOTg0Niw0OS4yMTIyMjg0IEMyNS40ODQ5Mjg5LDQ5LjEyNjg0NDggMjkuODI2MTI5Niw0My4yODM4MjQ4IDM4LjY0NzU4NjksMzEuNjgzMTY4MyBMNzIuODcxMjg3MSwzMi41NTQ0MjUgTDY1LjI4MDk3Myw2Ny42NzYzNDIxIEw1MS4xMTIyODk5LDc3LjM3NjE0NCBMMzUuMDA3NzE4LDgwIFoiIGlkPSJQYXRoLTIyIiBmaWxsPSIjMDAyODY4Ij48L3BhdGg+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMCwzNy43MzA0NDA1IEwyNy4xMTQ1MzcsMC4yNTcxMTE0MzYgQzYyLjM3MTUxMjMsLTEuOTkwNzE3MDEgODAsMTAuNTAwMzkyNyA4MCwzNy43MzA0NDA1IEM4MCw2NC45NjA0ODgyIDY0Ljc3NjUwMzgsNzkuMDUwMzQxNCAzNC4zMjk1MTEzLDgwIEM0Ny4wNTUzNDg5LDc3LjU2NzA4MDggNTMuNDE4MjY3Nyw3MC4zMTM2MTAzIDUzLjQxODI2NzcsNTguMjM5NTg4NSBDNTMuNDE4MjY3Nyw0MC4xMjg1NTU3IDM2LjMwMzk1NDQsMzcuNzMwNDQwNSAyNS4yMjc0MTcsMzcuNzMwNDQwNSBDMTcuODQzMDU4NiwzNy43MzA0NDA1IDkuNDMzOTE5NjYsMzcuNzMwNDQwNSAwLDM3LjczMDQ0MDUgWiIgaWQ9IlBhdGgtMTkiIGZpbGw9IiMzNzkzRUYiPjwvcGF0aD4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+' > </img>
# Created in <span style='font-weight:600;margin-left:4px;'>Deepnote</span></a>
