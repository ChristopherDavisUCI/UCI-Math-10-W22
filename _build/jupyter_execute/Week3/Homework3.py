#!/usr/bin/env python
# coding: utf-8

# # Homework 3
# 
# **Remark**: This might not reflect updates, so check the Deepnote file for the official version.
# 
# List your name and the names of any collaborators at the top of this notebook.
# 
# (Reminder: It's encouraged to work together; you can even submit the exact same homework as another student or two students, but you must list each other's names at the top.)

# ## Practice with the Cars dataset
# 
# These exercises refer to the cars dataset.  
# 
# 1. Load the data from the `cars.csv` file using `pd.read_csv`.
# 1. Use [this method](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rename.html) and a Python dictionary to rename the `Miles_per_Gallon` column as `mpg` and the `Weight_in_lbs` column as `Weight`.
# 1.  Which `Year` appears in the dataset *least* often?
# 1.  How many distinct values occur in the `Name` column?
# 1.  Using [this method](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.corr.html), create a correlation matrix for the (numeric) columns in the cars dataset.  Which two columns are most *negatively* correlated?
# 1. Make a dictionary whose keys are *USA*, *Japan*, and *Europe*, and whose values are the average weight in pounds of the cars in the dataset from that origin.  (So for example, the value corresponding to *USA*, would be the average weight of all cars in the dataset whose origin is USA.)
# 2. Make the same thing (weights for USA, Japan, and Europe) as a pandas Series.  (One option is to convert the dictionary you made into a pandas Series.)

# ## In-class quiz practice
# 
# (This week's in-class quiz may involve instructions like this.)
# 
# Define a pandas DataFrame `df` using the following code.  (Be sure to put all of these lines in the same cell; otherwise, you might get different answers.  The important thing is that `X` and `A` get created immediately after `rng`.)
# 
# ```
# import numpy as np
# import pandas as pd
# rng = np.random.default_rng(seed=12)
# X = np.concatenate(([np.nan],np.arange(100)))
# A = rng.choice(X,size=(10**5,5))
# df = pd.DataFrame(A,columns=list("abcde"))
# ```
# 
# 1.  How many null values are there in the `c` column of `df`?
# 1.  How many of the values in the `b` column are equal to 49 or to 50?
# 2.  Consider the sub-DataFrame consisting of all the rows in which the `b` column is strictly greater than 95 and the `c` column is less than 5.  What is the average value in the `a` column for the resulting DataFrame?  Enter your answer correct to two decimal places.
# 4.  How many of the rows of `df` contain the number 0?  (Warning.  This is not the same as asking how many 0s occur in the DataFrame.  Use `sum`, `any`, `axis`, and a Boolean DataFrame.)

# ## Submission
# 
# Download the .ipynb file for this notebook (click on the folder icon to the left, then the ... next to the file name) and upload the file on Canvas.

# <a style='text-decoration:none;line-height:16px;display:flex;color:#5B5B62;padding:10px;justify-content:end;' href='https://deepnote.com?utm_source=created-in-deepnote-cell&projectId=9d4574ed-aaea-4c74-a2ad-73f63cd530d1' target="_blank">
# <img alt='Created in deepnote.com' style='display:inline;max-height:16px;margin:0px;margin-right:7.5px;' src='data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iODBweCIgaGVpZ2h0PSI4MHB4IiB2aWV3Qm94PSIwIDAgODAgODAiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8IS0tIEdlbmVyYXRvcjogU2tldGNoIDU0LjEgKDc2NDkwKSAtIGh0dHBzOi8vc2tldGNoYXBwLmNvbSAtLT4KICAgIDx0aXRsZT5Hcm91cCAzPC90aXRsZT4KICAgIDxkZXNjPkNyZWF0ZWQgd2l0aCBTa2V0Y2guPC9kZXNjPgogICAgPGcgaWQ9IkxhbmRpbmciIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJBcnRib2FyZCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTEyMzUuMDAwMDAwLCAtNzkuMDAwMDAwKSI+CiAgICAgICAgICAgIDxnIGlkPSJHcm91cC0zIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxMjM1LjAwMDAwMCwgNzkuMDAwMDAwKSI+CiAgICAgICAgICAgICAgICA8cG9seWdvbiBpZD0iUGF0aC0yMCIgZmlsbD0iIzAyNjVCNCIgcG9pbnRzPSIyLjM3NjIzNzYyIDgwIDM4LjA0NzY2NjcgODAgNTcuODIxNzgyMiA3My44MDU3NTkyIDU3LjgyMTc4MjIgMzIuNzU5MjczOSAzOS4xNDAyMjc4IDMxLjY4MzE2ODMiPjwvcG9seWdvbj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0zNS4wMDc3MTgsODAgQzQyLjkwNjIwMDcsNzYuNDU0OTM1OCA0Ny41NjQ5MTY3LDcxLjU0MjI2NzEgNDguOTgzODY2LDY1LjI2MTk5MzkgQzUxLjExMjI4OTksNTUuODQxNTg0MiA0MS42NzcxNzk1LDQ5LjIxMjIyODQgMjUuNjIzOTg0Niw0OS4yMTIyMjg0IEMyNS40ODQ5Mjg5LDQ5LjEyNjg0NDggMjkuODI2MTI5Niw0My4yODM4MjQ4IDM4LjY0NzU4NjksMzEuNjgzMTY4MyBMNzIuODcxMjg3MSwzMi41NTQ0MjUgTDY1LjI4MDk3Myw2Ny42NzYzNDIxIEw1MS4xMTIyODk5LDc3LjM3NjE0NCBMMzUuMDA3NzE4LDgwIFoiIGlkPSJQYXRoLTIyIiBmaWxsPSIjMDAyODY4Ij48L3BhdGg+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMCwzNy43MzA0NDA1IEwyNy4xMTQ1MzcsMC4yNTcxMTE0MzYgQzYyLjM3MTUxMjMsLTEuOTkwNzE3MDEgODAsMTAuNTAwMzkyNyA4MCwzNy43MzA0NDA1IEM4MCw2NC45NjA0ODgyIDY0Ljc3NjUwMzgsNzkuMDUwMzQxNCAzNC4zMjk1MTEzLDgwIEM0Ny4wNTUzNDg5LDc3LjU2NzA4MDggNTMuNDE4MjY3Nyw3MC4zMTM2MTAzIDUzLjQxODI2NzcsNTguMjM5NTg4NSBDNTMuNDE4MjY3Nyw0MC4xMjg1NTU3IDM2LjMwMzk1NDQsMzcuNzMwNDQwNSAyNS4yMjc0MTcsMzcuNzMwNDQwNSBDMTcuODQzMDU4NiwzNy43MzA0NDA1IDkuNDMzOTE5NjYsMzcuNzMwNDQwNSAwLDM3LjczMDQ0MDUgWiIgaWQ9IlBhdGgtMTkiIGZpbGw9IiMzNzkzRUYiPjwvcGF0aD4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+' > </img>
# Created in <span style='font-weight:600;margin-left:4px;'>Deepnote</span></a>
