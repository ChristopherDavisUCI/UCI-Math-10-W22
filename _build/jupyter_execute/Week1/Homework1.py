#!/usr/bin/env python
# coding: utf-8

# # Homework 1
# 
# Due: 5pm Friday, Week 1, uploaded on Canvas.
# 
# Here are the main goals of this homework:
# * Become familiar with the basic functionality of Deepnote notebooks (which are very similar to Jupyter notebooks).
# * Learn about some places where you can get more information about Python functions.
# * Think about what type of material you would want to be covered in a course called, *Introduction to Programming for Data Science*.
# 
# Click the *Duplicate* button at the top right to get a version of this project that you can edit.

# ## Exercise 1
# 
# Make a markdown cell underneath this cell, and put your name and student ID in it.

# ## Exercise 2
# 
# Make a code cell underneath this cell, and using Python's `print` statement, display your name and student ID.  Execute this cell by hitting `shift+enter`.

# ## Exercise 3
# 
# Read Subsection 8.1 and Subsection 8.2 of the attached article, *50 Years of Data Science*.  These sections describe six proposed divisions of "Greater Data Science".  If you were to rank these six divisions, from "Most important to study in Math 10" to "Least important to study in Math 10", how would you rank them?
# 
# Give your personal ranking, from most important to least important, and then give a few sentences of description of why you chose that ranking.  (Two total sentences is plenty.  For example, you could say why you think the top two are the most important, or why the bottom two are least important.  You should not try to mention all six divisions in these sentences, just whatever you feel you have something to say about.)

# ## Exercise 4
# 
# My favorite website for Python tutorials is *Real Python*.  Read some of this Real Python article titled [Python's Counter: The Pythonic Way to Count Objects](https://realpython.com/python-counter/) and then answer the following questions.
# 
# 1.  What is the task/goal that article is addressing?  (Answer in about one sentence.)
# 1.  What is something from that article that you feel like you understand pretty well?
# 1.  What is something from that article that you feel like you don't yet understand?

# ## Exercise 5
# 
# Here is another way to count objects, that is specific to the Python library pandas.  Go to the *Files* section on Canvas, then to the *data* folder, and download the csv file called `spotify_dataset.csv`.  Upload that file here in Deepnote, by clicking on the folder icon on the leeft side, and then the plus icon, and then *upload file*.  After the file is uploaded, run the following cell.  It should display the top 5 rows of the dataset.

# In[1]:


import pandas as pd
df = pd.read_csv("spotify_dataset.csv")
df.head()


# Once the data is loaded and stored with the variable name `df`, run the following command: `df["Artist"].value_counts()`.  If you want to read more about this function, here is the [pandas documentation](https://pandas.pydata.org/docs/reference/api/pandas.Series.value_counts.html) for `value_counts`.
# 
# Answer the following question in a markdown cell: which Artist appears most often in this dataset?

# ## Exercise 6
# 
# Search for something related to Python on [Stack Overflow](https://stackoverflow.com/) and then do the following.
# 
# 1.  Make a new markdown cell below this cell.
# 1.  Put a heading in that cell using `##` and writing a descriptive heading.  (The heading should be a brief description of the topic you searched for.)
# 1.  Give a hyperlink to a useful post you found on Stack Overflow.  The syntax for making a hyperlink in markdown is `[text here](url here)`.
# 1.  Write a sentence saying what you learned from the linked page.
# 

# ## How to submit this homework
# 
# Go to the Files section on the left, click on the ... next to Homework1.ipynb, and choose *export as .ipynb*.  That IPython notebook is what you should upload on Canvas.

# <a style='text-decoration:none;line-height:16px;display:flex;color:#5B5B62;padding:10px;justify-content:end;' href='https://deepnote.com?utm_source=created-in-deepnote-cell&projectId=9e797310-a62a-4b4e-a010-2ed014f71979' target="_blank">
# <img alt='Created in deepnote.com' style='display:inline;max-height:16px;margin:0px;margin-right:7.5px;' src='data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iODBweCIgaGVpZ2h0PSI4MHB4IiB2aWV3Qm94PSIwIDAgODAgODAiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8IS0tIEdlbmVyYXRvcjogU2tldGNoIDU0LjEgKDc2NDkwKSAtIGh0dHBzOi8vc2tldGNoYXBwLmNvbSAtLT4KICAgIDx0aXRsZT5Hcm91cCAzPC90aXRsZT4KICAgIDxkZXNjPkNyZWF0ZWQgd2l0aCBTa2V0Y2guPC9kZXNjPgogICAgPGcgaWQ9IkxhbmRpbmciIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJBcnRib2FyZCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTEyMzUuMDAwMDAwLCAtNzkuMDAwMDAwKSI+CiAgICAgICAgICAgIDxnIGlkPSJHcm91cC0zIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxMjM1LjAwMDAwMCwgNzkuMDAwMDAwKSI+CiAgICAgICAgICAgICAgICA8cG9seWdvbiBpZD0iUGF0aC0yMCIgZmlsbD0iIzAyNjVCNCIgcG9pbnRzPSIyLjM3NjIzNzYyIDgwIDM4LjA0NzY2NjcgODAgNTcuODIxNzgyMiA3My44MDU3NTkyIDU3LjgyMTc4MjIgMzIuNzU5MjczOSAzOS4xNDAyMjc4IDMxLjY4MzE2ODMiPjwvcG9seWdvbj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0zNS4wMDc3MTgsODAgQzQyLjkwNjIwMDcsNzYuNDU0OTM1OCA0Ny41NjQ5MTY3LDcxLjU0MjI2NzEgNDguOTgzODY2LDY1LjI2MTk5MzkgQzUxLjExMjI4OTksNTUuODQxNTg0MiA0MS42NzcxNzk1LDQ5LjIxMjIyODQgMjUuNjIzOTg0Niw0OS4yMTIyMjg0IEMyNS40ODQ5Mjg5LDQ5LjEyNjg0NDggMjkuODI2MTI5Niw0My4yODM4MjQ4IDM4LjY0NzU4NjksMzEuNjgzMTY4MyBMNzIuODcxMjg3MSwzMi41NTQ0MjUgTDY1LjI4MDk3Myw2Ny42NzYzNDIxIEw1MS4xMTIyODk5LDc3LjM3NjE0NCBMMzUuMDA3NzE4LDgwIFoiIGlkPSJQYXRoLTIyIiBmaWxsPSIjMDAyODY4Ij48L3BhdGg+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMCwzNy43MzA0NDA1IEwyNy4xMTQ1MzcsMC4yNTcxMTE0MzYgQzYyLjM3MTUxMjMsLTEuOTkwNzE3MDEgODAsMTAuNTAwMzkyNyA4MCwzNy43MzA0NDA1IEM4MCw2NC45NjA0ODgyIDY0Ljc3NjUwMzgsNzkuMDUwMzQxNCAzNC4zMjk1MTEzLDgwIEM0Ny4wNTUzNDg5LDc3LjU2NzA4MDggNTMuNDE4MjY3Nyw3MC4zMTM2MTAzIDUzLjQxODI2NzcsNTguMjM5NTg4NSBDNTMuNDE4MjY3Nyw0MC4xMjg1NTU3IDM2LjMwMzk1NDQsMzcuNzMwNDQwNSAyNS4yMjc0MTcsMzcuNzMwNDQwNSBDMTcuODQzMDU4NiwzNy43MzA0NDA1IDkuNDMzOTE5NjYsMzcuNzMwNDQwNSAwLDM3LjczMDQ0MDUgWiIgaWQ9IlBhdGgtMTkiIGZpbGw9IiMzNzkzRUYiPjwvcGF0aD4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+' > </img>
# Created in <span style='font-weight:600;margin-left:4px;'>Deepnote</span></a>
