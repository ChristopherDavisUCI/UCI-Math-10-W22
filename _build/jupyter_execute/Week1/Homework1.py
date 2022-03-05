#!/usr/bin/env python
# coding: utf-8

# # Homework 1
# 
# **Update** 1/3/22: I was wrong in class when I said you should duplicate this into the team space.  I think you need to duplicate it into your personal workspace.  You can then share with collaborators at the top right (give them *edit* access if you want them to be able to edit the code).
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
# Read Subsection 8.1 and Subsection 8.2 of the attached article, *50 Years of Data Science*.  (To access this article, click the Folder icon on the left, and then you should see this pdf file listed.)  These sections describe six proposed divisions of "Greater Data Science".  If you were to rank these six divisions, from "Most important to study in Math 10" to "Least important to study in Math 10", how would you rank them?
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

# In[ ]:


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
