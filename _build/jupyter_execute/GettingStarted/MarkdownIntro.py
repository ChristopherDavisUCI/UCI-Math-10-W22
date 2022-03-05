#!/usr/bin/env python
# coding: utf-8

# # Markdown Introduction
# 
# This notebook is meant to introduce some basic functionality of markdown in Deepnote.
# 
# If you want to make edits in this notebook, click the *Duplicate* button at the top-right.
# 
# There are two basic types of cells, *code cells* and *markdown cells*.  This content is being written in a markdown cell, whereas the next part, starting with `a = 5`, is an example of a code cell.

# In[1]:


a = 5
print(a+2)


# To execute a code cell, put your cursor in it (the cursor doesn't have to be at the end of the cell) and hit `shift+return`.  It might take about 30 seconds to start the hardware.  For this reason, I often run a short cell like `2+2` right when I open a notebook, even before I've decided what to do.

# Here is a basic [guide to markdown](https://www.markdownguide.org/basic-syntax).  (**Warning**: there are many different flavors of markdown, and while the basic functionality is the same, things that work on GitHub, for example, might not work on Deepnote.)
# 
# As an example, to make the above text, I used the following:
# ```
# Here is a basic [guide to markdown](https://www.markdownguide.org/basic-syntax).  (**Warning**: there are many ...
# ```
# <br>
# 
# To make a short code example, like `pd.read_csv("penguins.csv")`, surround it in backticks `. On my keyboard, the backtick symbol is above the tab key.  (Backtick is not the same as apostrophe.)  To make a longer code block, surround it by triple backticks.  
# 
# ```
# for i in range(6):
#     print(i)
# ```
# 
# Even though the for loop looks like a code cell, it was written in markdown, so you can't execute it.  If you want to see how it was made, double-click anywhere in this markdown cell.
