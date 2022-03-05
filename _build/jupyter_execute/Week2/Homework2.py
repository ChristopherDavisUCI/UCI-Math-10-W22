#!/usr/bin/env python
# coding: utf-8

# # Homework 2
# 
# List your name and the names of any collaborators at the top of this notebook.
# 
# (Reminder: It's encouraged to work together; you can even submit the exact same homework as another student or two students, but you must list each other's names at the top.)

# ## Exercise 1
# 
# Using the Python function `getsizeof` (hint: you need to import a library to get access to it), find out how much space in memory the following take:
# 
# 1.  The integer 0.
# 1.  An empty list.
# 1.  A list containing the integers 1,2,3.
# 1.  A list containing the strings 1,2,3.
# 1.  A NumPy array containg the integers (or floats) 1,2,3.  (We haven't talked about the data type of elements inside NumPy arrays, so don't worry about that.)
# 1.  The `range` object `range(0,10**100,3)`.
# 

# ## Exercise 2
# 
# Write a function `make_arr` that takes as input a positive integer `n`, and as output returns a length `n` NumPy array containing random integers between 1 and 100 (inclusive).  Use the `integers` method of an object produced by NumPy's `default_rng()`.

# ## Exercise 3
# 
# Using your function `make_arr`, create a length one million array of random integers between 1 and 100 (inclusive).  Save this array with the variable name `arr`.

# ## Exercise 4
# 
# Compute the reciprocals of each element in `arr` by evaluating `1/arr`.  How long does it take?  (Use `%%timeit`.)

# ## Exercise 5
# 
# Convert `arr` into a list called `my_list`, and then use list comprehension to compute the reciprocals of each element in `my_list`.  Time how long this takes using `%%timeit`.  (Don't include the conversion to a list in the `%%timeit` cell; do the conversion before.)

# ## Exercise 6
# 
# In a markdown cell, indicate how the times compare for these two methods.

# ## Exercise 7
# 
# What proportion of the elements in `arr` are equal to 100?  Answer this question a few different ways; all these answers should be equal and should be very close to `0.01`).
# 
# 1.  Use `my_list` that you created above and the `count` method of a list to determine how often 100 occurs.  Then divide by the total length.  (To get the total length, use `len`, don't type out the length explicitly.)
# 
# 1. Using list comprehension, make a list containing all the elements of `arr` which are equal to 100 (you don't need to use `my_list`; just pretend `arr` is a list and everything will work fine).  Then compute the length of this new list divided by the length of `arr`.  (This isn't a great strategy; it is mostly an excuse to practice with list comprehension.)
# 
# 2.  Make a Boolean array which is `True` whereever `arr` is 100 and which is `False` everywhere else.  Then use the NumPy function `np.count_nonzero`, and then divide by the length of `arr`.
# 
# 3.  Convert the array into a pandas Series, and then apply the method `.value_counts()`, then compute `s[100]`, where `s` represents the output of `.value_counts()`, then divide by the length of `arr`.

# ## Exercise 8
# 
# Repeat each of the previous computations, this time using `%%timeit` to see how long they take.  In a markdown cell, report what answers you get.  (For the pandas Series part, convert to the pandas Series outside of the `%%timeit` cell... It gives NumPy an unfair advantage if that conversion is included in the timing portion.)

# ## Exercise 9
# 
# In a markdown cell, answer the following question:  Was one of the four methods significantly faster than the rest?  Was one of the four methods significantly slower than the rest?

# ## Exercise 10
# 
# Many of these exercises are about how to make various operations run faster by choosing appropriate data types.  What do you think is one of the main reasons that this is relevant to data science?

# ## Submission
# 
# Download the .ipynb file for this notebook (click on the folder icon to the left, then the ... next to the file name) and upload the file on Canvas.
