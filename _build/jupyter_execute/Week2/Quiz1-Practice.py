#!/usr/bin/env python
# coding: utf-8

# # Quiz 1 Practice Exercises
# 
# These questions relate to the [learning objectives](https://canvas.eee.uci.edu/courses/42645/pages/learning-objectives?module_item_id=1612181) for the Quiz 1 (which is scheduled for Thursday of Week 2).
# 
# For each of these topics, we state the learning objective and then give one or more practice exercises.

# ## Documentation
# 
# * Extract basic information from Python documentation (such as the result of using help).

# ### Exercise: 
# Use NumPy's `arange` (look it up using `help`) function to create the NumPy array
# ```
# array([3. , 3.5, 4. , 4.5, 5. , 5.5, 6. , 6.5, 7. , 7.5])
# ```
# 
# What goes wrong if you try to make the list version using Python's built-in `range` function?

# ## Errors
# 
# * Extract basic information from a Python error message.

# ### Exercise:
# 
# There is a function called `zeros` built into NumPy.  The documentation for this function shows the following:
# ```
# Help on built-in function zeros in module numpy:
# 
# zeros(...)
#     zeros(shape, dtype=float, order='C')
#     
#     Return a new array of given shape and type, filled with zeros.
#     
#     Parameters
#     ----------
#     shape : int or tuple of ints
#         Shape of the new array, e.g., ``(2, 3)`` or ``2``.
#     dtype : data-type, optional
#         The desired data-type for the array, e.g., `numpy.int8`.  Default is
#         `numpy.float64`.
#     order : {'C', 'F'}, optional, default: 'C'
#         Whether to store multi-dimensional data in row-major
#         (C-style) or column-major (Fortran-style) order in
#         memory.
# ```

# If we evaluate `np.zeros(3,5)`, we get the following error:
# ```
# ---------------------------------------------------------------------------
# TypeError                                 Traceback (most recent call last)
# <ipython-input-3-25c3f14ba1b8> in <module>
# ----> 1 np.zeros(3,5)
# 
# TypeError: Cannot interpret '5' as a data type
# ```

# * The user was trying to make a 3-by-5 array of zeros.  Do you see what the user did wrong?
# * How should the code be corrected?

# ## Data types
# 
# * Choose from among list, tuple, range, set, NumPy array, depending on which is an appropriate data type for a particular task.

# ### Exercise:
# 
# For each pair of data types, give an advantage/disadvantage between them.

# ## Convert
# 
# * Convert between different data-types using syntax like `np.array(my_tuple)`.

# ### Exercise:
# 
# What is the difference between the following?  Give a specific example, and explain how the results are different. `str(list(x))` vs `list(str(x))`.  Is it possible for these two to produce the same output?

# ## Replace for loops
# 
# * Replace code written with a for-loop with code using list comprehension.

# ### Exercise:
# 
# Rewrite the following using list comprehension.  (What is an example of a value of `my_list` for which this code makes sense?)

# In[ ]:


new_list = []
for x in my_list:
    if len(x) > 2:
        new_list.append(x)


# Do the same thing for the following.

# In[ ]:


new_list = []
for x in my_list:
    if len(x) > 2:
        new_list.append(x)
    else:
        new_list.append(0)


# Do the same for the following.  Use `my_list[:5]` to refer to the first 5 elements in `my_list`.

# In[ ]:


new_list = []
for i in range(5):
    new_list.append(my_list[i]**2)


# ## Random
# 
# * Generate random numbers (real numbers or integers) in NumPy using `default_rng()`.
# 
# [Relevant section in the course notes](https://christopherdavisuci.github.io/UCI-Math-10-W22/Week2/rng.html)

# ### Exercise:
# 
# Using NumPy's `default_rng()` and the method `random`, make a length 10 NumPy array of random real numbers (not integers) uniformly distributed between 1 and 4.  (To get full credit, you must use these items, and you must make the array in an efficient manner, not for example using a for loop or even a list comprehension.)

# ## list comprehension sublist
# 
# * Find elements in a list or NumPy array which satisfy a given condition using list comprehension.	
# 

# ### Exercise:
# 
# For your random array made above, using list comprehension, get the sublist of elements which `round` to 3.  (There is probably a better way to do this using NumPy methods, but the point of this is to practice with list comprehension.  We haven't discussed the `round` function which is a built-in function Python.  You can look up its documentation or just try it out!)

# ## Defining a function
# 
# * Write code defining a function

# ### Exercise:
# 
# Write code which takes as input a list `my_list` and as output returns:
# * the word *yes* if the length of `my_list` is strictly bigger than 3;
# * the word *no* if the length of `my_list` is less than or equal to 3;
# * the phrase *not a list* if `my_list` is not a list.
# **Hints**.  Make sure that you are returning values, not printing values.  Use `isinstance` to check if `my_list` is a list, like from discussion section on Thursday.

# ## Counting in a NumPy array
# 
# * Count elements in a NumPy array satisfying a given condition using a Boolean array and `np.count_nonzero`.
# 
# [Relevant section in the course notes](https://christopherdavisuci.github.io/UCI-Math-10-W22/Week2/BooleanArray.html)

# ### Exercise:
# 
# For the random NumPy array you made above, count how many of its elements are strictly bigger than 3.  Use the following strategy: first convert into a Boolean array which is `True` where the elements are strictly bigger than 3, and then use `np.count_nonzero`.

# ## Random simulation
# 
# * Estimate a probability using a random simulation and the formula successes/experiments.

# ### Exercise:
# 
# Using a random simulation and NumPy, estimate the following: if you choose a random real number between 1 and 4, what is the probability that the number is strictly bigger than 3?  (Of course we could instead find this probability exactly.  Use enough "experiments" that your probability estimate is accurate to at least 3 digits.)

# <a style='text-decoration:none;line-height:16px;display:flex;color:#5B5B62;padding:10px;justify-content:end;' href='https://deepnote.com?utm_source=created-in-deepnote-cell&projectId=a56b40ff-804d-4506-baf3-b9c9f18df909' target="_blank">
# <img alt='Created in deepnote.com' style='display:inline;max-height:16px;margin:0px;margin-right:7.5px;' src='data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iODBweCIgaGVpZ2h0PSI4MHB4IiB2aWV3Qm94PSIwIDAgODAgODAiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8IS0tIEdlbmVyYXRvcjogU2tldGNoIDU0LjEgKDc2NDkwKSAtIGh0dHBzOi8vc2tldGNoYXBwLmNvbSAtLT4KICAgIDx0aXRsZT5Hcm91cCAzPC90aXRsZT4KICAgIDxkZXNjPkNyZWF0ZWQgd2l0aCBTa2V0Y2guPC9kZXNjPgogICAgPGcgaWQ9IkxhbmRpbmciIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJBcnRib2FyZCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTEyMzUuMDAwMDAwLCAtNzkuMDAwMDAwKSI+CiAgICAgICAgICAgIDxnIGlkPSJHcm91cC0zIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxMjM1LjAwMDAwMCwgNzkuMDAwMDAwKSI+CiAgICAgICAgICAgICAgICA8cG9seWdvbiBpZD0iUGF0aC0yMCIgZmlsbD0iIzAyNjVCNCIgcG9pbnRzPSIyLjM3NjIzNzYyIDgwIDM4LjA0NzY2NjcgODAgNTcuODIxNzgyMiA3My44MDU3NTkyIDU3LjgyMTc4MjIgMzIuNzU5MjczOSAzOS4xNDAyMjc4IDMxLjY4MzE2ODMiPjwvcG9seWdvbj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0zNS4wMDc3MTgsODAgQzQyLjkwNjIwMDcsNzYuNDU0OTM1OCA0Ny41NjQ5MTY3LDcxLjU0MjI2NzEgNDguOTgzODY2LDY1LjI2MTk5MzkgQzUxLjExMjI4OTksNTUuODQxNTg0MiA0MS42NzcxNzk1LDQ5LjIxMjIyODQgMjUuNjIzOTg0Niw0OS4yMTIyMjg0IEMyNS40ODQ5Mjg5LDQ5LjEyNjg0NDggMjkuODI2MTI5Niw0My4yODM4MjQ4IDM4LjY0NzU4NjksMzEuNjgzMTY4MyBMNzIuODcxMjg3MSwzMi41NTQ0MjUgTDY1LjI4MDk3Myw2Ny42NzYzNDIxIEw1MS4xMTIyODk5LDc3LjM3NjE0NCBMMzUuMDA3NzE4LDgwIFoiIGlkPSJQYXRoLTIyIiBmaWxsPSIjMDAyODY4Ij48L3BhdGg+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMCwzNy43MzA0NDA1IEwyNy4xMTQ1MzcsMC4yNTcxMTE0MzYgQzYyLjM3MTUxMjMsLTEuOTkwNzE3MDEgODAsMTAuNTAwMzkyNyA4MCwzNy43MzA0NDA1IEM4MCw2NC45NjA0ODgyIDY0Ljc3NjUwMzgsNzkuMDUwMzQxNCAzNC4zMjk1MTEzLDgwIEM0Ny4wNTUzNDg5LDc3LjU2NzA4MDggNTMuNDE4MjY3Nyw3MC4zMTM2MTAzIDUzLjQxODI2NzcsNTguMjM5NTg4NSBDNTMuNDE4MjY3Nyw0MC4xMjg1NTU3IDM2LjMwMzk1NDQsMzcuNzMwNDQwNSAyNS4yMjc0MTcsMzcuNzMwNDQwNSBDMTcuODQzMDU4NiwzNy43MzA0NDA1IDkuNDMzOTE5NjYsMzcuNzMwNDQwNSAwLDM3LjczMDQ0MDUgWiIgaWQ9IlBhdGgtMTkiIGZpbGw9IiMzNzkzRUYiPjwvcGF0aD4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+' > </img>
# Created in <span style='font-weight:600;margin-left:4px;'>Deepnote</span></a>
