#!/usr/bin/env python
# coding: utf-8

# # Wednesday Lecture

# ## Practice Exercises
# 
# Try to solve the following:
# 
# 1.  Print the word *hello* 5 times.  Use a for loop.
# 2.  Make a list containing the word *hello* 5 times.  Use list comprehension.
# 3.  Make a list containing the even numbers from 0 to 20 (inclusive).  Save it with the name `even_list`.
# 4.  Make a list containing the squares of the numbers in `even list`.
# 5.  (This is probably too hard.  Try to find the answer by Google searching.)  Write code that turns an integer like `8245313` into a list of integers, like `[8,2,4,5,3,1,3]`.
# 
# [Zoom recording from lecture on 1/5/2022](https://uci.zoom.us/rec/play/iYHrTAH6uuXDXghcxkT0-11CiVtobYMjXwTdz7Yg5Cs-DTTI9Fp-Fc93541ehy_R1BYe7PsKctpp_2sQ.P7YkKZ5pxC5j1S9f)

# ## Answer 1

# In[1]:


for i in range(5):
    print('hello')


# Since the variable name `i` is never used in the above code, some people would replace it with an underscore `_`.  I don't necessarily recommend doing this, but it is good to recognize it if you see it in somebody else's code.

# In[2]:


for _ in range(5):
    print('hello')


# ## Answer 2
# 
# I don't have a precise definition for what makes the following list comprehension, but the most important aspects are:
# * Square brackets (that makes it a list)
# * The thing you want to go into the list (in this case, the string `"hello"`)
# * How many repetitions/iterations there should be.

# In[3]:


["hello" for x in range(5)]


# Some other approaches which are not list comprehension.
# 
# You can combine two lists using `+`:

# In[4]:


[1,3,5]+["hello" for i in range(2)]


# So it makes sense that if you use `*`, that combines multiple copies of the same list:

# In[5]:


["hello"]*5


# ## Answer 3
# 
# Here is one approach, using list comprehension:

# In[6]:


even_list = [x for x in range(0,22,2)]
even_list


# Here is another approach, which converts the `range` object to a `list` object directly:

# In[7]:


even_list = list(range(0,22,2))
even_list


# ## Timing in Deepnote notebooks
# 
# The two approaches seem comparable in terms of speed.

# In[8]:


get_ipython().run_cell_magic('timeit', '', '[x for x in range(0,2000000,2)]')


# In[9]:


get_ipython().run_cell_magic('timeit', '', 'list(range(0,2000000,2))')


# ## Answer 4
# 
# Again using list comprehension:

# In[10]:


[i**2 for i in even_list]


# ## Answer 5
# 
# This question was difficult, but I hope each step in the answer makes sense.

# In[11]:


n = 8245313
n_str = str(n)
n_list = list(n_str)
n_final = [int(x) for x in n_list]
n_final


# If you want, you can do the whole thing at once:

# In[12]:


n = 8245313
n_final = [int(x) for x in list(str(n))]
n_final


# ## Defining a function in Python
# 
# Here is the basic syntax.

# In[13]:


def f(x):
    return x**2


# In[14]:


f(10)


# It works equally well with multiple variables and multiple lines of code (just make sure they're all indented).

# In[15]:


def g(x,y):
    z = x+y
    return ["hello" for i in range(z)]


# In[16]:


g(2,1)


# In[17]:


def h(n):
    i = n+2
    return i**2


# In[18]:


h(10)


# In[19]:


h(2)+2


# It is okay to have the function not return anything, but then there is no output to use.

# In[20]:


def j(n):
    for i in range(n):
        print("hi")


# Maybe it looks like there is an output:

# In[21]:


z = j(5)


# But we can see that the "output" is a `NoneType`:

# In[22]:


type(z)


# This can be a common source of mistakes.  For example, maybe the author of the next code wants `v` to be equal to `[3,3,2,5]`, but it isn't.

# In[23]:


w = [3,3,2]
v = w.append(5)


# In[24]:


v


# In[25]:


type(v)


# The expression `w.append(5)` did not generate an output; instead, it changed the value of `w`.

# In[26]:


w

