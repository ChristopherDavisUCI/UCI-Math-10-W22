#!/usr/bin/env python
# coding: utf-8

# # Making a for loop more Pythonic
# 
# We'll start with a piece of Python code that is correct but which can be improved.  The code can be more concise (in fact, some of the code does not do anything).  We'll see at the end how to do the same thing using **list comprehension**.  Using list comprehension instead of an explicit for loop probably is not much more efficient, but it is much more *Pythonic*.
# 
# [Zoom recording from class on 1/4/2022](https://uci.zoom.us/rec/play/NIZp8kg2qyM5TdkKvWB3QK8I-sap1KhXmjwm1ePTNNKuUsr4POfO_Qrv4aWIstl0p5mVtxBOh59SoJdU.7t5v5MV3wE1yktKw)

# ## Starter code
# 
# Try to read this code, step by step, and figure out what it does.

# In[1]:


my_list = [3.14,10,-4,3,5,5,-1,20.3,14,0]


# In[2]:


new_list = []
for i in range(0,len(my_list)):
    if my_list[i] > 4:
        new_list.append(my_list[i])
    else:
        new_list = new_list


# In[3]:


new_list


# ## Revision 1
# 
# Remove the `else` statement, which wasn't doing anything.

# In[4]:


new_list = []
for i in range(0,len(my_list)):
    if my_list[i] > 4:
        new_list.append(my_list[i])


# In[5]:


new_list


# ## Revision 2
# 
# Iterate through the elements in `my_list`, not the numbers from 0 to 9.

# In[6]:


new_list = []
for x in my_list:
    if x > 4:
        new_list.append(x)


# In[7]:


new_list


# ## Final version
# 
# Using list comprehension.

# In[8]:


new_list = [x for x in my_list if x > 4]


# In[9]:


new_list


# ## Miscellaneous examples
# 
# The above portion is what you should focus on, but we briefly touched on lots of quick examples.  Here are some of them.

# for loops are the standard way to repeat something.

# In[10]:


for i in range(6):
    print("hi")


# Notice that the right endpoint is not included.  For example, in the following, we never set `i` equal to `5`.

# In[11]:


for i in range(0,5):
    print(i)


# The significance of indentation after a for loop:

# In[12]:


for i in range(0,5):
    print("i")
print(i)


# In[13]:


for i in range(0,5):
    print("i")
    print(i)


# You don't need the zero: `range(0,5)` is the same as `range(5)`.

# In[14]:


for i in range(5):
    print(i)


# This might remind you of the colon operator from Matlab, although the order is rearranged (and in Matlab, the right endpoint is included).

# In[15]:


for i in range(0,20,3):
    print(i)


# In[16]:


for i in range(1,20,3):
    print(i)


# Here is the error message that shows up if you forget the colon:

# In[17]:


for i in range(0,5)
    print("i")
    print(i)


# `range` is sort of like a `list`, but it's not the same data type.

# In[18]:


y = range(5)


# In[19]:


type(y)


# In[20]:


z = range(0,20,3)


# In[21]:


type(z)


# Converting from a `range` object to a `list` object:

# In[22]:


list(z)


# Converting from a `str` (string) object to a `list` object.

# In[23]:


list("chris")


# As one example of an advantage of `range` over `list`, you can create huge `range` objects that would never fit in the computer's memory if literally made as a list.  Also, notice that you do not make exponents in Python using the caret symbol `^`; instead you make exponents using `**`.

# In[24]:


z = range(0,10**20)


# Practice with `append`.

# In[25]:


z = list("chris")


# In[26]:


z.append(15)


# In[27]:


z


# In[28]:


help(z.append)


# One way to insert something into a list at a specific position (as opposed to `append` which inserts it at the end).  You can see again here that right endpoints are not included.

# In[29]:


z[0:3]+["hello"]+z[3:]


# Another way, using `insert`.

# In[30]:


help(z.insert)


# In[31]:


z.insert(3,"goodbye")


# In[32]:


z


# Notice that sometimes nothing gets displayed after a command, like the following.  That is a clue that possibly the original object got changed.

# In[33]:


z.append(12)


# In[34]:


z


# Practice with list comprehension.

# In[35]:


my_list


# In[36]:


new_list = [x for x in my_list if x > 4]


# In[37]:


new_list


# In[38]:


[x if x > 4 else 100 for x in my_list]


# In[39]:


[x+1 if x > 4 else x-1 for x in my_list]


# In[40]:


[x+1 for x in my_list if x > 4]


# Python is pretty strict about where you put the `if`.  Notice the subtle differences between the following.

# In[41]:


[x if x > 4 else 100 for x in my_list]


# In[42]:


[x for x in my_list if x > 4 else 100]


# And in the following, which seems almost like the opposite.

# In[43]:


[x+1 for x in my_list if x > 4]


# In[44]:


[x+1 if x > 4 for x in my_list]


# Another example of list comprehension.  Here we get the words in the sentence that start with the letter h.

# In[45]:


text = "hello there how are you doing?  I am fine."


# In[46]:


list_of_words = text.split()


# In[47]:


list_of_words


# In[48]:


[word for word in list_of_words if word[0] == "h"]


# There's nothing special about the use of the word `word`, it could be any variable name.

# In[49]:


[i for i in list_of_words if i[0] == "h"]


# Notice that numbering in Python starts at 0.

# In[50]:


list_of_words[0]


# In[51]:


list_of_words[1]


# In[52]:


list_of_words[len(list_of_words)]

