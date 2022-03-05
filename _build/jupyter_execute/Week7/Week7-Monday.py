#!/usr/bin/env python
# coding: utf-8

# # PyTorch and Neural Networks
# 
# [YuJa video](https://uci.yuja.com/V/Video?v=4372302&node=14708610&a=6152974&autoplay=1) from lecture

# In[1]:


from tqdm.std import tqdm, trange
from tqdm import notebook
notebook.tqdm = tqdm
notebook.trange = trange

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader


# The first four lines
# ```
# from tqdm.std import tqdm, trange
# from tqdm import notebook
# notebook.tqdm = tqdm
# notebook.trange = trange
# ```
# are an ad hoc suggestion I read on a [Deepnote forum](https://community.deepnote.com/c/bugs/tqdm-call-causes-importerror) to help prevent a minor error (the error is just because of a progress bar, nothing important).  Don't worry about them.

# In[2]:


# Load the data
training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)


# YouTube video on *Neural Networks* from 3Blue1Brown.  Recommended clips:
# * 2:42-5:30
# * 8:40-12:40
# 
# <iframe width="560" height="315" src="https://www.youtube.com/embed/aircAruvnKk" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

# In[3]:


training_data.data.shape


# In[4]:


training_data.data[13]


# Using the default color map.

# In[5]:


fig, ax = plt.subplots()
ax.imshow(training_data.data[13])


# Using the `binary` color map.

# In[6]:


fig, ax = plt.subplots()
ax.imshow(training_data.data[13],cmap='binary')


# Switching to the reversed color map, by appending `_r` to the end of the `cmap` name.

# In[7]:


fig, ax = plt.subplots()
ax.imshow(training_data.data[13],cmap='binary_r')


# That should correspond to the number 6.

# In[8]:


training_data.targets[13]


# To convert a length-one PyTorch tensor to a single number, we use `.item()`.

# In[9]:


training_data.targets[13].item()


# In the 3Blue1Brown video, the grid of image pixels gets "flattened" out into a length 784 vector.  PyTorch has a standard way of doing this, using `nn.Flatten()`.

# In[10]:


flatten = nn.Flatten()


# In[11]:


flatten(training_data.data[:13])


# In[12]:


training_data.data[:13].shape


# In[13]:


flatten(training_data.data[:13]).shape


# PyTorch uses many elements of *Object Oriented* programming.  In the following, we are defining a new type of object called `ThreeBlue`.  You don't need to understand all of the details; we will try to make clear what you should understand and what is less important for us, for example, in the class Learning Objectives.
# 
# The name `__init__` defined below begins and ends with two underscores.  These sorts of methods are called *dunder methods* for double underscore.  They lie in the background of many Python operations.  For example, when you add two objects together, that is often (always?) secretly using the `__add__` dunder method.  The `__init__` method is a method that is called when a new object is created of this class.
# 
# The class `ThreeBlue` below is the beginning of a tiny neural network.

# In[14]:


class ThreeBlue(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()

    def forward(self,x):
        y = self.flatten(x)
        return y


# In[15]:


mon = ThreeBlue()


# I don't totally understand what the `__main__` means in the following; the important part is that the object `mon` we have instantiated is of type `ThreeBlue`.

# In[16]:


type(mon)


# We defined a `flatten` attribute in the `__init__` method above, so that is why `mon` has a `flatten` attribute.  We can use it just like the `flatten` above.

# In[17]:


mon.flatten


# In[18]:


mon.flatten(training_data.data[:13])


# Here is a slightly bigger neural network.

# In[19]:


class ThreeBlue(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(784,10)
        )

    def forward(self,x):
        y = self.flatten(x)
        z = self.layers(y)
        return z


# In[20]:


mon = ThreeBlue()


# The entries in `training_data.data` are integers between 0 and 255.  We want floats instead, and numbers between 0 and 1 is more natural anyway, so we can fix the following error by dividing by 255.  This can also be accomplished using the `ToTensor` function that is written above (we will see it later).

# In[21]:


mon(training_data.data)


# In[22]:


training_data.data.numpy().max()


# In[23]:


training_data.data.max()


# In[24]:


mon(training_data.data/255)


# Notice that we never explicitly call the `forward` method of `mon`.  PyTorch is calling this for us in the background; we should not call it directly.
# 
# Here is the input shape.

# In[25]:


training_data.data.shape


# Here is the output shape.

# In[26]:


mon(training_data.data/255).shape


# Think of a neural network as a function.  For each input data point for MNIST handwritten digits, we want 10 outputs.  After rescaling, we can think of these 10 outputs as probabilities for each possible true value of the digit.
# 
# In the current version, some of these 10 numbers are negative, so we can't think of them directly as probabilities.

# In[28]:


mon(training_data.data[13:14]/255)


# But we can at least find the index of the largest number, which could be our predicted digit.  (Think of `dim=1` as analogous to `axis=1` from pandas and NumPy.)

# In[29]:


mon(training_data.data[13:14]/255).argmax(dim=1)


# Another option is that we could rescale these numbers, using the sigmoid function, then they will at least be between 0 and 1, so they can be interpreted as probabilities.  Notice how the original outputs were clustered around 0, and the new outputs are clustered around 0.5.

# In[32]:


sigmoid = nn.Sigmoid()
sigmoid(mon(training_data.data[13:14]/255))


# Another option, which we will actually not use very often, is to apply a function called Softmax, which not only scales the numbers into the correct range, but also makes them sum to 1.

# In[34]:


softmax = nn.Softmax(dim=1)
softmax(mon(training_data.data[13:14]/255))


# In[35]:


softmax(mon(training_data.data[13:14]/255)).sum()

