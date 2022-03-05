#!/usr/bin/env python
# coding: utf-8

# # Week 8 Wednesday
# 
# [Yuja recording](https://uci.yuja.com/V/Video?v=4446584&node=14938258&a=1900985263&autoplay=1)
# 
# Before the recording, at the board we went over some different components related to Neural Networks and PyTorch, and especially we went over an example of performing gradient descent.
# 
# The goal of today's class is to get more comfortable with the various components involved in building and training a neural network using PyTorch.

# In[1]:


import numpy as np
import torch
from torch import nn
from torchvision.transforms import ToTensor


# ## Gradient descent

# Gradient descent can be used to try to find a minimum of any differentiable function.  (Often it will only find a local minimum, not a global minimum, even if a global minimum exists.)  We usually use gradient descent for very complicated functions, but here we give an example of performing gradient descent to attempt to find a minimum of the function
# 
# $$
# f(x,y) = (x-3)^2 + (y+2)^2 + 8.
# $$
# 
# We call this function `loss_fn` so that the syntax is the same as what we're used to in PyTorch.

# In[2]:


loss_fn = lambda t: (t[0] - 3)**2 + (t[1] + 2)**2 + 8 


# To perform gradient descent, you need to begin with an initial guess.  We guess (10,10) and then gradually adjust this, hoping to move towards a minimum.  Notice the decimal point after 10... this is a shortcut for telling PyTorch that these should be treated as floats.

# In[3]:


a = torch.tensor([10.,10], requires_grad=True)
a


# In[4]:


loss_fn([10,10])


# In[5]:


loss_fn(a)


# In[6]:


type(loss_fn)


# Because we specified `requires_grad=True` as a keyword argument, we will be able to find gradients of computations involving `a`.  There isn't any gradient yet because we haven't computed one.

# In[7]:


a.grad


# Here we define a stochastic gradient descent optimizer like usual in PyTorch.  The first input is usually something like `model.parameters()`.  Here we try to use `a` as the first argument.  That is almost right, but we need to put it in a list (or some other type of *iterable*).

# In[8]:


optimizer = torch.optim.SGD(a, lr = 0.1)


# In[9]:


optimizer = torch.optim.SGD([a], lr = 0.1)


# In[10]:


loss = loss_fn(a)


# This next `optimizer.zero_grad()` is not important yet, but it is good to be in the habit, because otherwise multiple gradient computations will accumulate, and we want to start over each time.

# In[11]:


optimizer.zero_grad()


# In[12]:


type(loss)


# Next we compute the gradient.  This typically uses an algorithm called *backpropagation*, which is where the name `backward` comes from.

# In[13]:


loss.backward()


# In[14]:


a


# Now the `grad` attribute of `a` has a value.  You should be able to compute this value by hand in this case, since our `loss_fn` is so simple.

# In[15]:


a.grad


# Now we replace add a multiple (the learning rate `lr`) of the negative gradient to `a`.  Again, you should be able to compute this by hand in this case.  The formula is
# 
# $$
# a \leadsto a - lr \cdot \nabla
# $$

# In[16]:


optimizer.step()


# In[17]:


a


# In[18]:


loss_fn = lambda t: (t[0] - 3)**2 + (t[1] + 2)**2 + 8 


# Notice how the value of `a` is approaching the minimum (3,-2), and notice how `loss` is approaching the minimum of our `loss_fn`, which is 8.  (The only reason we're using the terms `loss` and `loss_fn` is because those are the terms we usually use in PyTorch.  In this case, `loss_fn` is just an ordinary two-variable function like from Math 2D which we are trying to minimize.)

# In[19]:


epochs = 20
a = torch.tensor([10.,10], requires_grad=True)
optimizer = torch.optim.SGD([a], lr = 0.1)
for i in range(epochs):
    loss = loss_fn(a)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("Epoch " + str(i))
    print(a)
    print(loss)
    print("")


# If we want `a` to approach the minimum (3,-2) faster, we can make the learning rate bigger, but here is an example of what can go wrong if we make the learning rate too big.

# In[20]:


epochs = 20
a = torch.tensor([10.,10], requires_grad=True)
optimizer = torch.optim.SGD([a], lr = 10)
for i in range(epochs):
    loss = loss_fn(a)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("Epoch " + str(i))
    print(a)
    print(loss)
    print("")


# Here is an example for what seems to be a good choice of `lr`.

# In[21]:


epochs = 20
a = torch.tensor([10.,10], requires_grad=True)
optimizer = torch.optim.SGD([a], lr = 0.25)
for i in range(epochs):
    loss = loss_fn(a)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("Epoch " + str(i))
    print(a)
    print(loss)
    print("")

