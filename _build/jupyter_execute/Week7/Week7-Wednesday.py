#!/usr/bin/env python
# coding: utf-8

# # PyTorch and Neural Networks 2
# 
# [YuJa recording of lecture](https://uci.yuja.com/V/Video?v=4417722&node=14870050&a=1646074732&autoplay=1)
# 
# Topics mentioned at the board (not in this notebook):
# * Importance of using activation functions to break linearity.
# * Common choices of activation functions: sigmoid and relu.
# * Concept of *one hot encoding*.

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


# Second YouTube video on *Neural Networks* from 3Blue1Brown.  This video is on *gradient descent*.  Recommended clips:
# * 0:25-1:24
# * 3:18-4:05
# * 5:15-7:50
# 
# <iframe width="560" height="315" src="https://www.youtube.com/embed/IHZwWFHWa-w" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

# This is what we finished with on Monday:

# In[3]:


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


# We instantiate an object in this class as follows.

# In[4]:


wed = ThreeBlue()


# In class (see the YuJa recording above), we gradually built up to the following code.  It was designed to match the 3Blue1Brown video's neural network.

# In[7]:


class ThreeBlue(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(784,16),
            nn.Sigmoid(),
            nn.Linear(16,16),
            nn.Sigmoid(),
            nn.Linear(16,10),
            nn.Sigmoid()
        )

    def forward(self,x):
        x = x/255
        y = self.flatten(x)
        z = self.layers(y)
        return z


# In[8]:


wed = ThreeBlue()


# Here are the weights and biases for this neural network.  When we talk about fitting or training a neural network, we mean adjust the weights and biases to try to minimize some loss function.

# In[10]:


for p in wed.parameters():
    print(p.shape)


# In[11]:


for p in wed.parameters():
    print(p.numel())


# Notice that this is the same 13002 number which appeared in the 3Blue1Brown videos.

# In[14]:


sum([p.numel() for p in wed.parameters()])


# You can even do the same thing without the square brackets.  This is secretly using a *generator expression* instead of a list comprehension.

# In[15]:


sum(p.numel() for p in wed.parameters())


# In[16]:


wed


# In the line that begins `self.layers = ` above, we were specifying that each ThreeBlue object should have a `layers` attribute.  Here is that attribute for the case of `wed`.

# In[20]:


wed.layers


# You can access for example the second element of `wed.layers` using subscripting, `wed.layers[2]`.

# In[21]:


wed.layers[2]


# In[22]:


wed.layers[2].weight.shape


# In[23]:


wed.layers[2].bias.shape


# On Monday, we were having to divide by 255 each time we input data to our neural network.  Today, we've put that step directly into the `forward` method of the neural network; it's the line `x = x/255`.

# In[29]:


wed(training_data.data)[:3]


# In[30]:


y_pred = wed(training_data.data)


# In[31]:


training_data.targets[:3]


# To match the 3Blue1Brown video, we are going to convert the targets, which are integers like `5`, into length 10 vectors like `[0,0,0,0,0,1,0,0,0,0]`.  This procedure is called *one-hot encoding*, and it also exists in scikit-learn.

# In[32]:


from torch.nn.functional import one_hot


# In[33]:


one_hot(training_data.targets[:3], num_classes=10).to(torch.float)


# In[34]:


y_true = one_hot(training_data.targets, num_classes=10).to(torch.float)


# In[35]:


y_true.shape


# Using Mean-Squared Error on the probabilities for this classification problem is not considered the best approach, but it is easy to understand, and we will follow this approach for now to match the 3Blue1Brown video.

# In[36]:


loss_fn = nn.MSELoss()


# Here is the performance of the randomly initialized model.  The output of this sort of loss function is not so easy to analyze in isolation.  The important thing is that if we can lower this number, then the model is performing better (on the training data).

# In[38]:


loss_fn(y_pred, y_true)


# Here we try to find better weights and biases using *gradient descent*.  Try to get comfortable with these steps (they can take some time to internalize).

# In[39]:


optimizer = torch.optim.SGD(wed.parameters(), lr=0.1)


# There aren't yet any gradients associated with the parameters of the model (the weights and biases).

# In[40]:


for p in wed.parameters():
    print(p.grad)


# In[41]:


loss = loss_fn(y_pred, y_true)


# Still no gradients.

# In[42]:


for p in wed.parameters():
    print(p.grad)


# In[43]:


loss.backward()


# The line `loss.backward()` told PyTorch to compute the gradients of the loss calculation with respect to the 13002 weights and biases.

# In[44]:


for p in wed.parameters():
    print(p.grad)


# The next line adjusts the weights and biases by adding a multiple of the negative gradient.  (We are trying to minimize the loss, and the gradient points in the direction of fastest ascent, and the negative gradient points in the direction of fastest descent.)  The multiple we use is determined by the *learning rate* `lr` that we specified when we created the optimizer above.

# In[45]:


optimizer.step()


# In[46]:


wed(training_data.data)[:3]


# We now want to repeat that procedure.  Here we will repeat it 10 times, but often we will want to repeat it many more times.  What we hope is that the loss value is decreasing.

# In[47]:


epochs = 10

for i in range(epochs):
    y_true = one_hot(training_data.targets, num_classes=10).to(torch.float)
    y_pred = wed(training_data.data)
    loss = loss_fn(y_true,y_pred)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(loss)


# An important thing to point out is that if we run the same code again, we won't be starting back at the beginning.  Each time we run this training procedure, it will begin where the last training procedure left off.

# In[48]:


epochs = 100

for i in range(epochs):
    y_true = one_hot(training_data.targets, num_classes=10).to(torch.float)
    y_pred = wed(training_data.data)
    loss = loss_fn(y_true,y_pred)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if i%2 == 0:
        print(loss)


# Notice how the loss is steadily decreasing.  That's the best result we can hope for.  If we were to choose a learning rate that was much too big, the performance would be very different.  Here we set `lr=500` which is much too big.

# In[96]:


wed = ThreeBlue()


# In[97]:


optimizer = torch.optim.SGD(wed.parameters(), lr=500)


# In[98]:


epochs = 10

for i in range(epochs):
    y_true = one_hot(training_data.targets, num_classes=10).to(torch.float)
    y_pred = wed(training_data.data)
    loss = loss_fn(y_true,y_pred)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(loss)


# Here it improves for one iteration of gradient descent, and then it seems to get stuck.
