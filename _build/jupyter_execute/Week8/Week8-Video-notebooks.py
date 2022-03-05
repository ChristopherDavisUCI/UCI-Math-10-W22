#!/usr/bin/env python
# coding: utf-8

# # Week 8 Video notebooks

# In[ ]:


import numpy as np
import torch
from torch import nn
from torchvision.transforms import ToTensor


# ## A little neural network
# 
# ![Screenshare1](../images/Week8-ipad1.jpg)

# ## Neural network for logical or

# In[ ]:


X = torch.tensor([[0,0],[0,1],[1,0],[1,1]]).to(torch.float)
y = torch.tensor([0,1,1,1]).to(torch.float).reshape(-1,1)


# In[ ]:


class LogicOr(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2,1),
            nn.Sigmoid()
        )

    def forward(self,x):
        return self.layers(x)
    


# In[ ]:


model = LogicOr()


# In[ ]:


model(X)


# In[ ]:


for p in model.parameters():
    print(p)


# ## Evaluating our neural network

# In[ ]:


loss_fn = nn.BCELoss()


# In[ ]:


y.shape


# In[ ]:


loss_fn(model(X),y)


# ## Optimizer

# In[ ]:


optimizer = torch.optim.SGD(model.parameters(), lr=0.1)


# In[ ]:


for p in model.parameters():
    print(p)
    print(p.grad)
    print("")


# In[ ]:


loss = loss_fn(model(X),y)


# In[ ]:


optimizer.zero_grad()
loss.backward()


# In[ ]:


for p in model.parameters():
    print(p)
    print(p.grad)
    print("")


# In[ ]:


optimizer.step()


# In[ ]:


for p in model.parameters():
    print(p)
    print(p.grad)
    print("")


# ## Training the model

# In[ ]:


100%50


# In[ ]:


epochs = 1000

for i in range(epochs):
    loss = loss_fn(model(X),y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i%50 == 0:
        print(loss)


# In[ ]:


epochs = 1000

for i in range(epochs):
    loss = loss_fn(model(X),y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i%100 == 0:
        print(loss)


# In[ ]:


epochs = 1000

for i in range(epochs):
    loss = loss_fn(model(X),y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i%100 == 0:
        print(loss)


# In[ ]:


for p in model.parameters():
    print(p)
    print(p.grad)
    print("")


# In[ ]:


model(X)


# ## A little neural network: results
# 
# ![screenshare2](../images/Week8-ipad2.jpg)
