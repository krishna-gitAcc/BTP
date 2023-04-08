#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


def get_A(pk):
    A=np.zeros((pk.shape[1],pk.shape[1]))
    for i in range(pk.shape[0]):
        A=A+np.outer(pk[i],pk[i])
    
    return A


# In[ ]:




