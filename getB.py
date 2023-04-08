#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


def get_B(pk, stress_p):
    B1=np.zeros(pk.shape[1])
    for i in range(pk.shape[0]):
        B1=B1+np.dot(pk[i].T, stress_p[i][0][0])
    
    B2=np.zeros(pk.shape[1])
    for i in range(pk.shape[0]):
        B2=B2+np.dot(pk[i].T, stress_p[i][0][1])
    
    B3=np.zeros(pk.shape[1])
    for i in range(pk.shape[0]):
        B3=B3+np.dot(pk[i].T, stress_p[i][0][2])
        
    return B1, B2,  B3


# In[ ]:




