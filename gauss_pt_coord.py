#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import displacement_solver
import mesh_gen
import quadrature
import shape_function
import constitutive
import stress_gauss


# In[2]:


def gauss_pts(ngp, vertex_coord, gp,el_type):
    gausspts=np.zeros((ngp*ngp,2))
    temp=0
    for i in range(ngp):
        for j in range(ngp):
            xi = gp[i]
            eta = gp[j]
            N_matrix=shape_function.ShapeFunction(xi, eta, el_type)
            N=N_matrix.get_N_matrix()
            ans=np.dot(N, vertex_coord)#.reshape(-1)
            gausspts[temp][:]=ans
            temp=temp+1
    
    return gausspts


# In[ ]:




