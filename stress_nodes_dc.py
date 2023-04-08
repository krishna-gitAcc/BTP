#!/usr/bin/env python
# coding: utf-8

# In[5]:


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
import gauss_pt_coord
import patch_n_int_nodes


# In[9]:


def stress_dc(connect, coord, u, nel, el_type, C):
    para_nodes=np.array([[-1,-1],[1,-1],[1,1],[-1,1]])
    stress_nodes=np.zeros((np.amax(connect)+1,4))
    strain_nodes=np.zeros((np.amax(connect)+1,4))
    
    for i in range(nel):
        node = connect[i, :]
        vertex_coord = coord[node, :]
        nd=np.zeros((node.shape[0]*2,1), dtype=int)
        ind=np.zeros((node.shape[0]*2,1), dtype=int)
        for p in range(node.shape[0]):
            ind[2*p]=node[p]*2
            ind[(2*p)+1]=node[p]*2+1

        u_el=u[ind]

        for j in range(4):
            xi = para_nodes[j][0]
            eta = para_nodes[j][1]
            shape_func = shape_function.ShapeFunction(xi, eta, el_type)
            shape_func.compute_Jacobian(vertex_coord)
            J = shape_func.J
            B = shape_func.get_B_matrix()
            strain_i_n=np.dot(B, u_el)
            stress_i_n=np.dot(C, strain_i_n).reshape(3)

            nd=node[j]
            old_stress=stress_nodes[nd][0:3]
            old_strain=strain_nodes[nd][0:3]
            occurences=stress_nodes[nd][3]
            if(occurences!=0):
                stress_nodes[nd][0:3]=(old_stress*occurences+stress_i_n)/(1+occurences)
                strain_nodes[nd][0:3]=(old_strain*occurences+strain_i_n.reshape(3))/(1+occurences)
                stress_nodes[nd][3]=occurences+1
                strain_nodes[nd][3]=occurences+1
            else:
                stress_nodes[nd][0:3]=stress_i_n
                strain_nodes[nd][0:3]=strain_i_n.reshape(3)
                stress_nodes[nd][3]=1
                strain_nodes[nd][3]=1
                
    stress_nodes=np.delete(stress_nodes, 3,1)
    strain_nodes=np.delete(strain_nodes, 3,1)
    return stress_nodes, strain_nodes


# In[ ]:




