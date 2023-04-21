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


def patch_n_int_nodes(ms):
    patch=np.zeros((ms-1,ms-1,4),dtype=int)
    # print(patch)
    temp=1
    for j in range(ms-1):
        for i in range(ms-1):
            patch[j][i][0]=temp
            patch[j][i][1]=temp+1
            patch[j][i][2]=temp+ms
            patch[j][i][3]=temp+ms+1
            temp=temp+1
        temp=temp+1
    # print(patch)
    patch=patch.reshape(patch.shape[0]*patch.shape[1],-1)
    n_patches=patch.shape[0]

    #internal nodes
    temp=1
    int_nodes=[]
    for i in range(ms-1):
        itr=temp*(ms+1)+1
        for j in range(ms-1):
            int_nodes.append(itr)
            itr=itr+1

        temp=temp+1
    return patch, n_patches, int_nodes


def patch_n_int_nodes_plat_with_hole(ms):
    patch=np.zeros((2*ms-1,ms-1,4),dtype=int)
    # print(patch)
    # print(len(patch))
    temp=1
    for j in range(2*ms-1):
        for i in range(ms-1):
            patch[j][i][0]=temp
            patch[j][i][1]=temp+1
            patch[j][i][2]=temp+ms
            patch[j][i][3]=temp+ms+1
            temp=temp+1
        temp=temp+1
    # print(patch)
    patch=patch.reshape(patch.shape[0]*patch.shape[1],-1)
    n_patches=patch.shape[0]
    
    #internal nodes
    temp=1
    int_nodes=[]
    for i in range(2*ms-1):
        itr=temp*(ms+1)+1
        for j in range(ms-1):
            int_nodes.append(itr)
            itr=itr+1

        temp=temp+1

    return patch, n_patches, int_nodes
