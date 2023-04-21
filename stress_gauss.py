import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import displacement_solver
import mesh_gen
import quadrature
import shape_function
import constitutive

#element stress calculation
def get_element_stress(ele, ngp, el_type, connect,coord, u,C):
#     stress_i_gp=np.zeros((3,1))
#     stress_i_n=np.zeros((3,1))
    stress_i=np.zeros(((ngp*ngp),3))
#     stress_i_nodes=np.zeros((4,3))
    node = connect[ele, :]
    vertex_coord = coord[node, :]

    ind=np.zeros((node.shape[0]*2,1), dtype=int)
    for i in range(node.shape[0]):
        ind[2*i]=node[i]*2
        ind[(2*i)+1]=node[i]*2+1

    u_el=u[ind]
    u_el = u_el.reshape((8, 1))

    gp, weights = quadrature.quadrature(ngp)

    strain_g=0
    temp=0
    for i in range(ngp):
        for j in range(ngp):
            xi = gp[i]
            eta = gp[j]
            shape_func = shape_function.ShapeFunction(xi, eta, el_type)
            shape_func.compute_Jacobian(vertex_coord)
            J = shape_func.J
            B = shape_func.get_B_matrix()
            strain_i_gp=np.dot(B, u_el)
            strain_g=strain_i_gp
            stress_i_gp=np.dot(C, strain_i_gp)
            stress_i[temp][:]=stress_i_gp.reshape(-1)
            temp=temp+1


    #FOR Nodes
#     for i in range(len(para_nodes)):
#         xi = para_nodes[i][0]
#         eta = para_nodes[i][1]
#         shape_func = shape_function.ShapeFunction(xi, eta, el_type)
#         shape_func.compute_Jacobian(vertex_coord)
#         J = shape_func.J
#         B = shape_func.get_B_matrix()
#         strain_i_n=np.dot(B, u_el)
#         stress_i_n=np.dot(C, strain_i_n)
#         stress_i_nodes[i][:]=stress_i_n.reshape(-1)


    return stress_i, strain_g


