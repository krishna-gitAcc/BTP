import numpy as np
from scipy.sparse.linalg import spsolve
import mesh_gen
import stiffness_matrix
import force_vector
import boundary_data_generation
import prescribed_dof


def solve_fem(N, E, nu, ngp2d, ngp1d, el_type, problem_type, domain_coord, b, T):
    """Function to calculate the nodal displacements"""
    # Mesh parameters
    nx = N  #number of mesh in x-direction.
    ny = N  #number of mesh in y-direction.
    n_nodes = 0     #Total Number of Node for entire domain.
    if el_type == 0:
        n_nodes = int((nx + 1) * (ny + 1))
    elif el_type == 1:
        n_nodes = int((nx + 1)*(ny + 1) + nx*(ny + 1) + ny*(nx + 1))
    elif el_type == 2:
        n_nodes = int((nx + 1)*(ny + 1) + nx*(ny + 1) + ny*(nx + 1) + nx*ny)

    n_dof = 2*n_nodes

    # 2) Generate mesh data
    mesh_obj = mesh_gen.MeshGenerator(nx, ny, domain_coord.reshape(16, 1), el_type)
    # Coordinate matrix
    coord = mesh_obj.coord_array()
    # Connectivity matrix
    connect = mesh_obj.connectivity()
    # Boundary data
    bc_type = boundary_data_generation.boundary_data_generator(nx, ny, 4)

    # 3) Generate global stiffness matrix
    K_global = stiffness_matrix.global_stiffness(coord, connect, E, nu, el_type, problem_type, ngp2d)
    # plt.spy(K_global)
    # plt.show()

    # 4) Generate global force vector
    f_global = force_vector.f_global(coord, connect, b, T, bc_type, el_type, ngp2d, ngp1d)
    # print(f_global)

    # 5) Application of Dirichlet BC
    dbc_dof, dof_left = prescribed_dof.find_prescribed_dof(bc_type, coord, connect)
    u_prescribed = np.zeros((len(dbc_dof), 1))
    K_reduced = ((K_global[dof_left]).tocsc()[:, dof_left]).tocsr()
    f_reduced = f_global[dof_left, :] - ((K_global[dof_left]).tocsc()[:, dbc_dof]).tocsr().dot(u_prescribed)
    # f_reduced = f_global[dof_left]

    # 6) Solving the system of equations
    u_reduced = spsolve(K_reduced, f_reduced).reshape(-1, 1)

    # 7) Nodal displacement vector
    u_node = np.zeros((n_dof, 1))
    u_node[dbc_dof, :] = u_prescribed
    u_node[dof_left, :] = u_reduced
    u_node = u_node.reshape(-1)
    return u_node

def solve_fem_plat_with_hole(N, E, nu, ngp2d, ngp1d, el_type, problem_type, domain_coord, b, T):
    """Function to calculate the nodal displacements"""
    # Mesh parameters
    nx = N  #number of mesh in x-direction.
    ny = N  #number of mesh in y-direction.
    n_nodes = 0     #Total Number of Node for entire domain.
    if el_type == 0:
        n_nodes = int((nx + 1) * (ny + 1))
    elif el_type == 1:
        n_nodes = int((nx + 1)*(ny + 1) + nx*(ny + 1) + ny*(nx + 1))
    elif el_type == 2:
        n_nodes = int((nx + 1)*(ny + 1) + nx*(ny + 1) + ny*(nx + 1) + nx*ny)

    n_dof = 2*n_nodes

    # 2) Generate mesh data
    mesh_obj = mesh_gen.platWithHole(nx, ny, domain_coord.reshape(10, 1), el_type)
    # Coordinate matrix
    coord = mesh_obj.coord_array()
    # Connectivity matrix
    connect = mesh_obj.connectivity()
    # Boundary data
    bc_type = boundary_data_generation.boundary_data_generator_plat_with_hole(nx, ny, 4)

    # 3) Generate global stiffness matrix
    K_global = stiffness_matrix.global_stiffness(coord, connect, E, nu, el_type, problem_type, ngp2d)
    # plt.spy(K_global)
    # plt.show()

    # 4) Generate global force vector
    f_global = force_vector.f_global(coord, connect, b, T, bc_type, el_type, ngp2d, ngp1d)
    # print(f_global)

    # 5) Application of Dirichlet BC
    dbc_dof, dof_left = prescribed_dof.find_prescribed_dof(bc_type, coord, connect)
    u_prescribed = np.zeros((len(dbc_dof), 1))
    K_reduced = ((K_global[dof_left]).tocsc()[:, dof_left]).tocsr()
    f_reduced = f_global[dof_left, :] - ((K_global[dof_left]).tocsc()[:, dbc_dof]).tocsr().dot(u_prescribed)
    # f_reduced = f_global[dof_left]

    # 6) Solving the system of equations
    u_reduced = spsolve(K_reduced, f_reduced).reshape(-1, 1)

    # 7) Nodal displacement vector
    u_node = np.zeros((n_dof, 1))
    u_node[dbc_dof, :] = u_prescribed
    u_node[dof_left, :] = u_reduced
    u_node = u_node.reshape(-1)
    return u_node
