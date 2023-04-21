import numpy as np
import shape_function
import quadrature
from joblib import Parallel, delayed


def ele_body_force(ele, coord, connect, b, el_type, ngp):
    """Function to calculate the element body force vector
    Inputs:
    1) ele = element no.
    2) coord = coordinate matrix
    3) connect = connectivity matrix
    4) b = body force vector
    5) el_type = element type (0 ---> 4 noded quadrilateral, 1 ---> 8 noded quadrilateral)
    6) ngp = number of Gauss points"""

    # Nodes
    node = connect[ele, :]
    # Coordinates of the nodes
    vertex_coord = coord[node, :]
    # Element stiffness matrix
    fbody_ele = np.zeros((2 * vertex_coord.shape[0], 1))
    # Gauss points and weights
    gp, weights = quadrature.quadrature(ngp)

    for i in range(ngp):
        for j in range(ngp):
            xi = gp[i]
            eta = gp[j]
            shape_func = shape_function.ShapeFunction(xi, eta, el_type)
            N_matrix = shape_func.get_N_matrix()
            shape_func.compute_Jacobian(vertex_coord)
            J = shape_func.J
            fbody_ele += (N_matrix.T @ b) * weights[i] * weights[j] * np.linalg.det(J)

    return fbody_ele.reshape(2 * vertex_coord.shape[0])


def ele_f_nbc(ele, coord, connect, bc_type, T, el_type, ngp):
    """Function used for calculating the Natural Boundary term of each element
    Inputs:
    a) ele = element no.
    b) coord = coordinate matrix
    c) connect = connectivity matrix
    d) bc_type = boundary data
    e) T = matrix containing all the traction vectors along all the Neumann boundaries
    g) ngp = no. of Gauss points used for line integration"""

    nodes_perele = connect.shape[1]
    ele_edges_nbc = list(bc_type[ele, :][bc_type[ele, :] > 0])
    nodes = connect[ele, :]
    vertex_coord = coord[nodes, :]
    corner_nodes = np.hstack((nodes[0:4], nodes[0]))
    f_nbc_ele = np.zeros((2 * vertex_coord.shape[0], 1))
    gp, weights = quadrature.quadrature(ngp)

    if len(ele_edges_nbc) != 0:
        for i in range(len(ele_edges_nbc)):
            side_no = ele_edges_nbc[i]
            t = T[side_no-1, :].reshape(2, 1)
            node1 = corner_nodes[side_no - 1]
            node2 = corner_nodes[side_no]
            p1 = coord[node1, :]
            p2 = coord[node2, :]
            detJ = np.linalg.norm(p2 - p1) / 2

            if side_no == 1:
                eta = -1
                for j in range(ngp):
                    xi = gp[j]
                    shape_func = shape_function.ShapeFunction(xi, eta, el_type)
                    N = shape_func.get_N_matrix()
                    f_nbc_ele += (N.T @ t) * weights[j] * detJ
            elif side_no == 2:
                xi = 1
                for j in range(ngp):
                    eta = gp[j]
                    shape_func = shape_function.ShapeFunction(xi, eta, el_type)
                    N = shape_func.get_N_matrix()
                    f_nbc_ele += (N.T @ t) * weights[j] * detJ
            elif side_no == 3:
                eta = 1
                for j in range(ngp):
                    xi = gp[j]
                    shape_func = shape_function.ShapeFunction(xi, eta, el_type)
                    N = shape_func.get_N_matrix()
                    f_nbc_ele += (N.T @ t) * weights[j] * detJ

    return f_nbc_ele.reshape(2 * vertex_coord.shape[0])


def f_global(coord, connect, b, T, bc_type, el_type, ngp2d, ngp1d):
    """Function to calculate the global load vector"""

    n_ele = connect.shape[0]  # No. of elements
    n_node = coord.shape[0]  # No. of nodes
    n_dof = 2 * n_node  # No. of DOFs

    f_body_par = Parallel(n_jobs=-1)(delayed(ele_body_force)(ele, coord, connect, b, el_type, ngp2d)
                                     for ele in range(n_ele))

    f_nbc_par = Parallel(n_jobs=-1)(delayed(ele_f_nbc)(ele, coord, connect, bc_type, T, el_type, ngp1d)
                                    for ele in range(n_ele))

    f_array = np.array(f_body_par) + np.array(f_nbc_par)
    print(f_nbc_par)
    print(len(f_nbc_par))
    f_global = np.zeros((n_dof, 1))
    for ele in range(n_ele):
        dof_ele = np.zeros((2*connect.shape[1],), dtype="int64")
        for j in range(connect.shape[1]):
            dof_ele[2*j] = 2*connect[ele, j]
            dof_ele[2*j + 1] = 2*connect[ele, j] + 1
        f_global[dof_ele, :] += f_array[ele, :].reshape(f_array.shape[1], 1)

    return f_global

def f_global_plat_with_hole(nx,coord, connectivity, T, ngp2d, ngp1d):
    n_ele = connectivity.shape[0];
    n_node = coord.shape[0];
    n_dof = 2*n_node;
    surf_ele = np.arange(nx-1, nx*nx, nx)
    f_global = np.zeros([n_dof,1]);

    gp, weights = quadrature.quadrature(ngp1d)

    for i in surf_ele:
        node_1 = connectivity[i, 1];
        node_2 = connectivity[i, 2];
        global_dof_number = np.array([2*node_1, 2*node_1+1, 2*node_2, 2*node_2+1]);
        fe = np.zeros((4, 1));
        node_1_coord = coord[node_1,:];
        node_2_coord = coord[node_2,:];
        le = np.linalg.norm(node_1_coord-node_2_coord)
        J = le/2;
        f_vector = np.zeros([4,1])
        for j in range(ngp1d):
            xi = gp[j]
            weight = weights[j];
            N1 = (1-xi)/2;
            N2 = (1+xi)/2;

            N = np.array([[N1, 0], [0, N1], [N2, 0], [0, N2]]);
            # print(T.shape)
            f_vector = np.dot(N, T.T) * j * weight + f_vector
            # print(f_vector.shape)
        # print(f_vector)
        f_global[global_dof_number, :] += f_vector

    return f_global


def f_global_plat_with_hole_test(nx,coord, connectivity, T, ngp2d, ngp1d):
    n_ele = connectivity.shape[0];
    n_node = coord.shape[0];
    n_dof = 2*n_node;
    surf_ele = np.arange(nx-1, nx*nx, nx)
    f_global = np.zeros([n_dof,1]);
    t = T.reshape(2, 1)
    gp, weights = quadrature.quadrature(ngp1d)

    for i in surf_ele:
        node_1 = connectivity[i, 1];
        node_2 = connectivity[i, 2];
        global_dof_number = np.array([2*node_1, 2*node_1+1, 2*node_2, 2*node_2+1]);
        fe = np.zeros((4, 1));
        node_1_coord = coord[node_1,:];
        node_2_coord = coord[node_2,:];
        le = np.linalg.norm(node_1_coord-node_2_coord)
        J = le/2;
        f_vector = np.zeros([8,1])
        xi = 1;
        for j in range(ngp1d):
            eta = gp[j]
            shape_func = shape_function.ShapeFunction(xi, eta, 0)
            N = shape_func.get_N_matrix()
            f_vector+=(N.T@t)*weights[j]*J
            dof_ele = np.zeros((2*connectivity.shape[1],), dtype = "int64")
            for k in range(connectivity.shape[1]):
                dof_ele[2*k] = 2*connectivity[i, k]
                dof_ele[2*k+1] = 2*connectivity[i, k]+1
            f_global[dof_ele, :] += f_vector
    return f_global
