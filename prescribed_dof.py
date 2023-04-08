import numpy as np


def find_prescribed_dof(bc_type, coord, connect):
    """Function to find the prescribed DOFs"""

    n_ele = connect.shape[0]
    n_nodes = coord.shape[0]
    nodes_perele = connect.shape[1]
    dbc_nodes = []

    for ele in range(n_ele):
        sides_dbc = list(np.argwhere(bc_type[ele, :] == -1).reshape(-1))
        if nodes_perele == 4:
            nodes = np.hstack((connect[ele, :], connect[ele, 0]))
            if len(sides_dbc) != 0:
                for i in range(len(sides_dbc)):
                    node1 = nodes[sides_dbc[i]]
                    node2 = nodes[sides_dbc[i] + 1]
                    dbc_nodes.append(node1)
                    dbc_nodes.append(node2)
        if nodes_perele == 8 or nodes_perele == 9:
            nodes = np.hstack((connect[ele, :4], connect[ele, 0]))
            if len(sides_dbc) != 0:
                for i in range(len(sides_dbc)):
                    node1 = nodes[sides_dbc[i]]
                    node2 = nodes[sides_dbc[i] + 1]
                    node3 = connect[ele, 2*sides_dbc[i] + 1]
                    dbc_nodes.append(node1)
                    dbc_nodes.append(node2)
                    dbc_nodes.append(node3)

    dbc_nodes = list(set(dbc_nodes))
    # print(len(dbc_nodes))
    # nodes_left = list(set(np.arange(n_nodes)) - set(dbc_nodes))
    dbc_dof = list(np.zeros(2*len(dbc_nodes)))
    for i in range(len(dbc_nodes)):
        dbc_dof[2*i] = 2*dbc_nodes[i]
        dbc_dof[2*i + 1] = 2*dbc_nodes[i] + 1

    dof_left = list(set(np.arange(2*n_nodes)) - set(dbc_dof))
    return dbc_dof, dof_left
    # return dbc_nodes, nodes_left
