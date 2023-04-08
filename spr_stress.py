
import numpy as np
import getA
import getB


def spr(gauss_coords, coord, connect, stress, int_nodes, n_patches, patch, ms):
    nodal_stress = np.zeros((n_patches, 3))
    a = np.zeros((n_patches, 3, 3))
    for i in range(n_patches):
        elems = patch[i][:]
        stress_p = stress[elems-1][:][:]
        gauss_p = gauss_coords[elems-1][:][:]
        int_node_p = int_nodes[i]
        int_node_coord = coord[int_node_p]
        int_node_coord = np.insert(int_node_coord, 0, 1)

        # pk=gauss_p-coord[int_node_p]
        pk = gauss_p
        pk = np.insert(pk, 0, np.ones(pk.shape[0]), axis=1)
        A = getA.get_A(pk)
        B1, B2, B3 = getB.get_B(pk, stress_p)

        a1 = np.linalg.solve(A, B1)
        a2 = np.linalg.solve(A, B2)
        a3 = np.linalg.solve(A, B3)
        a[i][0][:] = a1
        a[i][1][:] = a2
        a[i][2][:] = a3

        nodal_stress[i][0] = np.dot(int_node_coord.T, a1)
        nodal_stress[i][1] = np.dot(int_node_coord.T, a2)
        nodal_stress[i][2] = np.dot(int_node_coord.T, a3)

    # side pts stresses
    sp = []
    temp = ms+1
    for i in range(ms-1):
        sp.append(temp)
        sp.append(temp+ms)
        temp = temp+ms+1
    for i in range(ms+1):
        sp.append(temp)
        temp = temp+1

    for i in range(ms+1):
        sp.append(i)

    sp = sorted(sp)

    comp = [0, 1, 2]
    nod_stress = np.zeros((len(sp), 1))
    spr_stress = np.zeros(((ms+1)*(ms+1), 3))

    for j in range(len(comp)):
        for i in range(len(sp)):
            dd = np.where(connect[patch-1] == sp[i])
            lst = dd[0]
        #     for j in range(len(dd)):
        #         lst.append(dd[j][0])
            # find patches common to that node
            pat = list(set(lst))
            strs = 0
            for k in range(len(pat)):
                a_temp = a[pat[k]][comp[j]][:]
                coo_node = coord[sp[i]]
                coo_node = np.insert(coo_node, 0, 1)
                strs = np.dot(coo_node, a_temp)
            nod_stress[i] = strs/len(pat)

        for l in range((ms+1)*(ms+1)):
            if l in int_nodes:
                index = int_nodes.index(l)
                spr_stress[l][comp[j]] = nodal_stress[index][comp[j]]
            if l in sp:
                index = sp.index(l)
                spr_stress[l][comp[j]] = nod_stress[index]

    return spr_stress
