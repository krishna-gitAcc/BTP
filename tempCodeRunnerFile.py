domain_coord = np.array([[5, 0], [100, 0], [100, 50], [0, 50], [0, 5]]);
b = np.array([[0], [0]]);

#Traction components
q = 1/16

T = np.array([[1, 0], [0, 0], [0, 0]]);

E = 1.0;
nu = 1/3;

problem_type = 0;

el_type = 0;

ngp2d = 1;
ngp1d = 2;

N = 4;

u_list_q4 = displacement_solver.solve_fem_plat_with_hole(N, E, nu, ngp2d, ngp1d, el_type, problem_type, domain_coord, b, T)
print(u_list_q4)