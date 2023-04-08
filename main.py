import mesh_gen
import connectivity_plat_hole

mesh_object = mesh_gen.platWithHole(4, 4, [], 0)
connect = mesh_object.connectivity()
print(connect)