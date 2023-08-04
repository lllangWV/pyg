import my_lib
from my_lib.utils import TEST_DATA
import pyvista as pv
import numpy as np
# pv.set_jupyter_backend('trame')  

sdr_a=my_lib.sdr.SlopeDiagramRepresentation(verts=TEST_DATA['cube']['vertices'])
# sdr_b=my_lib.sdr.SlopeDiagramRepresentation(verts=my_lib.math.rot_z(TEST_DATA['cube']['vertices'],45))
sdr_b=my_lib.sdr.SlopeDiagramRepresentation(verts=TEST_DATA['cube']['vertices'])

print(sdr_a.center)
print(sdr_a.verts)
print(sdr_a.volume)
print(sdr_a.normals)
print(sdr_a.areas)
print(sdr_a.faces)
print(sdr_a.face_neighbors)
print(sdr_a.face_centers)
print(sdr_a.face_edges)
print(sdr_a.face_edges.shape)
print(sdr_a.face_edges_weights)
print(sdr_a.calculate_volume())


print(sdr_a.mind_arcs)
print(sdr_a.maxd_arcs)
print(sdr_a.d_pts)

# print(sdr_a.mixed_volume(sdr_b))



# print(sdr_a.vol_minkowski_sum(sdr_b))
# print(sdr_a.sigma_1(sdr_b))
# print(sdr_a.sigma_2(sdr_b))
# print(sdr_a.sigma_3(sdr_b))
# poly = pv.PolyData(cube.verts,faces=cube.pyvista_faces)
# p = pv.Plotter(notebook=False)
# p.add_arrows(cube.face_centers[:1], cube.normals[:1], 0.1)
# p.add_mesh(poly)
# p.show_grid()
# p.show()


