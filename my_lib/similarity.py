import itertools
import copy

import numpy as np

from my_lib.sdr import SlopeDiagramRepresentation

class SimilarityOpt:
    def __init__(self,verts_a,verts_b):
        self.sdr_a=SlopeDiagramRepresentation(verts_a)
        self.sdr_b=SlopeDiagramRepresentation(verts_b)

        self.point_arc_combinations = list(itertools.product(self.sdr_a.spherical_point_indices, self.sdr_b.spherical_arc_indices))
        self.n = len(self.point_arc_combinations)

    def _similarity_main_loop(self):
        self.s=0
        for i in range(0, self.n - 2):
            for j in range(i+1 ,self.n - 1):
                for k in range(j+1 ,self.n):
                    if self._feasible_distances(i,j,k):
                        if self._well_posed(i,j,k):
                            crit_rotations = self._calculaute_critcal_orientations(i,j,k)
                            for rotation in crit_rotations:
                                if self._points_on_arcs(i,j,k,r=rotation):
                                    rotated_sdr=copy.copy(self.sdr_b)
                                    rotated_sdr.rotate(rotation)
                                    new_similarity = self._calculate_similarity(self.sdr_a,rotated_sdr)
                                    self.s = max(self.s, new_similarity)

    def _calculate_similarity(self,sdr,sdr_other,sigma=2):
        if sigma==0:
            similarity = sdr.sigma_1(sdr_other)
        elif sigma==1:
            similarity = sdr.sigma_2(sdr_other)
        else:
            similarity = sdr.sigma_3(sdr_other)
        return similarity
    
    def _feasible_distances(self, i, j, k):
        # Extract the point and arc indices for each pair
        point_a, arc_k = self.point_arc_combinations[i]
        point_b, arc_l = self.point_arc_combinations[j]
        point_c, arc_m = self.point_arc_combinations[k]

        # Calculate point distances
        d_ab = self.sdr_a.d_pts[point_a, point_b]
        d_bc = self.sdr_a.d_pts[point_b, point_c]
        d_ca = self.sdr_a.d_pts[point_c, point_a]

        # Check each condition. If any of the conditions is not satisfied, return False
        if not (self.sdr_b.mind_arcs[arc_k, arc_l] <= d_ab <= self.sdr_b.maxd_arcs[arc_k, arc_l]):
            return False
        if not (self.sdr_b.mind_arcs[arc_l, arc_m] <= d_bc <= self.sdr_b.maxd_arcs[arc_l, arc_m]):
            return False
        if not (self.sdr_b.mind_arcs[arc_m, arc_k] <= d_ca <= self.sdr_b.maxd_arcs[arc_m, arc_k]):
            return False

        # If all conditions are satisfied, return True
        return True

    def _well_posed(self, i, j, k):
        # Extract the point and arc indices for each pair
        point_a, arc_k = self.point_arc_combinations[i]
        point_b, arc_l = self.point_arc_combinations[j]
        point_c, arc_m = self.point_arc_combinations[k]

        # Get points and normal vectors
        a = self.sdr_a.points[point_a]
        b = self.sdr_a.points[point_b]
        c = self.sdr_a.points[point_c]

        k = self.sdr_b.arcs[arc_k].normal
        l = self.sdr_b.arcs[arc_l].normal
        m = self.sdr_b.arcs[arc_m].normal

        # Check conditions. If any of them is satisfied, return True.
        if self._is_parallel(a, b) and self._is_parallel(b, c) and self._is_coplanar(k, l, m):
            return True
        elif self._is_parallel(a, b) and self._is_perpendicular(c, b) and self._is_perpendicular(k, m) and self._is_parallel(l, m):
            return True
        elif self._is_parallel(a, c) and self._is_perpendicular(b, c) and self._is_perpendicular(k, l) and self._is_parallel(m, l):
            return True
        elif self._is_parallel(b, c) and self._is_perpendicular(a, c) and self._is_perpendicular(l, k) and self._is_parallel(m, k):
            return True
        elif self._is_perpendicular(b, a) and self._is_perpendicular(c, a) and self._is_parallel(l, m) and self._is_perpendicular(k, m):
            return True
        elif self._is_perpendicular(a, b) and self._is_perpendicular(c, b) and self._is_parallel(k, m) and self._is_perpendicular(l, m):
            return True
        elif self._is_perpendicular(a, c) and self._is_perpendicular(b, c) and self._is_parallel(k, l) and self._is_perpendicular(m, l):
            return True
        elif self._is_coplanar(a, b, c) and self._is_parallel(k, l) and self._is_parallel(l, m):
            return True

        # If none of the conditions is satisfied, return False
        return False
    
    def _is_coplanar(self, v1, v2, v3):
        matrix = np.stack([v1, v2, v3])
        return np.linalg.det(matrix) == 0
    
    def _is_parallel(self, v1, v2):
        return np.allclose(np.cross(v1, v2), np.zeros_like(v1))

    def _is_perpendicular(self, v1, v2):
        return np.allclose(np.dot(v1, v2), 0)

    def _points_on_arcs(self,i,j,k,r):
        # Extract the point and arc indices for each pair
        point_a, arc_k = self.point_arc_combinations[i]
        point_b, arc_l = self.point_arc_combinations[j]
        point_c, arc_m = self.point_arc_combinations[k]

        # Get points
        a = self.sdr_a.points[point_a]
        b = self.sdr_a.points[point_b]
        c = self.sdr_a.points[point_c]

        # Rotate points
        a_rot = np.dot(r, a)
        b_rot = np.dot(r, b)
        c_rot = np.dot(r, c)

        # Check if the rotated points lie on the corresponding arcs
        if not self.sdr_b.arcs[arc_k].point_on_arc(a_rot):
            return False
        if not self.sdr_b.arcs[arc_l].point_on_arc(b_rot):
            return False
        if not self.sdr_b.arcs[arc_m].point_on_arc(c_rot):
            return False

        # If all points lie on the arcs, return True
        return True

    # def _point_on_arc(self, point, arc):
    #     # Calculate the vectors between the points and the ends of the arc
    #     v1 = np.cross(self.p1.vector, point.vector)
    #     v2 = np.cross(point.vector, self.p2.vector)

    #     # If the cross product of these vectors is not zero, the point does not lie on the great circle of the arc
    #     if not np.allclose(v1, 0) or not np.allclose(v2, 0):
    #         return False

    #     # Otherwise, calculate the angles between the point and the arc ends
    #     angle1 = self.p1.angle(point)
    #     angle2 = self.p2.angle(point)

    #     # If the sum of these angles is equal to the angle of the arc, the point lies on the arc
    #     return np.allclose(angle1 + angle2, self.angle())

    def _calculaute_critcal_orientations(self,i,j,k):
        pass
