import os
import copy
import itertools

import numpy as np
from coxeter.shapes import ConvexPolyhedron


class SlopeDiagramRepresentation:
    """ A class to represent the Slope Diagram Representation (SDR) of a Convex Polyhedron
    """

    def __init__(self,verts: np.ndarray):
        """contructor method

        Args:
            convex_polyhedron (ConvexPolyhedron): a convex polyhedron to SDR.
        """
        
        try:
            polyhedron=ConvexPolyhedron(verts)
            # polyhedraon.merge_faces()
        except:
            raise "Initialization of convex polyhedron failt"
        

        self.center=polyhedron.vertices.mean(axis=0)
        self.initial_verts=verts
        self.verts=polyhedron.vertices - self.center
        self.faces=polyhedron.faces
        self.volume=polyhedron.volume
        self.normals=polyhedron.normals
        self.areas=polyhedron.get_face_area()
        self.face_neighbors=polyhedron.neighbors
        self.n_verts=polyhedron.num_vertices
        self.n_faces=polyhedron.num_faces

        self.pyvista_faces=self._get_pyvista_faces()
        self.face_centers=self._get_face_centers()
        self.face_edges=self._get_face_edges()
        self.face_edges_weights=self._get_face_edges_weights()

        self.spherical_points=copy.copy(self.normals)
        self.spherical_point_indices = np.arange(len(self.spherical_points))
        self.spherical_point_weights=copy.copy(self.areas)

        self.spherical_arcs=copy.copy(self.face_edges)
        self.spherical_arc_indices=np.arange(len(self.spherical_arcs))
        self.spherical_arc_weights=copy.copy(self.face_edges_weights)

        self._compute_distances()

    def h_support(self, u):
        """Calculate the support function h(A; u) of the polyhedron in direction u

        Args:
            u (np.ndarray): the direction vector

        Returns:
            float: the support function h(A; u) of the polyhedron in direction u
        """
        return np.max(np.dot(self.verts, u))
    
    def calculate_volume(self):
        """Calculate the volume of the polyhedron using the support function and the areas of the facets.

        Returns:
            float: the volume of the polyhedron.
        """
        volume = 0
        for i in range(self.n_faces):
            u = self.normals[i]  # outward unit normal at facet i
            S = self.areas[i]  # area of the facet i
            volume += self.h_support(u) * S

        volume /= 3

        return volume
    
    def mixed_volume(self, other):
        """Calculate the mixed volume V(self, other, other)

        Args:
            other (SlopeDiagramRepresentation): another SlopeDiagramRepresentation instance

        Returns:
            float: the mixed volume V(self, other, other)
        """
        # compute the mixed volume
        volume = 0
        for i in range(self.n_faces):
            u = self.normals[i]  # outward unit normal at facet i
            S = self.areas[i]  # area of the facet i
            volume += other.h_support(u) * S

        volume /= 3

        return volume
    
    def vol_minkowski_sum(self, sdr_b):
        v = self.volume + 3 * self.mixed_volume(sdr_b) + 3 * sdr_b.mixed_volume(self) + sdr_b.volume
        return v

    def sigma_1(self, sdr_b):
        v_mink_sum = self.vol_minkowski_sum(sdr_b)
        similarity = (8 * self.volume**0.5 * sdr_b.volume**0.5 ) / v_mink_sum
        return similarity

    def sigma_2(self, sdr_b):
        similarity = (self.volume**(2/3) * sdr_b.volume**(1/3)) / self.mixed_volume(sdr_b)
        return similarity

    def sigma_3(self, sdr_b):
        similarity = 0.5 * (self.sigma_2(sdr_b) + sdr_b.sigma_2(self))
        return similarity

    def rotate(self, rotation_matrix):
        """Rotates the Slope Diagram Representation (SDR) by a given rotation matrix.

        Args:
            rotation_matrix (np.ndarray): a 3x3 rotation matrix.
        """
        # Check that the rotation matrix is of correct size
        assert rotation_matrix.shape == (3, 3), "Rotation matrix must be 3x3"
        
        # Apply the rotation matrix to each point and recalculate the SDR
        self.verts = np.dot(self.verts, rotation_matrix.T)
        
        # Recalculate the spherical points, normals and edges since the positions of vertices have changed
        self.spherical_points = np.dot(self.spherical_points, rotation_matrix.T)
        self.normals = np.dot(self.normals, rotation_matrix.T)
        self.face_edges = self._get_face_edges()  # assuming this method uses self.verts to calculate the edges
        self.face_edges_weights = self._get_face_edges_weights()  # assuming this method uses self.verts to calculate the edges weights

    def _get_face_centers(self):
        face_centers=np.zeros(shape=(self.n_faces,3))
        for i_face,i_verts in enumerate(self.faces):
            face_verts=self.verts[i_verts]
            face_centers[i_face]=face_verts.mean(axis=0)
        return face_centers
    
    def _get_pyvista_faces(self):
        pyvista_faces=[]
        for i_face,i_verts in enumerate(self.faces):
            n_verts=len(i_verts)
            pyvista_face=np.append(np.array([n_verts]),i_verts)
            pyvista_faces.append(pyvista_face)
        pyvista_faces = np.hstack(pyvista_faces)
        return pyvista_faces
    
    def _get_face_edges(self):
        face_edges = []
        for i, neighbors in enumerate(self.face_neighbors):
            for neighbor in neighbors:
                # Avoid duplicate edges by only considering pairs where neighbor > i
                if neighbor > i:
                    face_edges.append([i, neighbor])
        return np.array(face_edges)
    
    def _get_face_edges_weights(self):
        face_edges_weights = []
        for edge in self.face_edges:
            normal1 = self.normals[edge[0]]
            normal2 = self.normals[edge[1]]
            # calculate the spherical distance between the two normals
            spherical_distance = np.arccos(np.dot(normal1, normal2))
            face_edges_weights.append(spherical_distance)
        return np.array(face_edges_weights)
    
    def _compute_distances(self):
        """Compute distances between points and arcs."""

        # Calculate distances between all pairs of spherical points
        self.d_pts = np.sqrt(np.sum((self.spherical_points[:, np.newaxis] - self.spherical_points)**2, axis=2))

        # Calculate distances between all pairs of spherical arcs
        arc_distances = np.sqrt(np.sum((self.spherical_arcs[:, np.newaxis] - self.spherical_arcs)**2, axis=2))

        # Find the indices of the minimum and maximum distances for each arc
        self.mind_arcs = np.argmin(arc_distances, axis=1)
        self.maxd_arcs = np.argmax(arc_distances, axis=1)
    