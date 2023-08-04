import numpy as np

def rot_z(points, angle_deg):
    # Convert the angle to radians
    angle_rad = np.radians(angle_deg)
    
    # Create the rotation matrix
    rotation_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad), 0],
        [np.sin(angle_rad), np.cos(angle_rad), 0],
        [0, 0, 1]
    ])
    
    # Transform the points
    transformed_points = np.dot(points, rotation_matrix)
    
    return transformed_points