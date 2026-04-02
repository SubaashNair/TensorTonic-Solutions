import numpy as np

def apply_homogeneous_transform(T, points):
    """
    Apply 4x4 homogeneous transform T to 3D point(s).
    """
    points = np.array(points)
    single = points.ndim == 1
    if single:
        points = points.reshape(1, 3)
    
    ones = np.ones((points.shape[0], 1))
    points_h = np.hstack([points, ones])
    
    transformed_h = (T @ points_h.T).T
    result = transformed_h[:, :3]
    
    if single:
        return result.reshape(3,)
    return result