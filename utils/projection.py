import numpy as np

def project_depth_to_3d_points(
    z_map: np.ndarray,
    fx: float = 500.0,
    fy: float = 500.0,
    cx: float = None,
    cy: float = None,
    subsample_stride: int = 2
) -> dict:
    """
    Project 2D depth matrix Z(u,v) into 3D world coordinates (X_w, Y_w, Z_w).
    
    Formula:
      X_w = ((u - cx) / fx) * Z
      Y_w = ((v - cy) / fy) * Z
      Z_w = Z
    """
    h, w = z_map.shape
    if cx is None:
        cx = w / 2.0
    if cy is None:
        cy = h / 2.0

    # Subsample grid for efficiency
    u_coords = np.arange(0, w, subsample_stride)
    v_coords = np.arange(0, h, subsample_stride)
    u_grid, v_grid = np.meshgrid(u_coords, v_coords)

    z_subsampled = z_map[v_coords[:, None], u_coords]

    x_w = ((u_grid - cx) / fx) * z_subsampled
    y_w = ((v_grid - cy) / fy) * z_subsampled
    z_w = z_subsampled

    return {
        "x": x_w.astype(np.float32),
        "y": y_w.astype(np.float32),
        "z": z_w.astype(np.float32),
        "stride": subsample_stride,
        "shape": x_w.shape
    }
