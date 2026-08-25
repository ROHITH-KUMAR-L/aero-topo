import os
import numpy as np
import trimesh
from typing import Tuple, Dict, Any, Optional
from app.geometry.camera import CameraIntrinsics

def depth_to_mesh(
    depth_map: np.ndarray,
    rgb_image: np.ndarray,
    camera: CameraIntrinsics,
    subsample: int = 2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Construct regular grid 3D surface mesh from depth map and RGB texture.
    Returns:
        vertices: float32 (V, 3)
        colors: float32 (V, 3) normalized [0, 1]
        faces: int32 (F, 3) triangle vertex indices
        metadata: Dict
    """
    points_3d = camera.project_depth_to_camera_space(depth_map)
    
    if subsample > 1:
        points_grid = points_3d[::subsample, ::subsample, :]
        colors_grid = rgb_image[::subsample, ::subsample, :]
    else:
        points_grid = points_3d
        colors_grid = rgb_image

    h, w, _ = points_grid.shape

    # Vertices & colors
    vertices = points_grid.reshape(-1, 3)
    colors = (colors_grid.reshape(-1, 3).astype(np.float32) / 255.0)

    # Generate regular quad grid faces (split into 2 triangles per quad)
    faces = []
    for r in range(h - 1):
        for c in range(w - 1):
            top_left = r * w + c
            top_right = top_left + 1
            bottom_left = (r + 1) * w + c
            bottom_right = bottom_left + 1

            # Triangle 1: top-left, bottom-left, top-right
            faces.append([top_left, bottom_left, top_right])
            # Triangle 2: top-right, bottom-left, bottom-right
            faces.append([top_right, bottom_left, bottom_right])

    faces = np.array(faces, dtype=np.int32)

    # Filter out faces with extreme depth discontinuities / edges
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    
    edge_len1 = np.linalg.norm(v1 - v0, axis=1)
    edge_len2 = np.linalg.norm(v2 - v1, axis=1)
    edge_len3 = np.linalg.norm(v0 - v2, axis=1)

    max_edge = np.maximum(edge_len1, np.maximum(edge_len2, edge_len3))
    # Threshold edge length to break invalid silhouette faces across depth jumps
    median_edge = np.median(max_edge)
    valid_faces = max_edge < (median_edge * 5.0 + 0.5)

    filtered_faces = faces[valid_faces]

    metadata = {
        "num_vertices": int(vertices.shape[0]),
        "num_faces": int(filtered_faces.shape[0]),
        "subsample_stride": subsample
    }

    return vertices, colors, filtered_faces, metadata

def save_mesh_obj(vertices: np.ndarray, colors: np.ndarray, faces: np.ndarray, file_path: str):
    """
    Save 3D surface mesh to OBJ file format with per-vertex colors.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    colors_uint8 = (colors * 255.0).clip(0, 255).astype(np.uint8)

    with open(file_path, "w") as f:
        f.write("# Aero-Topo Surface Mesh OBJ Export\n")
        for i in range(vertices.shape[0]):
            x, y, z = vertices[i]
            r, g, b = colors_uint8[i]
            f.write(f"v {x:.4f} {y:.4f} {z:.4f} {r/255.0:.3f} {g/255.0:.3f} {b/255.0:.3f}\n")
        
        for face in faces:
            # OBJ is 1-indexed
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

def export_mesh_glb(vertices: np.ndarray, colors: np.ndarray, faces: np.ndarray, file_path: str):
    """
    Export 3D surface mesh to GLB format using trimesh for Three.js loading.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    colors_uint8 = (colors * 255.0).clip(0, 255).astype(np.uint8)
    
    # Create RGBA colors
    colors_rgba = np.hstack([colors_uint8, np.full((colors_uint8.shape[0], 1), 255, dtype=np.uint8)])

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=colors_rgba, process=False)
    mesh.export(file_path, file_type="glb")
