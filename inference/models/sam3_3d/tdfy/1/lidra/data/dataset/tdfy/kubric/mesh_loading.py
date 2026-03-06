from typing import Any, List, Tuple
import os
from iopath.common.file_io import PathHandler, PathManager
import pytorch3d as pt3d
import trimesh
import numpy as np
import torch
from PIL import Image
import json


################################################################################
# Trimesh loading
################################################################################
def load_gso_kubric_mesh(base_dir: str, asset_id: str) -> Tuple[trimesh.Trimesh, dict]:
    """Load a GSO mesh using trimesh."""
    mesh_path = os.path.join(base_dir, asset_id, "visual_geometry.obj")
    mesh = trimesh.load_mesh(mesh_path, process=False)
    data_json = json.load(open(os.path.join(base_dir, asset_id, "data.json")))
    return mesh, data_json


def load_gso_official_mesh(base_dir: str, asset_id: str) -> trimesh.Trimesh:
    """Load a GSO mesh using trimesh."""
    mesh_path = os.path.join(base_dir, asset_id, "meshes", "model.obj")
    mesh = trimesh.load_mesh(mesh_path, process=False)

    # Create and apply texture to mesh
    if hasattr(mesh.visual, "material"):
        texture_path = os.path.join(
            base_dir, asset_id, "materials", "textures", "texture.png"
        )
        texture = Image.open(texture_path)
        mesh.visual.material.image = texture
    else:
        raise ValueError("Mesh does not have a material")

    return mesh


################################################################################
# PT3D loading
################################################################################
class CustomPathHandler(PathHandler):
    def __init__(self, base_dir: str, remap_from: str, remap_to: str):
        super().__init__()
        self.base_dir = (
            base_dir  # e.g. "/checkpoint/3dfy/shared/datasets/gso_dataset/raw"
        )
        self.remap_from = remap_from
        self.remap_to = remap_to

    def _get_local_path(self, path: str, force: bool = False, **kwargs: Any) -> str:
        if self.remap_from in path:
            # Replace 'meshes/texture.png' with 'materials/textures/texture.png'
            return path.replace(self.remap_from, self.remap_to)
        return path

    def _get_supported_prefixes(self) -> List[str]:
        # Only handle paths that start with our base_dir
        return [self.base_dir]

    def _exists(self, path: str, **kwargs: Any) -> bool:
        local_path = self._get_local_path(path)
        return os.path.exists(local_path)

    def _isfile(self, path: str, **kwargs: Any) -> bool:
        local_path = self._get_local_path(path)
        return os.path.isfile(local_path)

    def _isdir(self, path: str, **kwargs: Any) -> bool:
        local_path = self._get_local_path(path)
        return os.path.isdir(local_path)

    def _open(self, path: str, mode: str = "r", **kwargs: Any):
        local_path = self._get_local_path(path)
        return open(local_path, mode, **kwargs)


def load_gso_official_meshes_pt3d(
    base_dir: str, asset_name: str, create_texture_atlas: bool = False
):
    # Create and register the handler
    path_manager = PathManager()
    custom_handler = CustomPathHandler(
        base_dir, "meshes/texture.png", "materials/textures/texture.png"
    )
    path_manager.register_handler(custom_handler)

    # Now try loading the mesh
    mesh_with_uv = pt3d.io.load_objs_as_meshes(
        [os.path.join(base_dir, f"{asset_name}/meshes/model.obj")],
        path_manager=path_manager,
        load_textures=True,
        create_texture_atlas=create_texture_atlas,
    )
    return mesh_with_uv
