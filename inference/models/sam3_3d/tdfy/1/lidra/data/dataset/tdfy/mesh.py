import os
from typing import Any, List

import numpy as np
import pytorch3d as pt3d
import torch
import trimesh
from pytorch3d.renderer import TexturesAtlas, TexturesUV, TexturesVertex
from pytorch3d.structures import Meshes, packed_to_list
import pytorch3d.io


def convert_trimesh_to_pt3d(trimesh_mesh, resize_texture=None):
    verts = torch.tensor(trimesh_mesh.vertices, dtype=torch.float32)
    faces = torch.tensor(trimesh_mesh.faces, dtype=torch.int64)

    # Get UV coordinates and texture image if available
    if hasattr(trimesh_mesh.visual, "uv") and hasattr(trimesh_mesh.visual, "material"):
        # Get UV coordinates
        verts_uvs = torch.tensor(trimesh_mesh.visual.uv, dtype=torch.float32)

        # Get texture image
        if hasattr(trimesh_mesh.visual.material, "image"):
            texture_image = trimesh_mesh.visual.material.image
            if resize_texture is not None:
                texture_image = texture_image.resize(resize_texture)
            texture_image = torch.from_numpy(np.array(texture_image)).float() / 255.0
            texture_image = texture_image[None]  # Add batch dimension [1, H, W, 3]
        else:
            # Default texture if no image
            texture_image = torch.ones((1, 4, 4, 3), dtype=torch.float32) * 0.5

        # Create TexturesUV with correct texture format
        textures = pt3d.renderer.TexturesUV(
            maps=texture_image,
            faces_uvs=[faces],  # Assuming UV faces match geometry faces
            verts_uvs=[verts_uvs],
            align_corners=False,
        )

    else:
        # Fallback to simple vertex colors if no UV mapping
        vertex_colors = torch.ones((len(verts), 3), dtype=torch.float32) * 0.5
        textures = pt3d.renderer.TexturesVertex([vertex_colors])

    mesh = Meshes(verts=[verts], faces=[faces], textures=textures)
    return mesh


# To visualize using plotly, we need to convert the textures to vertex textures
# Which can significantly degrade the resolution of the texture
def textures_uv_to_textures_vertex(
    textures_uv: TexturesUV, meshes: Meshes
) -> TexturesVertex:
    verts_colors_packed = torch.zeros_like(meshes.verts_packed())
    verts_colors_packed[meshes.faces_packed()] = (
        textures_uv.faces_verts_textures_packed()
    )  # (*)
    return TexturesVertex(
        packed_to_list(verts_colors_packed, meshes.num_verts_per_mesh())
    )
