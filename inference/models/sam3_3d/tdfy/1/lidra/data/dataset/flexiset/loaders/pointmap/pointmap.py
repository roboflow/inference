import os
import torch
import numpy as np
from PIL import Image
from loguru import logger
from pytorch3d.transforms import Transform3d

from lidra.data.dataset.flexiset.loaders.base import Base
from lidra.data.dataset.flexiset.loaders.mesh.pytorch3d.ply_vertices_and_faces import (
    PlyVerticesAndFaces,
)
from lidra.data.dataset.flexiset.loaders.json.from_file import FromFile as Json
from lidra.data.dataset.tdfy.trellis.pose_loader import R3


# COMMENT(Pierre) very ugly below, does a lot of stuff, will have to be refactored
class Pointmap(Base):
    mesh_loader = PlyVerticesAndFaces()
    metadata_loader = Json()

    def _get_paths(self, path, image_basename):
        ply_path = os.path.join(path, image_basename, f"{image_basename}.ply")
        metadata_path = os.path.join(
            path, image_basename, f"{image_basename}_metadata.json"
        )
        return ply_path, metadata_path

    def _get_pointmap_colors(self, raw_image: torch.Tensor, pointmap: torch.Tensor):
        # get the colors of the pointmap
        if raw_image.shape[0] == 3:
            raw_image = raw_image.permute(1, 2, 0)
        colors_tensor = Image.fromarray(
            (raw_image[..., :3] * 255).numpy().astype(np.uint8)
        ).resize((pointmap.shape[1], pointmap.shape[0]))
        colors_tensor = torch.from_numpy(np.array(colors_tensor)).float() / 255.0
        return colors_tensor

    def _transform_pointmap(self, pointmap):
        camera_convention_transform = Transform3d().rotate(
            R3.r3_camera_to_pytorch3d_camera().rotation
        )
        pointmap = camera_convention_transform.transform_points(pointmap)
        return pointmap

    def _load(self, path, metadata, image):
        assert isinstance(image, torch.Tensor)
        image_basename = metadata["image_basename"]

        # get paths
        ply_path, metadata_path = self._get_paths(path, image_basename)

        # load pointmap and colors
        verts, _ = self.mesh_loader._load(ply_path)
        pointmap_metadata = self.metadata_loader._load(metadata_path)
        image_dims = pointmap_metadata["image_size"]
        pointmap = verts.reshape(image_dims[1], image_dims[0], 3)
        pointmap = self._transform_pointmap(pointmap)
        pointmap_colors = self._get_pointmap_colors(image, pointmap)

        return pointmap, pointmap_colors


# the things we do for consistency ...
class PointmapScale(Base):
    @staticmethod
    def _get_pointmap_scale(pointmap: torch.Tensor):
        assert pointmap.shape[-1] == 3, "Pointmap must be 3D"
        pointmap_scale = pointmap.norm(dim=-1, keepdim=True)
        return pointmap_scale.nanmedian()

    def _load(self, uid, pointmap):
        pointmap_scale = PointmapScale._get_pointmap_scale(pointmap)
        if torch.isnan(pointmap_scale).any():
            logger.warning(
                f"NaN values detected in pointmap_scale for sha256: {uid} (pointmap must be all nans)"
            )
            return None
        return pointmap_scale
