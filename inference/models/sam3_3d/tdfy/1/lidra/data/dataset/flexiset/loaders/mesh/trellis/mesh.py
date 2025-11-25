import os

from lidra.data.dataset.flexiset.loaders.base import Base
from lidra.data.dataset.flexiset.loaders.mesh.pytorch3d.mesh import (
    Mesh as PyTorch3DMesh,
)


class Mesh(Base):
    mesh_loader = PyTorch3DMesh()

    def _load(self, path, uid):
        filepath = os.path.join(path, uid, "mesh.ply")
        return self.mesh_loader._load(path=filepath)
