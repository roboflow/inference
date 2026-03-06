from pytorch3d.structures import Meshes

from lidra.data.dataset.flexiset.loaders.base import Base


class Faces(Base):
    def _load(self, mesh):
        assert isinstance(mesh, Meshes)
        return mesh.faces_packed()
