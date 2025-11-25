from pytorch3d.structures import Meshes

from lidra.data.dataset.flexiset.loaders.base import Base


class Vertices(Base):
    def _load(self, mesh):
        assert isinstance(mesh, Meshes)
        return mesh.verts_packed()
