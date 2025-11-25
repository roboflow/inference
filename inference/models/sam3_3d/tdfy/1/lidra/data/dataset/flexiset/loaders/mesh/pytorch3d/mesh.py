from pytorch3d.io import IO
from pytorch3d.io.experimental_gltf_io import MeshGlbFormat

from lidra.data.dataset.flexiset.loaders.base import Base


class Mesh(Base):
    IO = IO()

    def _load(self, path):
        return Mesh.IO.load_mesh(path)


Mesh.IO.register_meshes_format(MeshGlbFormat())
