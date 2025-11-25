from pytorch3d.io import IO
from pytorch3d.io.experimental_gltf_io import MeshGlbFormat

pt3dio = IO()
pt3dio.register_meshes_format(MeshGlbFormat())


def load_trellis_mesh(mesh_path: str):
    trellis_mesh = pt3dio.load_mesh(mesh_path)
    return {
        "mesh_vertices": trellis_mesh.verts_packed(),
        "mesh_faces": trellis_mesh.faces_packed(),
    }
