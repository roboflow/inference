from lidra.data.dataset.flexiset.flexi.loader import Loader as FlexiLoader
from lidra.data.dataset.flexiset.loaders.ops.copy import Copy
from lidra.data.dataset.flexiset.loaders.latent.sparse_latent import SparseLatent
from lidra.data.dataset.flexiset.loaders.latent.structured_latent import (
    StructuredLatent,
)
from lidra.data.dataset.flexiset.loaders.mesh.trellis.mesh import Mesh as TrellisMesh
from lidra.data.dataset.flexiset.loaders.mesh.pytorch3d.faces import (
    Faces as TrellisFaces,
)
from lidra.data.dataset.flexiset.loaders.mesh.pytorch3d.vertices import (
    Vertices as TrellisVertices,
)
from lidra.data.dataset.flexiset.loaders.pointmap.pointmap import (
    Pointmap,
    PointmapScale,
)
from lidra.data.dataset.flexiset.loaders.image.rgb import RGB
from lidra.data.dataset.flexiset.loaders.image.rgba import RGBA
from lidra.data.dataset.flexiset.loaders.mask.from_alpha import (
    FromAlpha as MaskFromAlpha,
)
from lidra.data.dataset.flexiset.loaders.view.random_view_path_and_pose import (
    RandomViewPathAndPose,
)
from lidra.data.dataset.flexiset.loaders.mesh.local_path import (
    LocalPath as MeshLocalPath,
)
from lidra.data.dataset.flexiset.loaders.view.random_view import RandomViews
from lidra.data.dataset.flexiset.loaders.rendering.blender import Blender


from lidra.data.dataset.tdfy.trellis.pose_loader import load_trellis_pose


def all_loaders(
    sparse_latent_path="latents/dinov2_vitl14_reg_slat_enc_swin8_B_64l8_fp16",
    structured_latent_path="ss_latents/ss_enc_conv3d_16l8_fp16",
    pose_loader=load_trellis_pose,
):
    return [
        FlexiLoader(
            "sparse_t",
            SparseLatent(),
            subpath=sparse_latent_path,
        ),
        FlexiLoader(
            outputs=("mean", "logvar"),
            loader=StructuredLatent(),
            subpath=structured_latent_path,
        ),
        FlexiLoader(
            "trellis_mesh",
            TrellisMesh(),
            subpath="renders",
        ),
        FlexiLoader(
            "mesh_vertices",
            TrellisVertices(),
            input_mapping={"mesh": "trellis_mesh"},
        ),
        FlexiLoader(
            "mesh_faces",
            TrellisFaces(),
            input_mapping={"mesh": "trellis_mesh"},
        ),
        FlexiLoader(
            (
                "image_path",
                "instance_quaternion_l2c",
                "instance_position_l2c",
                "instance_scale_l2c",
                "camera_K",
            ),
            RandomViewPathAndPose(pose_loader),
            subpath="renders_cond",
        ),
        FlexiLoader(
            "rgb_image_mask",
            MaskFromAlpha(threshold=0.0),
            input_mapping={"rgba": "rgba_image"},
        ),
        FlexiLoader(
            "rgb_image",
            RGB(),
            input_mapping={"image": "rgba_image"},
        ),
        FlexiLoader(
            "transformed_image",
            Copy(),
            input_mapping={"data": "rgb_image"},
        ),
        FlexiLoader(
            "transformed_mask",
            Copy(),
            input_mapping={"data": "rgb_image_mask"},
        ),
        FlexiLoader(
            "rgba_image",
            RGBA(),
            input_mapping={"path": "image_path"},
        ),
        FlexiLoader(
            ("pointmap", "pointmap_colors"),
            Pointmap(),
            input_mapping={"image": "rgb_image"},
            subpath="/checkpoint/3dfy/shared/r3_data/pointmaps/moge",
        ),
        FlexiLoader(
            "pointmap_scale",
            PointmapScale(),
            input_mapping={"pointmap": "pointmap"},
        ),
        FlexiLoader(
            "original_mesh_path",
            MeshLocalPath(),
        ),
        FlexiLoader(
            "rendering.random_views",
            RandomViews(),
            input_mapping={"n": "n_views"},
        ),
        FlexiLoader(
            (
                "rendering.images",
                "rendering.cam_matrices",
                "rendering.scale",
                "rendering.offset",
            ),
            Blender(engine="BLENDER_EEVEE_NEXT"),
            input_mapping={
                "path": "original_mesh_path",
                "views": "rendering.random_views",
            },
        ),
    ]
