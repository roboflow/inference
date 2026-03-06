import os

from lidra.data.dataset.tdfy.trellis.dataset import PreProcessor
from lidra.data.dataset.tdfy.trellis.dataset import (
    PerSubsetDataset as OldTrellisDataset,
)
from lidra.data.dataset.flexiset.dataset import FlexiDataset
from lidra.data.dataset.tdfy.trellis.pose_loader import load_trellis_pose
from lidra.data.dataset.tdfy.flexi.loaders import all_loaders
from lidra.data.dataset.tdfy.flexi.transforms import all_transforms

from lidra.data.dataset.tdfy.metadata_filter import (
    custom_metadata_filter,
    data_query_filter,
)

from lidra.data.dataset.tdfy.trellis.pose_loader import R3


SUBSET_TO_KWARGS = {
    "GSO": dict(pointmap=False),
    "ObjaverseXL_sketchfab": dict(
        pointmap=False,
        pointmap_scale=False,
    ),
    "ObjaverseXL_github": dict(
        pointmap=False, pointmap_scale=False, trellis_mesh=False
    ),
    "HSSD": dict(
        pointmap=False,
        pointmap_scale=False,
    ),
    "ABO": dict(
        pointmap=False,
        pointmap_scale=False,
        sparse_latent=False,
    ),
    "r3_25_morphing": dict(),
    "rp_25_morphing": dict(
        pointmap=False,
        sparse_latent=False,
    ),
    "r3_25_March_morphing": dict(),
    "variantF_v1_complement": dict(
        pointmap=False,
        pointmap_scale=False,
        trellis_mesh=False,
        sparse_latent=False,
    ),
    "variantF_v1": dict(
        pointmap=False,
        pointmap_scale=False,
        trellis_mesh=False,
        sparse_latent=False,
    ),
    "variantF_v2": dict(
        pointmap=False,
        pointmap_scale=False,
        trellis_mesh=False,
        sparse_latent=False,
    ),
}

MODALITY_TO_LOADER_TO_OUTPUTS = dict(
    original_mesh=("original_mesh_path",),
    sparse_latent=("sparse_t",),
    structured_latent=("mean", "logvar"),
    trellis_mesh=("trellis_mesh", "mesh_vertices", "mesh_faces"),
    view_image=("image_path", "rgb_image", "rgba_image", "transformed_image"),
    view_mask=("rgb_image_mask", "transformed_mask"),
    view_pose=(
        "instance_quaternion_l2c",
        "instance_position_l2c",
        "instance_scale_l2c",
        "camera_K",
    ),
    pointmap=("pointmap", "pointmap_colors"),
    pointmap_scale=("pointmap_scale",),
)

MODALITY_TO_LOADER_TO_OUTPUT_MAPPING = dict(
    original_mesh={},
    sparse_latent={},
    structured_latent={},
    trellis_mesh={},
    view_image={"image": "transformed_image"},
    view_mask={"mask": "transformed_mask"},
    view_pose={},
    pointmap={},
)


def outputs_from_kwargs(**kwargs):
    outputs = []
    output_mappings = {}

    for modality, loader_outputs in MODALITY_TO_LOADER_TO_OUTPUTS.items():
        if kwargs.get(modality, True):
            # if the modality is enabled in kwargs, add its outputs
            outputs.extend(loader_outputs)
            # also update the output mapping if applicable
            output_mapping = MODALITY_TO_LOADER_TO_OUTPUT_MAPPING.get(modality, {})
            output_mappings.update(output_mapping)

    return outputs, output_mappings


def trellis_dataset(
    path: str,
    metadata_filename: str = "metadata.csv",
    metadata_filter=custom_metadata_filter(
        (data_query_filter("num_voxels>0 and cond_rendered"),)
    ),
    pose_loader=load_trellis_pose,
    preprocessor: PreProcessor = None,
):
    dataset_name = os.path.basename(path)
    if not dataset_name in SUBSET_TO_KWARGS:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. Available options are: {set(SUBSET_TO_KWARGS.keys())}"
        )
    kwargs = SUBSET_TO_KWARGS[dataset_name]

    outputs, output_mapping = outputs_from_kwargs(**kwargs)

    ds = FlexiDataset(
        path,
        metadata_filename=metadata_filename,
        metadata_filter=metadata_filter,
        loaders=all_loaders(pose_loader=pose_loader),
        transforms=all_transforms(preprocessor),
        outputs=outputs,
        output_mapping=output_mapping,
    )

    return ds


def trellis_from_old_dataset(dataset: OldTrellisDataset):
    return trellis_dataset(
        dataset.path,
        metadata_filename=dataset.metadata_fname,
        metadata_filter=dataset.metadata_filter,
        pose_loader=dataset.pose_loader,
        preprocessor=dataset.preprocessor,
    )
