"""Workload declarations of the transformation blocks.

Cropping, geometric transformation and detection post-processing are three
different costs and must not collapse into one label. Blocks driven by UQL
declare the expression evaluation they perform, and the tracker blocks declare
tracking plus the state caveat the legacy API already carried.
"""

from roboflow_workflows.core_steps.transformations.absolute_static_crop.v1 import (
    BlockManifest as AbsoluteStaticCropManifest,
)
from roboflow_workflows.core_steps.transformations.byte_tracker.v3 import (
    ByteTrackerBlockManifest,
)
from roboflow_workflows.core_steps.transformations.detection_offset.v1 import (
    BlockManifest as DetectionOffsetManifest,
)
from roboflow_workflows.core_steps.transformations.dynamic_crop.v1 import (
    BlockManifest as DynamicCropManifest,
)
from roboflow_workflows.core_steps.transformations.perspective_correction.v1 import (
    PerspectiveCorrectionManifest,
)
from roboflow_workflows.execution_engine.entities.workload import WorkOperation


def test_crop_blocks_declare_cropping_not_a_generic_transform() -> None:
    absolute = AbsoluteStaticCropManifest(
        type="roboflow_core/absolute_static_crop@v1",
        name="crop",
        image="$inputs.image",
        x_center=100,
        y_center=100,
        width=50,
        height=50,
    )
    dynamic = DynamicCropManifest(
        type="roboflow_core/dynamic_crop@v1",
        name="dynamic_crop",
        images="$inputs.image",
        predictions="$steps.model.predictions",
    )
    assert absolute.discover_work_operations() == [WorkOperation.IMAGE_CROP]
    assert dynamic.discover_work_operations() == [WorkOperation.IMAGE_CROP]


def test_detection_post_processing_is_not_an_image_operation() -> None:
    manifest = DetectionOffsetManifest(
        type="roboflow_core/detection_offset@v1",
        name="offset",
        predictions="$steps.model.predictions",
        offset_width=10,
        offset_height=10,
    )
    operations = manifest.discover_work_operations()
    assert operations == [WorkOperation.DETECTION_PROCESSING]
    assert WorkOperation.IMAGE_TRANSFORM not in operations


def test_perspective_correction_declares_both_of_its_outputs() -> None:
    """It warps the image AND rewrites detection coordinates."""
    manifest = PerspectiveCorrectionManifest(
        type="roboflow_core/perspective_correction@v1",
        name="perspective",
        images="$inputs.image",
        predictions="$steps.model.predictions",
        perspective_polygons="$steps.zones.zones",
    )
    assert manifest.discover_work_operations() == [
        WorkOperation.IMAGE_TRANSFORM,
        WorkOperation.DETECTION_PROCESSING,
    ]


def test_byte_tracker_declares_tracking_and_the_legacy_state_caveat() -> None:
    manifest = ByteTrackerBlockManifest(
        type="roboflow_core/byte_tracker@v3",
        name="tracker",
        image="$inputs.image",
        detections="$steps.model.predictions",
    )
    assert manifest.discover_work_operations() == [WorkOperation.TRACKING]
    portable_codes = [
        restriction.code for restriction in manifest.discover_portable_restrictions()
    ]
    assert portable_codes == [
        "stateful_video_state_resets_on_stateless_http",
        "temporal_block_no_benefit_on_still_image",
    ]
    # the portable declaration has exactly one entry per legacy restriction
    assert len(ByteTrackerBlockManifest.get_restrictions()) == len(portable_codes)
