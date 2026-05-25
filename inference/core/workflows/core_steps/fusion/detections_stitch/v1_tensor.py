from inference.core.workflows.core_steps.common.wrap_consumer import (
    make_tensor_wrapper_block,
)
from inference.core.workflows.core_steps.fusion.detections_stitch.v1 import (
    DetectionsStitchBlockV1 as _NumpyImpl,
)

DetectionsStitchBlockV1 = make_tensor_wrapper_block(_NumpyImpl)
