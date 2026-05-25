from inference.core.workflows.core_steps.common.wrap_consumer import (
    make_tensor_wrapper_block,
)
from inference.core.workflows.core_steps.transformations.detections_merge.v1 import (
    DetectionsMergeBlockV1 as _NumpyImpl,
)

DetectionsMergeBlockV1 = make_tensor_wrapper_block(_NumpyImpl)
