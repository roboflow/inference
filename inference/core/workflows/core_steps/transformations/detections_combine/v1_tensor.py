from inference.core.workflows.core_steps.common.wrap_consumer import (
    make_tensor_wrapper_block,
)
from inference.core.workflows.core_steps.transformations.detections_combine.v1 import (
    DetectionsCombineBlockV1 as _NumpyImpl,
)

DetectionsCombineBlockV1 = make_tensor_wrapper_block(_NumpyImpl)
