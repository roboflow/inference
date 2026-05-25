from inference.core.workflows.core_steps.common.wrap_consumer import (
    make_tensor_wrapper_block,
)
from inference.core.workflows.core_steps.transformations.detections_transformation.v1 import (
    DetectionsTransformationBlockV1 as _NumpyImpl,
)

DetectionsTransformationBlockV1 = make_tensor_wrapper_block(_NumpyImpl)
