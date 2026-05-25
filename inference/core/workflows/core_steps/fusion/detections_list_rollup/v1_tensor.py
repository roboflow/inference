from inference.core.workflows.core_steps.common.wrap_consumer import (
    make_tensor_wrapper_block,
)
from inference.core.workflows.core_steps.fusion.detections_list_rollup.v1 import (
    DetectionsListRollUpBlockV1 as _NumpyImpl,
)

DetectionsListRollUpBlockV1 = make_tensor_wrapper_block(_NumpyImpl)
