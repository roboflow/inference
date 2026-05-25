from inference.core.workflows.core_steps.common.wrap_consumer import (
    make_tensor_wrapper_block,
)
from inference.core.workflows.core_steps.visualizations.keypoint.v1 import (
    KeypointVisualizationBlockV1 as _NumpyImpl,
)

KeypointVisualizationBlockV1 = make_tensor_wrapper_block(_NumpyImpl)
