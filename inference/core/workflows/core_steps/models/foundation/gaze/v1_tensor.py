from inference.core.workflows.core_steps.common.wrap_consumer import (
    make_tensor_wrapper_block,
)
from inference.core.workflows.core_steps.models.foundation.gaze.v1 import (
    GazeBlockV1 as _NumpyImpl,
)

GazeBlockV1 = make_tensor_wrapper_block(_NumpyImpl)
