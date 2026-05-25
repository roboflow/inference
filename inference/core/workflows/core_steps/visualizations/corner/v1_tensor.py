from inference.core.workflows.core_steps.common.wrap_consumer import (
    make_tensor_wrapper_block,
)
from inference.core.workflows.core_steps.visualizations.corner.v1 import (
    CornerVisualizationBlockV1 as _NumpyImpl,
)

CornerVisualizationBlockV1 = make_tensor_wrapper_block(_NumpyImpl)
