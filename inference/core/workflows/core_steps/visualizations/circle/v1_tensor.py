from inference.core.workflows.core_steps.common.wrap_consumer import (
    make_tensor_wrapper_block,
)
from inference.core.workflows.core_steps.visualizations.circle.v1 import (
    CircleVisualizationBlockV1 as _NumpyImpl,
)

CircleVisualizationBlockV1 = make_tensor_wrapper_block(_NumpyImpl)
