from inference.core.workflows.core_steps.common.wrap_consumer import (
    make_tensor_wrapper_block,
)
from inference.core.workflows.core_steps.visualizations.color.v1 import (
    ColorVisualizationBlockV1 as _NumpyImpl,
)

ColorVisualizationBlockV1 = make_tensor_wrapper_block(_NumpyImpl)
