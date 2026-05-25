from inference.core.workflows.core_steps.common.wrap_consumer import (
    make_tensor_wrapper_block,
)
from inference.core.workflows.core_steps.visualizations.dot.v1 import (
    DotVisualizationBlockV1 as _NumpyImpl,
)

DotVisualizationBlockV1 = make_tensor_wrapper_block(_NumpyImpl)
