from inference.core.workflows.core_steps.common.wrap_consumer import (
    make_tensor_wrapper_block,
)
from inference.core.workflows.core_steps.visualizations.trace.v1 import (
    TraceVisualizationBlockV1 as _NumpyImpl,
)

TraceVisualizationBlockV1 = make_tensor_wrapper_block(_NumpyImpl)
