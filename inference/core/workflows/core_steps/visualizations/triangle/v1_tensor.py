from inference.core.workflows.core_steps.common.wrap_consumer import (
    make_tensor_wrapper_block,
)
from inference.core.workflows.core_steps.visualizations.triangle.v1 import (
    TriangleVisualizationBlockV1 as _NumpyImpl,
)

TriangleVisualizationBlockV1 = make_tensor_wrapper_block(_NumpyImpl)
