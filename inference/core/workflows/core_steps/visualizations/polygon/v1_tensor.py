from inference.core.workflows.core_steps.common.wrap_consumer import (
    make_tensor_wrapper_block,
)
from inference.core.workflows.core_steps.visualizations.polygon.v1 import (
    PolygonVisualizationBlockV1 as _NumpyImpl,
)

PolygonVisualizationBlockV1 = make_tensor_wrapper_block(_NumpyImpl)
