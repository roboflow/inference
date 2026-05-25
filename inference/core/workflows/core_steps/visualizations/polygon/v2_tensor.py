from inference.core.workflows.core_steps.common.wrap_consumer import (
    make_tensor_wrapper_block,
)
from inference.core.workflows.core_steps.visualizations.polygon.v2 import (
    PolygonVisualizationBlockV2 as _NumpyImpl,
)

PolygonVisualizationBlockV2 = make_tensor_wrapper_block(_NumpyImpl)
