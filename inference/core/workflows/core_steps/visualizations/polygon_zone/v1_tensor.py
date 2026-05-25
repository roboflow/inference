from inference.core.workflows.core_steps.common.wrap_consumer import (
    make_tensor_wrapper_block,
)
from inference.core.workflows.core_steps.visualizations.polygon_zone.v1 import (
    PolygonZoneVisualizationBlockV1 as _NumpyImpl,
)

PolygonZoneVisualizationBlockV1 = make_tensor_wrapper_block(_NumpyImpl)
