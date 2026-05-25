from inference.core.workflows.core_steps.common.wrap_consumer import (
    make_tensor_wrapper_block,
)
from inference.core.workflows.core_steps.visualizations.line_zone.v1 import (
    LineCounterZoneVisualizationBlockV1 as _NumpyImpl,
)

LineCounterZoneVisualizationBlockV1 = make_tensor_wrapper_block(_NumpyImpl)
