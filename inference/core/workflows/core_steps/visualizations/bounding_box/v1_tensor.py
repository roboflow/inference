from inference.core.workflows.core_steps.common.wrap_consumer import (
    make_tensor_wrapper_block,
)
from inference.core.workflows.core_steps.visualizations.bounding_box.v1 import (
    BoundingBoxVisualizationBlockV1 as _NumpyImpl,
)

BoundingBoxVisualizationBlockV1 = make_tensor_wrapper_block(_NumpyImpl)
