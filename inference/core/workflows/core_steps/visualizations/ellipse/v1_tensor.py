from inference.core.workflows.core_steps.common.wrap_consumer import (
    make_tensor_wrapper_block,
)
from inference.core.workflows.core_steps.visualizations.ellipse.v1 import (
    EllipseVisualizationBlockV1 as _NumpyImpl,
)

EllipseVisualizationBlockV1 = make_tensor_wrapper_block(_NumpyImpl)
