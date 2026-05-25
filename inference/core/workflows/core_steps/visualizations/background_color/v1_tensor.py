from inference.core.workflows.core_steps.common.wrap_consumer import (
    make_tensor_wrapper_block,
)
from inference.core.workflows.core_steps.visualizations.background_color.v1 import (
    BackgroundColorVisualizationBlockV1 as _NumpyImpl,
)

BackgroundColorVisualizationBlockV1 = make_tensor_wrapper_block(_NumpyImpl)
