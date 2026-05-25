from inference.core.workflows.core_steps.common.wrap_consumer import (
    make_tensor_wrapper_block,
)
from inference.core.workflows.core_steps.visualizations.pixelate.v1 import (
    PixelateVisualizationBlockV1 as _NumpyImpl,
)

PixelateVisualizationBlockV1 = make_tensor_wrapper_block(_NumpyImpl)
