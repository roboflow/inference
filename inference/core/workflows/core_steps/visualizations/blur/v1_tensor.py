from inference.core.workflows.core_steps.common.wrap_consumer import (
    make_tensor_wrapper_block,
)
from inference.core.workflows.core_steps.visualizations.blur.v1 import (
    BlurVisualizationBlockV1 as _NumpyImpl,
)

BlurVisualizationBlockV1 = make_tensor_wrapper_block(_NumpyImpl)
