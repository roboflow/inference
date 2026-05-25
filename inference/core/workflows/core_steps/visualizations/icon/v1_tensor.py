from inference.core.workflows.core_steps.common.wrap_consumer import (
    make_tensor_wrapper_block,
)
from inference.core.workflows.core_steps.visualizations.icon.v1 import (
    IconVisualizationBlockV1 as _NumpyImpl,
)

IconVisualizationBlockV1 = make_tensor_wrapper_block(_NumpyImpl)
