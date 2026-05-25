from inference.core.workflows.core_steps.common.wrap_consumer import (
    make_tensor_wrapper_block,
)
from inference.core.workflows.core_steps.visualizations.mask.v1 import (
    MaskVisualizationBlockV1 as _NumpyImpl,
)

MaskVisualizationBlockV1 = make_tensor_wrapper_block(_NumpyImpl)
