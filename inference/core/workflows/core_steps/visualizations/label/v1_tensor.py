from inference.core.workflows.core_steps.common.wrap_consumer import (
    make_tensor_wrapper_block,
)
from inference.core.workflows.core_steps.visualizations.label.v1 import (
    LabelVisualizationBlockV1 as _NumpyImpl,
)

LabelVisualizationBlockV1 = make_tensor_wrapper_block(_NumpyImpl)
