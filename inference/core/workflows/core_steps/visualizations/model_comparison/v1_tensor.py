from inference.core.workflows.core_steps.common.wrap_consumer import (
    make_tensor_wrapper_block,
)
from inference.core.workflows.core_steps.visualizations.model_comparison.v1 import (
    ModelComparisonVisualizationBlockV1 as _NumpyImpl,
)

ModelComparisonVisualizationBlockV1 = make_tensor_wrapper_block(_NumpyImpl)
