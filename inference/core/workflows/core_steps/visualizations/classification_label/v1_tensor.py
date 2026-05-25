from inference.core.workflows.core_steps.common.wrap_consumer import (
    make_tensor_wrapper_block,
)
from inference.core.workflows.core_steps.visualizations.classification_label.v1 import (
    ClassificationLabelVisualizationBlockV1 as _NumpyImpl,
)

ClassificationLabelVisualizationBlockV1 = make_tensor_wrapper_block(_NumpyImpl)
