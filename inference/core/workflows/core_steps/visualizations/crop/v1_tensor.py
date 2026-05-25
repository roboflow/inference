from inference.core.workflows.core_steps.common.wrap_consumer import (
    make_tensor_wrapper_block,
)
from inference.core.workflows.core_steps.visualizations.crop.v1 import (
    CropVisualizationBlockV1 as _NumpyImpl,
)

CropVisualizationBlockV1 = make_tensor_wrapper_block(_NumpyImpl)
