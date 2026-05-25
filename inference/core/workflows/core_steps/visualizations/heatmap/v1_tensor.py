from inference.core.workflows.core_steps.common.wrap_consumer import (
    make_tensor_wrapper_block,
)
from inference.core.workflows.core_steps.visualizations.heatmap.v1 import (
    HeatmapVisualizationBlockV1 as _NumpyImpl,
)

HeatmapVisualizationBlockV1 = make_tensor_wrapper_block(_NumpyImpl)
