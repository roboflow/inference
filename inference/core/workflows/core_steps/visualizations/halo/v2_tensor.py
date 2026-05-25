from inference.core.workflows.core_steps.common.wrap_consumer import (
    make_tensor_wrapper_block,
)
from inference.core.workflows.core_steps.visualizations.halo.v2 import (
    HaloVisualizationBlockV2 as _NumpyImpl,
)

HaloVisualizationBlockV2 = make_tensor_wrapper_block(_NumpyImpl)
