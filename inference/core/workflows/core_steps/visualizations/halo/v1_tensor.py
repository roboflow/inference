from inference.core.workflows.core_steps.common.wrap_consumer import (
    make_tensor_wrapper_block,
)
from inference.core.workflows.core_steps.visualizations.halo.v1 import (
    HaloVisualizationBlockV1 as _NumpyImpl,
)

HaloVisualizationBlockV1 = make_tensor_wrapper_block(_NumpyImpl)
