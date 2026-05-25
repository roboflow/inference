from inference.core.workflows.core_steps.common.wrap_consumer import (
    make_tensor_wrapper_block,
)
from inference.core.workflows.core_steps.classical_cv.mask_edge_snap.v1 import (
    MaskEdgeSnapBlockV1 as _NumpyImpl,
)

MaskEdgeSnapBlockV1 = make_tensor_wrapper_block(_NumpyImpl)
