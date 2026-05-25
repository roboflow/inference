from inference.core.workflows.core_steps.common.wrap_consumer import (
    make_tensor_wrapper_block,
)
from inference.core.workflows.core_steps.transformations.dynamic_crop.v1 import (
    DynamicCropBlockV1 as _NumpyImpl,
)

# Wrap-and-delegate: dynamic_crop's per-detection mask slicing is heavily
# sv.Detections.data-shaped. Native tensor port is deferred; the wrapper
# materialises predictions at the boundary and reuses the numpy logic.
DynamicCropBlockV1 = make_tensor_wrapper_block(_NumpyImpl)
