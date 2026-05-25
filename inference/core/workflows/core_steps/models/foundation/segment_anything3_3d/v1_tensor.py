from inference.core.workflows.core_steps.common.wrap_consumer import (
    make_tensor_wrapper_block,
)
from inference.core.workflows.core_steps.models.foundation.segment_anything3_3d.v1 import (
    SegmentAnything3_3D_ObjectsBlockV1 as _NumpyImpl,
)

SegmentAnything3_3D_ObjectsBlockV1 = make_tensor_wrapper_block(_NumpyImpl)
