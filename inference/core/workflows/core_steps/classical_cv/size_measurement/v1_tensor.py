from inference.core.workflows.core_steps.common.wrap_consumer import (
    make_tensor_wrapper_block,
)
from inference.core.workflows.core_steps.classical_cv.size_measurement.v1 import (
    SizeMeasurementBlockV1 as _NumpyImpl,
)

SizeMeasurementBlockV1 = make_tensor_wrapper_block(_NumpyImpl)
