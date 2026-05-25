from inference.core.workflows.core_steps.common.wrap_consumer import (
    make_tensor_wrapper_block,
)
from inference.core.workflows.core_steps.classical_cv.distance_measurement.v1 import (
    DistanceMeasurementBlockV1 as _NumpyImpl,
)

DistanceMeasurementBlockV1 = make_tensor_wrapper_block(_NumpyImpl)
