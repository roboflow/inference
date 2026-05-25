from inference.core.workflows.core_steps.common.wrap_consumer import (
    make_tensor_wrapper_block,
)
from inference.core.workflows.core_steps.classical_cv.mask_area_measurement.v1 import (
    MaskAreaMeasurementBlockV1 as _NumpyImpl,
)

MaskAreaMeasurementBlockV1 = make_tensor_wrapper_block(_NumpyImpl)
