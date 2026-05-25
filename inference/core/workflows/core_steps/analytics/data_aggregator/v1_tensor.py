from inference.core.workflows.core_steps.common.wrap_consumer import (
    make_tensor_wrapper_block,
)
from inference.core.workflows.core_steps.analytics.data_aggregator.v1 import (
    DataAggregatorBlockV1 as _NumpyImpl,
)

DataAggregatorBlockV1 = make_tensor_wrapper_block(_NumpyImpl)
