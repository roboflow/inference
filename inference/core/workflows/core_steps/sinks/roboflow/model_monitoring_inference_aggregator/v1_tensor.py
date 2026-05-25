from inference.core.workflows.core_steps.common.wrap_consumer import (
    make_tensor_wrapper_block,
)
from inference.core.workflows.core_steps.sinks.roboflow.model_monitoring_inference_aggregator.v1 import (
    ModelMonitoringInferenceAggregatorBlockV1 as _NumpyImpl,
)

ModelMonitoringInferenceAggregatorBlockV1 = make_tensor_wrapper_block(_NumpyImpl)
