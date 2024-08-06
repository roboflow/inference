# Environmental vaiables

`Inference` behavior can be controlled by set of environmental variables. All environmental variables are listed in [inference/core/env.py](inference/core/env.py)

Below is a list of some environmental values that require more in-depth explanation.

Environmental variable                     | Default                                                                  | Description
------------------------------------------ | ------------------------------------------------------------------------ | -----------
ONNXRUNTIME_EXECUTION_PROVIDERS            | "[CUDAExecutionProvider,OpenVINOExecutionProvider,CPUExecutionProvider]" | List of execution providers in priority order, warning message will be displayed if provider is not supported on user platform
