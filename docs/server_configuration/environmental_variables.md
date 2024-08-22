# Environmental vaiables

`Inference` behavior can be controlled by set of environmental variables. All environmental variables are listed in [inference/core/env.py](inference/core/env.py)

Below is a list of some environmental values that require more in-depth explanation.

Environmental variable                     | Default                                                                  | Description
------------------------------------------ | ------------------------------------------------------------------------ | -----------
ONNXRUNTIME_EXECUTION_PROVIDERS            | "[CUDAExecutionProvider,OpenVINOExecutionProvider,CPUExecutionProvider]" | List of execution providers in priority order, warning message will be displayed if provider is not supported on user platform
SAM2_MAX_EMBEDDING_CACHE_SIZE                        | 100                                                                     | The number of sam2 embeddings that will be held in memory. The embeddings will be held in gpu memory. Each embedding takes 16777216 bytes.
SAM2_MAX_LOGITS_CACHE_SIZE                        | 1000                                                                     | The number of sam2 logits that will be held in memory. The the logits will be in cpu memory. Each logit takes 262144 bytes.
DISABLE_SAM2_LOGITS_CACHE                        | False                                                                     | If set to True, disables the caching of SAM2 logits. This can be useful for debugging or in scenarios where memory usage needs to be minimized, but may result in slower performance for repeated similar requests.
