# Developer Tools

Advanced utilities for custom model development.

## Overview

The `inference_models.developer_tools` module provides utilities for developers creating custom models that integrate with the `inference_models` package.

## Base functions

- **[get_model_package_contents](get-model-package-contents.md)** - Load files from model packages
- **[x_ray_runtime_environment](x-ray-runtime-environment.md)** - Inspect runtime environment
- **[download_files_to_directory](download-files-to-directory.md)** - Download files to a directory
- **[get_selected_onnx_execution_providers](get-selected-onnx-execution-providers.md)** - Get ONNX execution providers
- **[get_model_from_provider](get-model-from-provider.md)** - Get model metadata from provider
- **[register_model_provider](register-model-provider.md)** - Register a custom model provider

## Backend-Specific Utilities

### CUDA Utilities

Low-level CUDA context management for custom models using CUDA/TensorRT.

- **[use_primary_cuda_context](cuda/use-primary-cuda-context.md)** - Use primary CUDA context for operations
- **[use_cuda_context](cuda/use-cuda-context.md)** - Context manager for CUDA operations

### ONNX Utilities

Utilities for working with ONNX Runtime in custom models.

- **[set_onnx_execution_provider_defaults](onnx/set-onnx-execution-provider-defaults.md)** - Configure ONNX execution provider defaults
- **[run_onnx_session_with_batch_size_limit](onnx/run-onnx-session-with-batch-size-limit.md)** - Run ONNX session with batch size constraints
- **[run_onnx_session_via_iobinding](onnx/run-onnx-session-via-iobinding.md)** - Run ONNX session using IO binding for performance

### PyTorch Utilities

Utilities for PyTorch-based custom models.

- **[generate_batch_chunks](torch/generate-batch-chunks.md)** - Split batches into chunks for memory management

### TensorRT Utilities

Utilities for TensorRT-based custom models.

- **[get_trt_engine_inputs_and_outputs](trt/get-trt-engine-inputs-and-outputs.md)** - Inspect TensorRT engine inputs and outputs
- **[infer_from_trt_engine](trt/infer-from-trt-engine.md)** - Run inference using TensorRT engine
- **[load_trt_model](trt/load-trt-model.md)** - Load TensorRT engine from file

## Entities

- **[RuntimeXRayResult](runtime-xray-result.md)** - Runtime environment inspection result
- **[ModelMetadata](model-metadata.md)** - Model metadata structure
- **[ModelDependency](model-dependency.md)** - Model dependency specification
- **[ModelPackageMetadata](model-package-metadata.md)** - Model package metadata
- **[TorchScriptPackageDetails](torchscript-package-details.md)** - TorchScript package details
- **[ONNXPackageDetails](onnx-package-details.md)** - ONNX package details
- **[TRTPackageDetails](trt-package-details.md)** - TensorRT package details
- **[JetsonEnvironmentRequirements](jetson-environment-requirements.md)** - Jetson environment requirements
- **[ServerEnvironmentRequirements](server-environment-requirements.md)** - Server environment requirements
- **[FileDownloadSpecs](file-download-specs.md)** - File download specifications

## Usage

### Basic Usage

```python
from inference_models.developer_tools import (
    get_model_package_contents,
    x_ray_runtime_environment,
    register_model_provider,
)
```

### Backend-Specific Usage

Backend-specific utilities are available as lazy imports:

```python
from inference_models.developer_tools import (
    use_primary_cuda_context,  # CUDA utilities
    set_onnx_execution_provider_defaults,  # ONNX utilities
    generate_batch_chunks,  # PyTorch utilities
    load_trt_model,  # TensorRT utilities
)
```

!!! note "Lazy Loading"
    Backend-specific utilities are lazily loaded only when accessed. This means they won't cause import errors if the required dependencies (e.g., `tensorrt`, `onnxruntime`) are not installed, as long as you don't try to use them.
