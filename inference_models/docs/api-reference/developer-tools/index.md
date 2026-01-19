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
