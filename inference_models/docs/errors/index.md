# Error Reference

This section provides detailed information about errors you may encounter when using `inference-models`, including root causes and solutions.

All errors in `inference-models` inherit from `BaseInferenceError` and are organized by their base error class.

## Error Categories

### [Model Loading Errors](model-loading.md)

Errors that occur when loading or initializing models. These errors typically happen during the model instantiation phase and can be caused by various issues including missing files, corrupted packages, invalid configurations, security violations, or incompatible environments.

[View all Model Loading Errors →](model-loading.md)

### [Model Package Negotiation Errors](package-negotiation.md)

Errors that occur when the system cannot select an appropriate model package for the current environment. These errors happen during the package selection phase, before actual model loading, and are typically caused by incompatible hardware, unsupported backends, invalid configurations, or environment introspection failures.

[View all Model Package Negotiation Errors →](package-negotiation.md)

### [Model Retrieval Errors](model-retrieval.md)

Errors that occur when the system fails to retrieve model metadata from the weights provider (typically Roboflow API). These errors happen during the model discovery phase and are typically caused by authentication issues, network problems, or inconsistent metadata.

[View all Model Retrieval Errors →](model-retrieval.md)

### [File & Download Errors](file-download.md)

Errors that occur when the system fails to download model files or verify their integrity. These errors happen during the file download phase and are typically caused by network issues, corrupted downloads, or security validation failures.

[View all File & Download Errors →](file-download.md)

### [Runtime & Environment Errors](runtime-environment.md)

Errors that occur when the system detects issues with the runtime environment, configuration, or dependencies. These errors can happen at various stages and are typically caused by missing dependencies, invalid environment variables, or incorrect environment setup.

[View all Runtime & Environment Errors →](runtime-environment.md)

### [Model Input & Validation Errors](input-validation.md)

Errors that occur when invalid input or parameters are provided to the model or when internal assumptions are violated. These errors happen during input validation or runtime and are typically caused by incorrect data types, invalid values, or violated preconditions.

[View all Model Input & Validation Errors →](input-validation.md)

### [Model Runtime Errors](models-runtime.md)

Errors that occur during model execution (inference). These errors happen after the model has been successfully loaded and are typically caused by issues during the actual inference process, such as incompatible input shapes, out-of-memory conditions, or backend-specific failures.

[View all Model Runtime Errors →](models-runtime.md)

## Getting Help

If you encounter an error not covered in this documentation:

1. Check the error message for specific guidance and help URL
2. Review the category-specific documentation page for your error
3. Search existing [GitHub Issues](https://github.com/roboflow/inference/issues)
4. Create a new issue with:
   - Full error message and stack trace
   - Your environment details (OS, Python version, GPU info)
   - Minimal code to reproduce the issue
   - Model ID and configuration (if applicable)
