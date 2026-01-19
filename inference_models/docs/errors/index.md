# Error Reference

This section provides detailed information about errors you may encounter when using `inference-models`, including root causes and solutions.

All errors in `inference-models` inherit from `BaseInferenceError` and are organized by their base error class.

## Error Categories

Errors in `inference-models` are organized into the following categories based on their inheritance hierarchy:

### [Model Loading Errors](model-loading.md)

**Base Class:** `ModelLoadingError`

Errors that occur when loading or initializing models (11 error types):

- **ModelLoadingError** - Base class for model loading failures
- **ModelPackageAlternativesExhaustedError** - All model package alternatives failed to load
- **MissingModelInitParameterError** - Required model initialization parameter is missing
- **InvalidModelInitParameterError** - Model initialization parameter has invalid value
- **InsecureModelIdentifierError** - Model identifier contains invalid/insecure characters
- **DirectLocalStorageAccessError** - Attempted illegal direct access to local storage
- **ForbiddenLocalCodePackageAccessError** - Attempted access to forbidden local code package
- **ModelImplementationLoaderError** - Could not find or load model implementation
- **CorruptedModelPackageError** - Model package is corrupted or invalid
- **DependencyModelParametersValidationError** - Dependent model parameters validation failed
- **ModelPipelineInitializationError** - Failed to initialize model pipeline
  - **ModelPipelineNotFound** - Requested model pipeline not found

### [Model Package Negotiation Errors](package-negotiation.md)

**Base Class:** `ModelPackageNegotiationError`

Errors related to selecting the appropriate model package for your environment (7 error types):

- **ModelPackageNegotiationError** - Base class for package negotiation failures
- **UnknownBackendTypeError** - Requested backend type is not supported
- **UnknownQuantizationError** - Requested quantization type is not supported
- **InvalidRequestedBatchSizeError** - Requested batch size is invalid
- **RuntimeIntrospectionError** - Failed to introspect runtime environment
  - **JetsonTypeResolutionError** - Failed to determine Jetson device type
- **NoModelPackagesAvailableError** - No compatible model packages available
- **AmbiguousModelPackageResolutionError** - Multiple packages match the criteria

### [Model Retrieval Errors](model-retrieval.md)

**Base Class:** `ModelRetrievalError`

Errors that occur when retrieving model metadata from the weights provider (3 error types):

- **ModelRetrievalError** - Base class for model retrieval failures
- **UnauthorizedModelAccessError** - Unauthorized access to model (invalid/missing API key)
- **ModelMetadataConsistencyError** - Inconsistent model metadata from provider
- **ModelMetadataHandlerNotImplementedError** - Model metadata handler not implemented

### [File & Download Errors](file-download.md)

**Base Classes:** `RetryError`, `UntrustedFileError`, `FileHashSumMissmatch`

Errors related to file downloads and integrity verification (3 error types):

- **RetryError** - Transient network or server errors during download
- **FileHashSumMissmatch** - Downloaded file hash doesn't match expected value
- **UntrustedFileError** - File lacks required hash sum for verification

### [Runtime & Environment Errors](runtime-environment.md)

**Base Classes:** `EnvironmentConfigurationError`, `InvalidEnvVariable`, `MissingDependencyError`

Errors related to runtime environment and dependencies (3 error types):

- **EnvironmentConfigurationError** - Invalid environment configuration
- **InvalidEnvVariable** - Environment variable has invalid value
- **MissingDependencyError** - Required dependency is not installed

### [Model Input & Validation Errors](input-validation.md)

**Base Classes:** `ModelInputError`, `InvalidParameterError`, `AssumptionError`

Errors caused by invalid input or parameters (3 error types):

- **ModelInputError** - Invalid input provided to model
- **InvalidParameterError** - Invalid parameter value
- **AssumptionError** - Internal assumption violated

### [Model Runtime Errors](runtime.md)

**Base Class:** `ModelRuntimeError`

Errors that occur during model execution:

- **ModelRuntimeError** - Base class for model runtime failures

## Error Hierarchy

```
BaseInferenceError
├── ModelLoadingError
├── ModelPackageNegotiationError
├── ModelRetrievalError
├── RetryError
├── UntrustedFileError
├── FileHashSumMissmatch
├── EnvironmentConfigurationError
├── InvalidEnvVariable
├── MissingDependencyError
├── ModelInputError
├── InvalidParameterError
├── AssumptionError
└── ModelRuntimeError
```

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
