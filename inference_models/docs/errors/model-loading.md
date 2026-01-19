# Model Loading Errors

**Base Class:** `ModelLoadingError`

Model loading errors occur when the system fails to load or initialize a model. These errors typically happen during the model instantiation phase and can be caused by various issues including missing files, corrupted packages, invalid configurations, security violations, or incompatible environments.

## Common Characteristics

- **When they occur:** During model initialization via `load_model()` or `AutoModel.load()`
- **Root causes:** Missing files, corrupted packages, invalid parameters, security violations, missing dependencies
- **Impact:** Model cannot be loaded; inference cannot proceed
- **Recovery:** Usually requires fixing configuration, re-downloading model, adjusting parameters, or installing dependencies

## Error Hierarchy

```
ModelLoadingError
├── ModelPackageAlternativesExhaustedError
├── MissingModelInitParameterError
├── InvalidModelInitParameterError
├── InsecureModelIdentifierError
├── DirectLocalStorageAccessError
├── ForbiddenLocalCodePackageAccessError
├── ModelImplementationLoaderError
├── CorruptedModelPackageError
├── DependencyModelParametersValidationError
└── ModelPipelineInitializationError
    └── ModelPipelineNotFound
```

## Error Types

### ModelLoadingError

**Base class for all model loading errors.**

### ModelPackageAlternativesExhaustedError

**All available model package alternatives failed to load.**

### MissingModelInitParameterError

**Required model initialization parameter is missing.**

### InvalidModelInitParameterError

**Model initialization parameter has an invalid value.**

### InsecureModelIdentifierError

**Model identifier contains invalid or insecure characters.**

### DirectLocalStorageAccessError

**Attempted illegal direct access to local storage.**

### ForbiddenLocalCodePackageAccessError

**Attempted access to a forbidden local code package.**

### ModelImplementationLoaderError

**Could not find or load the model implementation.**

### CorruptedModelPackageError

**Model package is corrupted or invalid.**

### DependencyModelParametersValidationError

**Dependent model parameters validation failed.**

### ModelPipelineInitializationError

**Failed to initialize the model pipeline.**

### ModelPipelineNotFound

**Requested model pipeline was not found.**

Inherits from: `ModelPipelineInitializationError`

