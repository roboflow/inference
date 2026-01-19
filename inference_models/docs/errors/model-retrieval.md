# Model Retrieval Errors

**Base Class:** `ModelRetrievalError`

Model retrieval errors occur when the system fails to retrieve model metadata from the weights provider (typically Roboflow API). These errors happen during the model discovery phase and are typically caused by authentication issues, network problems, or inconsistent metadata.

## Common Characteristics

- **When they occur:** During model metadata retrieval from weights provider
- **Root causes:** Invalid/missing API key, network issues, inconsistent metadata, unimplemented handlers
- **Impact:** Cannot retrieve model information; model loading cannot proceed
- **Recovery:** Check API key, verify network connectivity, contact support for metadata issues

## Error Hierarchy

```
ModelRetrievalError
├── UnauthorizedModelAccessError
├── ModelMetadataConsistencyError
└── ModelMetadataHandlerNotImplementedError
```

## Error Types

### ModelRetrievalError

**Base class for all model retrieval errors.**

### UnauthorizedModelAccessError

**Unauthorized access to model (invalid or missing API key).**

### ModelMetadataConsistencyError

**Inconsistent model metadata returned by the weights provider.**

### ModelMetadataHandlerNotImplementedError

**Model metadata handler is not implemented for this model type.**

