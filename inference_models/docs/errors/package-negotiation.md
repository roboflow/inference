# Model Package Negotiation Errors

**Base Class:** `ModelPackageNegotiationError`

Model package negotiation errors occur when the system cannot select an appropriate model package for the current environment. These errors happen during the package selection phase, before actual model loading, and are typically caused by incompatible hardware, unsupported backends, invalid configurations, or environment introspection failures.

## Common Characteristics

- **When they occur:** During model package selection, before loading
- **Root causes:** Incompatible hardware, unsupported backends/quantization, invalid batch sizes, environment detection failures
- **Impact:** Cannot determine which model package to use; model loading cannot proceed
- **Recovery:** Adjust backend/quantization settings, fix batch size, ensure compatible hardware, or resolve environment issues

## Error Hierarchy

```
ModelPackageNegotiationError
├── UnknownBackendTypeError
├── UnknownQuantizationError
├── InvalidRequestedBatchSizeError
├── RuntimeIntrospectionError
│   └── JetsonTypeResolutionError
├── NoModelPackagesAvailableError
└── AmbiguousModelPackageResolutionError
```

## Error Types

### ModelPackageNegotiationError

**Base class for all model package negotiation errors.**

### UnknownBackendTypeError

**Requested backend type is not supported.**

### UnknownQuantizationError

**Requested quantization type is not supported.**

### InvalidRequestedBatchSizeError

**Requested batch size is invalid.**

### RuntimeIntrospectionError

**Failed to introspect the runtime environment.**

### JetsonTypeResolutionError

**Failed to determine the Jetson device type.**

Inherits from: `RuntimeIntrospectionError`

### NoModelPackagesAvailableError

**No compatible model packages are available for the current environment.**

### AmbiguousModelPackageResolutionError

**Multiple model packages match the selection criteria.**

