# Model Input & Validation Errors

**Base Classes:** `ModelInputError`, `InvalidParameterError`, `AssumptionError`

Model input and validation errors occur when invalid input or parameters are provided to the model or when internal assumptions are violated. These errors happen during input validation or runtime and are typically caused by incorrect data types, invalid values, or violated preconditions.

## Common Characteristics

- **When they occur:** During input validation or runtime
- **Root causes:** Invalid input data, incorrect parameter values, violated assumptions
- **Impact:** Cannot process input; inference fails
- **Recovery:** Fix input data, correct parameter values, review API documentation

## Error Hierarchy

```
BaseInferenceError
├── ModelInputError
├── InvalidParameterError
└── AssumptionError
```

## Error Types

### ModelInputError

**Invalid input provided to the model.**

### InvalidParameterError

**Invalid parameter value provided.**

### AssumptionError

**Internal assumption violated (indicates a bug or unexpected state).**

