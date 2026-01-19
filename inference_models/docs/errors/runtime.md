# Model Runtime Errors

**Base Class:** `ModelRuntimeError`

Model runtime errors occur during model execution (inference). These errors happen after the model has been successfully loaded and are typically caused by issues during the actual inference process, such as incompatible input shapes, out-of-memory conditions, or backend-specific failures.

## Common Characteristics

- **When they occur:** During model inference/execution
- **Root causes:** Incompatible input shapes, memory issues, backend failures, numerical errors
- **Impact:** Inference fails; cannot produce predictions
- **Recovery:** Adjust input, reduce batch size, check memory, verify backend compatibility

## Error Hierarchy

```
BaseInferenceError
└── ModelRuntimeError
```

## Error Types

### ModelRuntimeError

**Base class for errors that occur during model execution.**

