# Runtime & Environment Errors

**Base Classes:** `EnvironmentConfigurationError`, `InvalidEnvVariable`, `MissingDependencyError`

Runtime and environment errors occur when the system detects issues with the runtime environment, configuration, or dependencies. These errors can happen at various stages and are typically caused by missing dependencies, invalid environment variables, or incorrect environment setup.

## Common Characteristics

- **When they occur:** During initialization, model loading, or runtime
- **Root causes:** Missing dependencies, invalid environment variables, incorrect configuration
- **Impact:** Cannot proceed with operation; may affect multiple models
- **Recovery:** Install dependencies, fix environment variables, correct configuration

## Error Hierarchy

```
BaseInferenceError
├── EnvironmentConfigurationError
├── InvalidEnvVariable
└── MissingDependencyError
```

## Error Types

### EnvironmentConfigurationError

**Invalid environment configuration detected.**

### InvalidEnvVariable

**Environment variable has an invalid value.**

### MissingDependencyError

**Required dependency is not installed.**

