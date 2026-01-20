# Environment Variables Configuration

This guide covers all environment variables available in `inference-models` for configuring model loading, caching, API access, and runtime behavior.

## Quick Start

Set environment variables before importing `inference-models`:

```bash
# Set API key
export ROBOFLOW_API_KEY="your_api_key_here"

# Set model cache directory
export INFERENCE_HOME="/path/to/cache"

# Set device
export DEFAULT_DEVICE="cuda:0"
```

Or use a `.env` file in your project root:

```bash
# .env file
ROBOFLOW_API_KEY=your_api_key_here
MODEL_CACHE_DIR=/path/to/cache
DEFAULT_DEVICE=cuda:0
```

## Core Configuration

### API Authentication

**`ROBOFLOW_API_KEY`** (or `API_KEY`)  
Your Roboflow API key for accessing models.

```bash
export ROBOFLOW_API_KEY="your_api_key_here"
```

Get your API key from: https://docs.roboflow.com/api-reference/authentication

**`ROBOFLOW_ENVIRONMENT`**  
Environment to use: `prod` (default) or `staging`.

```bash
export ROBOFLOW_ENVIRONMENT="prod"
```

**`ROBOFLOW_API_HOST`**  
Override API host URL (auto-set based on environment).

```bash
export ROBOFLOW_API_HOST="https://api.roboflow.com"
```

### Model Cache

**`INFERENCE_HOME`**  
Directory where downloaded models are cached. Default: `/tmp/cache`

```bash
export INFERENCE_HOME="/home/user/.cache/inference-models"
```

### Device Selection

**`DEFAULT_DEVICE`**  
Default device for model inference: `cpu`, `cuda`, `cuda:0`, etc.

```bash
export DEFAULT_DEVICE="cuda:0"  # Use first GPU
export DEFAULT_DEVICE="cpu"     # Use CPU
```

## API Configuration

### Request Settings

**`API_CALLS_TIMEOUT`**  
Timeout for API calls in seconds. Default: `5`

```bash
export API_CALLS_TIMEOUT="10"
```

**`API_CALLS_MAX_TRIES`**  
Maximum retry attempts for API calls. Default: `3`

```bash
export API_CALLS_MAX_TRIES="5"
```

**`IDEMPOTENT_API_REQUEST_CODES_TO_RETRY`**  
HTTP status codes to retry (comma-separated). Default: `408,429,502,503,504`

```bash
export IDEMPOTENT_API_REQUEST_CODES_TO_RETRY="408,429,500,502,503,504"
```

## Backend Configuration

### ONNX Runtime

**`ONNXRUNTIME_EXECUTION_PROVIDERS`**
Override ONNX execution providers, comma separated, no spaces. 
Default: `CUDAExecutionProvider,OpenVINOExecutionProvider,CoreMLExecutionProvider,CPUExecutionProvider`

```bash
export ONNX_EXECUTION_PROVIDERS="CPUExecutionProvider"
```

## Logging

**`LOG_LEVEL`**  
Set the log level for the library. Default: `WARNING`

```bash
export LOG_LEVEL="DEBUG"
```

**`VERBOSE_LOG_LEVEL`**  
Set the log level for verbose logging. Default: `INFO`

```bash
export VERBOSE_LOG_LEVEL="DEBUG"
```

**`DISABLE_VERBOSE_LOGGER`**  
Disable verbose logging. Default: `false`

```bash
export DISABLE_VERBOSE_LOGGER="true"
```

**`DISABLE_INTERACTIVE_PROGRESS_BARS`**  
Disable interactive progress bars. Default: `false`

```bash
export DISABLE_INTERACTIVE_PROGRESS_BARS="true"
```


## Advanced Configuration

### Input Validation

**`ALLOW_URL_INPUT`**
Allow URLs as image input. Used by models like OWL-V2, when access to larger 
datasets provided as references is needed. Default: `true`

```bash
export ALLOW_URL_INPUT="true"
```

**`ALLOW_NON_HTTPS_URL_INPUT`**
Allow non-HTTPS URLs. Used by models like OWL-V2, when access to larger 
datasets provided as references is needed. Default: `false`

```bash
export ALLOW_NON_HTTPS_URL_INPUT="true"  # Use with caution
```

**`ALLOW_URL_INPUT_WITHOUT_FQDN`**
Allow URLs without FQDN. Used by models like OWL-V2, when access to larger 
datasets provided as references is needed. Default: `false`

```bash
export ALLOW_URL_INPUT_WITHOUT_FQDN="true"  # Use with caution
```

**`WHITELISTED_DESTINATIONS_FOR_URL_INPUT`**  
Comma-separated list of allowed destinations for URL input. Used by models like OWL-V2, when access to larger 
datasets provided as references is needed. Default: `None`

```bash
export WHITELISTED_DESTINATIONS_FOR_URL_INPUT="google.com,github.com"
```

**`BLACKLISTED_DESTINATIONS_FOR_URL_INPUT`**  
Comma-separated list of allowed destinations for URL input. Used by models like OWL-V2, when access to larger 
datasets provided as references is needed. Default: `None`

```bash
export BLACKLISTED_DESTINATIONS_FOR_URL_INPUT="google.com,github.com"
```

**`ALLOW_LOCAL_STORAGE_ACCESS_FOR_REFERENCE_DATA`**  
Allow local storage access for reference data. Used by models like OWL-V2, when access to larger 
datasets provided as references is needed. Default: `true`

```bash
export ALLOW_LOCAL_STORAGE_ACCESS_FOR_REFERENCE_DATA="true"
```
