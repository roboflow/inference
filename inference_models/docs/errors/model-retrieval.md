# Model Retrieval Errors

**Base Class:** `ModelRetrievalError`

Model retrieval errors occur when the system fails to retrieve model metadata from the weights provider (typically Roboflow API). These errors happen during the model discovery phase and are typically caused by authentication issues, network problems, or inconsistent metadata.

---

## ModelRetrievalError

**Base class for all model retrieval errors.**

This is the parent error class. Specific errors below provide detailed information about what went wrong during model metadata retrieval.

---

## UnauthorizedModelAccessError

**Unauthorized access to model - invalid or missing API key.**

### Overview

This error occurs when you try to access a model without proper authentication or with an invalid API key.

### When It Occurs

**Scenario 1: Missing API key**

- Trying to access a private Roboflow model without API key

- `ROBOFLOW_API_KEY` environment variable not set

- No `api_key` parameter provided to `AutoModel.from_pretrained()`

**Scenario 2: Invalid API key**

- API key is incorrect or malformed

- API key has been revoked or expired

- API key doesn't have access to the requested workspace/project

**Scenario 3: Wrong workspace**

- API key is valid but for a different workspace

- Model ID references a workspace you don't have access to

### What To Check

1. **Verify API key is set:**
   ```bash
   echo $ROBOFLOW_API_KEY
   ```

2. **Check API key validity:**
   ```python
   from inference_models import AutoModel

   # Try with explicit API key
   model = AutoModel.from_pretrained(
       "your-model/1",
       api_key="your_api_key_here"
   )
   ```

3. **Verify model ID format:**
   ```python
   # Correct format for Roboflow models:
   # usually - "project-id/version"
   model_id = "my-project/2"
   ```

### How To Fix

**Set API key as environment variable:**
```bash
export ROBOFLOW_API_KEY="your_api_key_here"
```

**Pass API key directly:**
```python
from inference_models import AutoModel

model = AutoModel.from_pretrained(
    "my-workspace/my-model/1",
    api_key="your_api_key_here"
)
```

**Get your API key:**

- Visit [Roboflow Dashboard](https://app.roboflow.com)

- Go to Settings → Roboflow API

- Copy your API key

- See [Authentication Guide](https://docs.roboflow.com/api-reference/authentication) for details

**Verify workspace access:**

- Make sure the API key belongs to the correct workspace

- Check that you have access to the project

- Verify the model version exists

---

## ModelMetadataConsistencyError

**Inconsistent or invalid model metadata returned by the weights provider.**

### Overview

This error occurs when the Roboflow API returns model metadata that is malformed, incomplete, or contains inconsistent information.

### When It Occurs

**Scenario 1: Unparseable metadata**

- API returns metadata in unexpected format

- JSON structure doesn't match expected schema

- Required fields are missing

**Scenario 2: Invalid batch size configuration**

- TRT package declares dynamic batch size but doesn't specify min/opt/max values

- Static batch size package has invalid or missing batch size value

**Scenario 3: Invalid version specification**

- Package manifest contains invalid version string

- Version format doesn't follow semantic versioning

**Scenario 4: Inconsistent package configuration**

- Package declares conflicting settings

- Required package files are missing from metadata

### What To Check

Since this is Weights Porvider issue, it is not actionable for users and should be reported to Roboflow, unless 
custom weights provider is used - then it should be fixed by the provider.

### How To Fix

**This is typically a Roboflow API issue:**

- [Report the issue](https://github.com/roboflow/inference/issues) with:

  - Full error message

  - Model ID

  - Workspace name

  - When the error started occurring

**If you're a custom weights provider:**

- Ensure your metadata follows the expected schema

- Validate all required fields are present

- Check version strings follow semantic versioning

- Verify batch size configuration is consistent

---

## ModelMetadataHandlerNotImplementedError

**Model metadata handler is not implemented for this model type.**

### Overview

This error occurs when the Roboflow API returns metadata for a model type that is not yet supported by your version of `inference-models`.

### When It Occurs

**Scenario 1: Outdated inference-models package**

- Using an old version of `inference-models`

- New model type added to Roboflow platform

- Your package doesn't have the handler for this model type

**Scenario 2: New/experimental model type**

- Model uses a new architecture not yet in stable release

- Beta/experimental model type

- Custom model type not in standard distribution

### What To Check

1. **Check your inference-models version:**
   ```python
   import inference_models
   print(inference_models.__version__)
   ```

2. **Check for available updates and install if available**

3. **Review error message:**

   - Usually indicates which model type/handler is missing

   - May suggest upgrading the package

### How To Fix

**Upgrade inference-models:**
```bash
# Upgrade to latest version
uv pip install --upgrade inference-models

# Or install specific version
uv pip install inference-models==x.y.z
```

**If already on latest version:**

- This model type may not be supported yet

- [Check GitHub issues](https://github.com/roboflow/inference/issues) for status

- [Open a new issue](https://github.com/roboflow/inference/issues/new) if not reported
