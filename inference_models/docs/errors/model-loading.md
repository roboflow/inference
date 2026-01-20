# Model Loading Errors

**Base Class:** `ModelLoadingError`

Model loading errors occur when the system fails to load or initialize a model. These errors typically happen during the model instantiation phase and can be caused by various issues including missing files, corrupted packages, invalid configurations, security violations, or incompatible environments.

---

## ModelLoadingError

**Base class for all model loading errors.**

This is the parent error class. Specific errors below provide detailed information about what went wrong.

---

## ModelPackageAlternativesExhaustedError

**All available model package alternatives failed to load.**

### Overview

This error occurs when the system tried multiple model package candidates but all of them failed to load. This is typically the final error after the system has exhausted all available options for loading your model.

### When It Occurs

**Scenario 1: Auto-selected packages all failed**

  - System automatically selected compatible packages

  - All packages failed due to missing dependencies or other issues

  - Likely indicates missing dependency, broken model packages or a bug in `inference-models` - details to be found in the summary of errors.

**Scenario 2: Manually specified package failed**

   - You explicitly specified which package to load

   - The specified package failed to load

   - Usually due to missing dependencies or broken model package

### What To Check

1. **Check if required dependencies are installed:**
   ```python
   from inference_models import AutoModel
   
   AutoModel.describe_compute_environment()
   ```

2. **Review the error summary** - The error message includes details about why each package failed

3. **Install missing backends:**
   Use the [Installation Guide](../getting-started/installation.md) to install the required backends.

4. **If auto-selection failed** - This is likely a bug. Please [report it urgently](https://github.com/roboflow/inference/issues) with:
    
     - Full error message including the summary of package failures
   
     - Your environment details (`AutoModel.describe_compute_environment()` output)
   
     - Model ID you're trying to load

---

## MissingModelInitParameterError

**Required model initialization parameter is missing.**

### Overview

This error occurs when loading a model (usually directly from a checkpoint file) without providing required parameters that cannot be inferred automatically.

### When It Occurs

**Scenario 1: Loading from checkpoint without `model_type`**

   - Using `AutoModel.from_pretrained()` with a local checkpoint path

   - `model_type` parameter not specified

   - System cannot determine model architecture from file alone


### What To Check

1. **Specify `model_type` when loading from checkpoint:**
   ```python
   from inference_models import AutoModel

   # ❌ Wrong - missing model_type
   model = AutoModel.from_pretrained("/path/to/checkpoint.pt")

   # ✅ Correct - specify model_type
   model = AutoModel.from_pretrained(
       "/path/to/checkpoint.pt",
       model_type="rfdetr-nano",  # or rfdetr-seg-preview, etc.
       task_type="object-detection"
   )
   ```

2. **Check supported model types:**

   - For RFDetr object detection: `rfdetr-nano`, `rfdetr-small`, `rfdetr-medium`, `rfdetr-large`
   
   - For RFDetr instance segmentation: `rfdetr-seg-preview`

3. **If using Roboflow hosted solution and seeing this error** - This is a bug. Contact support.

---

## InvalidModelInitParameterError

**Model initialization parameter has an invalid value.**

### Overview

This error occurs when a model initialization parameter has a value that doesn't meet the model's requirements.

### When It Occurs

**Scenario 1: Invalid task type for checkpoint**

   - Loading model from checkpoint with unsupported `task_type`

   - Example: Trying to use classification task with object detection model

**Scenario 2: Invalid resolution for RF-DETR**
 
   - `resolution` parameter is negative or not divisible by required value (56 for RF-DETR)

   - RF-DETR requires resolution divisible by `num_windows * patch_size`

**Scenario 3: Invalid `model_type`**
 
   - Specified `model_type` not in supported list

   - Typo in model type name

### What To Check

1. **For RFDetr resolution errors:**
   ```python
   # ❌ Wrong - not divisible by 56
   model = RFDetrForObjectDetectionTorch.from_checkpoint_file(
       checkpoint_path="/path/to/model.pt",
       model_type="rfdetr-nano",
       resolution=640  # Not divisible by 56!
   )

   # ✅ Correct - divisible by 56
   model = RFDetrForObjectDetectionTorch.from_checkpoint_file(
       checkpoint_path="/path/to/model.pt",
       model_type="rfdetr-nano",
       resolution=672  # 672 = 56 * 12
   )
   ```

2. **For task type errors:**
   ```python
   # ❌ Wrong - invalid task for this model
   model = AutoModel.from_pretrained(
       "/path/to/checkpoint.pt",
       model_type="rfdetr-nano",
       task_type="classification"  # Not supported!
   )

   # ✅ Correct - use supported task
   model = AutoModel.from_pretrained(
       "/path/to/checkpoint.pt",
       model_type="rfdetr-nano",
       task_type="object-detection"  # Supported
   )
   ```

3. **Check model documentation** for supported parameter values

---

## InsecureModelIdentifierError

**Model identifier contains invalid or insecure characters.**

### Overview

This error occurs when a model package ID contains characters that are not safe for filesystem operations. Package IDs must contain only ASCII letters and numbers to ensure safe local caching.

### When It Occurs

**Scenario: Package ID has special characters**
 
   - Model package ID from weights provider contains non-alphanumeric characters

   - Could include: spaces, slashes, dots, unicode characters, etc.

   - Detected during package ID validation before caching

   - Caused by poisoned metadata provided by Weights Provider

### What To Check

1. **If using Roboflow platform or default Weights Provider** - This is a bug in the model package metadata. Please [report it](https://github.com/roboflow/inference/issues).

2. **If using custom weights provider:**
   - Ensure package IDs only contain: `A-Z`, `a-z`, `0-9`
   - No spaces, special characters, or unicode

   ```python
   # ❌ Wrong package IDs
   "model-v1.0"      # Contains dots
   "model package"   # Contains space
   "model/variant"   # Contains slash

   # ✅ Correct package IDs
   "modelv10"
   "modelpackage"
   "modelvariant"
   ```

3. **Verify your weights provider implementation** returns safe package IDs

---

## DirectLocalStorageAccessError

**Attempted illegal direct access to local storage.**

### Overview

This error occurs when you try to load a model from a local filesystem path, but the system is configured to disallow direct local storage access for security reasons.

### When It Occurs

**Scenario: Local path provided when disabled**
 
   - You provided a local filesystem path as `model_id_or_path` of `AutoModel.from_pretrained(...)`

   - `allow_direct_local_storage_loading=False` (default in some contexts)

   - System blocks the operation for security

### What To Check

1. **Enable local loading if you control the environment:**
   ```python
   from inference_models import AutoModel

   # ❌ Wrong - local loading disabled by default
   model = AutoModel.from_pretrained("/path/to/model")

   # ✅ Correct - explicitly allow local loading
   model = AutoModel.from_pretrained(
       "/path/to/model",
       allow_direct_local_storage_loading=True
   )
   ```

2. **Use model ID instead of path** if loading from Roboflow:
   ```python
   # ❌ Wrong - in most cases, trying to use local path may be wrong
   model = AutoModel.from_pretrained("/workspace/yolov8n-640")

   # ✅ Correct - use model ID
   model = AutoModel.from_pretrained("yolov8n-640")
   ```

3. **Local path collides with model ID** - in some rare cases, there is a file or directory in your current working directory which has name colliding with model ID you attempt to load. In such case, either delete the file/directory or move it out of the way, or use absolute path to the model package you want to load.

4**Security consideration:** This restriction exists to prevent loading untrusted models from arbitrary filesystem locations. Only enable if you trust the source.

---

## ForbiddenLocalCodePackageAccessError

**Attempted access to a forbidden local code package.**

### Overview

This error occurs when trying to load a model package that contains custom Python code, but the system is configured to block execution of arbitrary code for security reasons.

### When It Occurs

**Scenario: Custom code package blocked**

- Model package includes custom Python implementation (`model_module` and `model_class`)

- `allow_loading_code_packages=False` or `allow_local_code_packages=False` (default in some contexts)

- System blocks loading for security

### What To Check

1. **Enable code package loading if you trust the source:**
   ```python
   from inference_models import AutoModel

   # ❌ Wrong - code packages disabled
   model = AutoModel.from_pretrained("./path/to/custom-model-with-code")

   # ✅ Correct - explicitly allow code packages
   model = AutoModel.from_pretrained(
       "./path/to/custom-model-with-code",
       allow_local_code_packages=True,
       allow_loading_code_packages=True
   )
   ```

2. **Verify the model package source:**

   - Only enable for models from trusted sources
   
   - Custom code packages can execute arbitrary Python code
   
   - Review the code before enabling if possible

3. **Security consideration:** This is a critical security feature. Arbitrary code execution can be dangerous. Only enable if you fully trust the model source.

---

## ModelImplementationNotFoundError

**Could not find or load the model implementation.**

### Overview

This error occurs when the system cannot find a registered implementation for the requested model architecture, task type, and backend combination.

### When It Occurs

**Scenario: No implementation registered**

- Requested combination of `model_architecture`, `task_type`, and `backend` has no registered implementation

- Example: Trying to use model which was not yet added to the library in a version that is currently installed


### What To Check

1. Verify the version of `inference-models` you are using and check if the model you are trying to load is supported in that version. If not, consider upgrading to the latest version or downgrading to a version that supports the model.
   ```python
   import inference_models
   
   print(inference_models.__version__)
   ```
   
2. If never version available on [PyPI](https://pypi.org/project/inference-models/) supports the model you are trying to load (to be found in [Changelog](../changelog.md)) - consider upgrade
   ```bash
   uv pip install --upgrade inference-models
   ```

---

## CorruptedModelPackageError

**Model package is corrupted or invalid.**

### Overview

This error occurs when the model package structure or contents violate expected contracts, are incomplete, or are corrupted. This is a broad error covering many package integrity issues.

### When It Occurs

**Scenario 1: Missing model config file when loading package from cache directory**
 
   - Loading from local directory

   - `model_config.json` file with informations to resolve model package to specific implementation not found

   - Could be corrupted cache or invalid package structure

**Scenario 2: Model package delivered with corrupted content**
 
   - Weights provider delivered corrupted model package

   - Usually happens due to provider errors, rather than network issue (but re-download may help in minor cases)

   - May happen when model package content got modified without care about backward compatibility

**Scenario 3: Fault model initialization code**
 
   - Bug in model initialization code


**Scenario 4: Missing model_module or model_class (custom model package loading)**
 
   - Config for custom code package missing required fields

   - `model_module` or `model_class` not specified

**Scenario 5: Model module file not found (custom model package loading)**
 
   - Config points to module file that doesn't exist

   - Path mismatch or missing file

**Scenario 6: Cannot load Python module (custom model package loading)**
 
   - Module file exists but cannot be imported

   - Syntax errors or missing dependencies in module

**Scenario 7: Class not found in module (custom model package loading**
 
   - Module loaded but doesn't contain specified class

   - Class name mismatch

**Scenario 8: Model has dependencies at max nesting depth**
    
   - Model requires other models as dependencies

   - Already at maximum dependency resolution depth

   - Prevents infinite recursion



### What To Check

1. **Clear cache and re-download:**
   ```bash
   # Clear cache directory
   rm -rf $INFERENCE_HOME/models-cache/
   # or if using default location:
   rm -rf /tmp/cache/models-cache/
   ```

   Then try loading again:
   ```python
   from inference_models import AutoModel
   model = AutoModel.from_pretrained("your-model-id")
   ```

2. **For local models, verify package structure:**
   ```
   model_directory/
   ├── model_config.json  # Must exist and be valid JSON
   ├── model.onnx         # Or other model files
   └── model.py           # If custom code package
   ```

3. **Validate model_config.json:**
   ```json
   {
     "model_architecture": "yolov8",
     "task_type": "object-detection",
     "backend_type": "onnx",
     "model_module": "model.py",  // If custom code
     "model_class": "MyModel"      // If custom code
   }
   ```

4. **Check supported backend types:**
    
   - Valid values: `"onnx"`, `"torch"`, `"tensorrt"`, etc.
   
   - Must match exactly (case-sensitive)

5. **If using Roboflow hosted solution and seeing this error:**
    
   - This is likely a bug in the model package
   
   - Contact support with full error details

---

## DependencyModelParametersValidationError

**Dependent model parameters validation failed.**

### Overview

This error occurs when a model has dependencies on other models, and the parameters provided for those dependency models are invalid.

### When It Occurs

**Scenario: Invalid dependency parameters**
 
  - Loading a model that depends on other models

  - Provided `dependency_models_params` contains invalid parameters

  - Validation of dependency parameters failed

### What To Check

1. **Review dependency model parameters:**
   ```python
   from inference_models import AutoModel

   # ❌ Wrong - invalid parameters for dependency
   model = AutoModel.from_pretrained(
       "model-with-dependencies",
       dependency_models_params={
           "dependency_model_name": {
               "invalid_param": "value"  # Not accepted by dependency
           }
       }
   )

   # ✅ Correct - valid parameters
   model = AutoModel.from_pretrained(
       "model-with-dependencies",
       dependency_models_params={
           "dependency_model_name": {
               "device": "cuda",  # Valid parameter
               "resolution": 640
           }
       }
   )
   ```

2. **Omit parameters to use defaults:**
   ```python
   # Let dependencies use default parameters
   model = AutoModel.from_pretrained("model-with-dependencies")
   ```

---

## ModelPipelineInitializationError

**Failed to initialize the model pipeline.**

### Overview

Base error for issues during model pipeline initialization. Model pipelines are pre-configured sequences of models working together.

### When It Occurs

- During initialization of a model pipeline
- Configuration or setup issues prevent pipeline creation
- See specific subclass `ModelPipelineNotFound` for common case

---

## ModelPipelineNotFound

**Requested model pipeline was not found.**

**Inherits from:** `ModelPipelineInitializationError`

### Overview

This error occurs when you request a model pipeline by name, but no pipeline with that name is registered in the system.

### When It Occurs

**Scenario: Invalid pipeline name**
 
  - Requested pipeline name doesn't exist

  - Typo in pipeline name

  - Pipeline not registered

### What To Check

1. **List available pipelines:**
   ```python
   from inference_models import AutoModelPipeline

   # Get all registered pipelines
   pipelines = AutoModelPipeline.list_available_pipelines()
   print("Available pipelines:", pipelines)
   ```

2. **Check pipeline name spelling**

---
