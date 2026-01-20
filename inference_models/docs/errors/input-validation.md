# Model Input & Validation Errors

**Base Classes:** `ModelInputError`, `InvalidParameterError`, `AssumptionError`

Model input and validation errors occur when invalid input or parameters are provided to the model or when internal assumptions are violated. These errors happen during input validation or runtime and are typically caused by incorrect data types, invalid values, or violated preconditions.

---

## ModelInputError

**Invalid input provided to the model.**

### Overview

This error occurs when the input data provided to a model is invalid, missing, or incompatible with the model's requirements. This is the most common user-facing error when using models incorrectly.

### When It Occurs

**Scenario 1: Missing required input**

- Neither `images` nor `embeddings` provided to SAM/SAM2

- Required prompts not provided (points, boxes, masks)

- Missing reference dataset for OWLv2

**Scenario 2: Batch size mismatch**

- Point coordinates batch size doesn't match image batch size

- Point labels batch size doesn't match embeddings batch size

- Boxes or masks batch size incompatible with input

**Scenario 3: Invalid prompt configuration**

- Missing mask input when required

- Cache lookup failed for required mask

- Incompatible prompt combinations

**Scenario 4: Invalid reference dataset**

- Empty reference dataset provided

- Duplicate class names in dataset

- Invalid image format in dataset

- Missing required fields

### What To Check

1. **Review the error message:**

     * Error message specifies which parameter is invalid

     * Shows expected vs. actual batch sizes

     * Indicates which input is missing

2. **Check your input data:**


3. **Verify model requirements:**

     * Check model documentation for required inputs

     * Understand batch size requirements

     * Review prompt format specifications

### How To Fix

Way of fixing problem depends on the model and the specific error message. In general, you should review the error message and the model documentation to understand what input is expected and what is provided. Then, you should adjust the input to match the expected format. If you have
further questions, please contact us through [Github issues](https://github.com/roboflow/inference/issues).

---

## InvalidParameterError

**Invalid parameter value provided.**

### Overview

This error occurs when a function or method is called with an invalid parameter value. This typically indicates incorrect API usage or a bug in the library.


### When It Occurs

**Scenario 1: Invalid internal parameter**

- Internal function called with wrong parameter value

- Bug in library code

- Incorrect parameter passed between internal functions

**Scenario 2: Invalid device specification**

- Device string cannot be parsed as torch device

- Invalid device format

- Unsupported device type

**Scenario 3: Invalid download configuration**

- Internal download function called with invalid `name_after` parameter

- Bug in file download logic

### What To Check

1. **Review the error message:**

     * Error message indicates which parameter is invalid

     * Shows the invalid value that was provided

     * Often indicates this is a bug

### How To Fix

If the error message indicates this is a bug in `inference-models`:

1. **Gather information:**
     * Full error message and stack trace

     * Code that triggered the error

     * Model ID and version

     * Python version and environment details

2. **Report the issue:**
     * Visit [GitHub Issues](https://github.com/roboflow/inference/issues)

     * Create a new issue with all gathered information

     * Include minimal reproducible example

3. **Check for updates**

---

## AssumptionError

**Internal assumption violated (indicates a bug or unexpected state).**

### Overview

This error indicates that an internal assumption in the library code has been violated. This almost always indicates a bug in the `inference-models` library and should be reported.

### When It Occurs

**Scenario 1: Internal state inconsistency**

- Model package filtering returned malformed result

- Internal data structures in unexpected state

- Logic error in library code

**Scenario 2: Prompt processing assumptions violated**

- SAM2 prompt elements have mismatched dimensions after preprocessing

- Expected tensor shapes don't match

- Internal broadcasting logic failed

**Scenario 3: Unexpected code path**

- Code reached a state that should be impossible

- Defensive programming check failed

- Invariant violation detected

### What To Check

1. **This is always a bug:**

     * `AssumptionError` always indicates a library bug

     * Not caused by user input (though user input may trigger it)

     * Should be reported to developers

2. **Review the error message:**

     * Describes which assumption was violated

     * Often includes link to GitHub issues

     * May suggest this is a bug

3. **Check if you can reproduce:**
   ```python
   # Try to create minimal reproducible example
   # This helps developers fix the bug faster
   ```

### How To Fix

**This error cannot be fixed by users - it requires a library fix.**

**Immediate steps:**

1. **Gather diagnostic information:**
   ```python
   import sys
   import torch
   import inference_models

   print(f"Python version: {sys.version}")
   print(f"PyTorch version: {torch.__version__}")
   print(f"inference-models version: {inference_models.__version__}")
   print(f"CUDA available: {torch.cuda.is_available()}")
   if torch.cuda.is_available():
       print(f"CUDA version: {torch.version.cuda}")
   ```

2. **Create minimal reproducible example:**
   ```python
   # Simplify your code to the minimum that reproduces the error
   from inference_models import AutoModel

   model = AutoModel.from_pretrained("model-id")
   # ... minimal code that triggers the error
   ```

3. **Report the issue:**

     * Visit [GitHub Issues](https://github.com/roboflow/inference/issues)

     * Create a new issue with:
       - Full error message and stack trace
       - Minimal reproducible example
       - Environment information (Python, PyTorch, CUDA versions)
       - Model ID and parameters used

     * Use a descriptive title like: "AssumptionError in SAM2 prompt processing"

**Check for existing issues:**

Before reporting, search existing issues:

- Visit [GitHub Issues](https://github.com/roboflow/inference/issues)
- Search for keywords from your error message
- Check if someone else reported the same issue
- Add your information to existing issue if found

**Stay updated:**

```bash
# Check for updates regularly
uv pip install --upgrade inference-models

# Or watch the GitHub repository for releases
# https://github.com/roboflow/inference
```
