# Modal Custom Python Blocks Implementation Plan

## Overview
Implementing Custom Python Blocks for Roboflow Workflows using Modal sandboxes for secure execution of untrusted user code on multi-tenant Serverless v2 infrastructure.

## Key Requirements
- One Modal App per workspace (named `inference-workspace-{workspace_id}`)
- One Modal Function per unique code block (identified by MD5 hash)
- Use Modal Restricted Functions with `restrict_modal_access=True`
- Serialize/deserialize inputs and outputs (no pickle)
- Use us-central1 (GCP Iowa) region
- Match inference version in Modal Image

## Progress Tracker

### ✅ Completed
- [x] Created new branch `feat/custom-python-blocks-modal-sandbox` from main
- [x] Created MODAL_PLAN.md tracking document
- [x] Created Modal requirements file (requirements.modal.txt)
- [x] Added Modal environment variables to env.py
- [x] Created Modal Image builder script (modal/build_modal_image.py)
- [x] Implemented modal_executor.py with ModalExecutor class
- [x] Created serializers.py for data serialization/deserialization
- [x] Updated block_scaffolding.py to support Modal execution
- [x] Updated Serverless v2 Dockerfile to include Modal SDK
- [x] Created test script (modal/test_modal_blocks.py)

### 🚧 In Progress
- [ ] Fix Modal Function creation approach (needs different strategy)

### 📝 TODO
- [ ] Push Modal Image to workspace
- [ ] Test end-to-end workflow
- [ ] Performance testing
- [ ] Security validation

## Implementation Details

### 1. Modal Image Creation
- Base: `modal.Image.debian_slim(python_version="3.11")`
- Install inference package via uv with version matching
- Pre-import common modules for faster cold starts
- Share across all workspace Apps

### 2. App Architecture
- Workspace-based App naming: `inference-workspace-{workspace_id}`
- Function naming: `block-{md5_hash}`
- Lazy Function creation on first use
- Restricted mode with max_inputs=1, timeout=20

### 3. Serialization
- Use existing inference serializers/deserializers
- Handle WorkflowImageData, Batch, numpy arrays, CV2 images
- JSON-safe transport (no pickle)

### 4. Security
- restrict_modal_access=True
- max_inputs=1 (fresh containers)
- 20-second timeout
- Code validation in Modal Function

### 5. Environment Variables
- WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE=modal
- MODAL_TOKEN_ID
- MODAL_TOKEN_SECRET
- MODAL_WORKSPACE_NAME

## File Structure
```
inference/
├── core/workflows/execution_engine/v1/dynamic_blocks/
│   ├── block_scaffolding.py (modify)
│   └── modal_executor.py (new)
├── core/env.py (modify)
├── requirements/requirements.modal.txt (new)
├── docker/dockerfiles/Dockerfile.onnx.gpu (modify)
└── modal/
    ├── build_modal_image.py (new)
    └── test_modal_blocks.py (new)
```

## Notes
- Workspace ID needs to be passed through from request context
- No network blocking (users may need external services)
- No pre-checking for Function existence (try-catch pattern)
- Use existing inference serializers
- No hardcoded secrets in code
