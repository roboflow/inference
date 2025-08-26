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
- [x] Committed initial implementation

### 🚧 Known Issues & Next Steps

1. **Modal Function Creation Strategy**: The current approach needs refinement for dynamic function creation in Modal.
   - Consider using Modal Sandboxes API directly instead of Functions
   - Or pre-create a generic executor function that loads and executes code dynamically
   - Modal functions typically need to be defined at module/deployment time

2. **Workspace ID Threading**: Ensure workspace_id is properly passed through the workflow execution context.

3. **Image Management**: The shared Modal Image needs to be pre-built and deployed before use.

4. **Testing**: Requires Modal credentials and deployed image to run tests.

### 📝 TODO
- [ ] Refactor to use Modal Sandboxes API or alternative approach
- [ ] Push Modal Image to workspace  
- [ ] Test end-to-end workflow
- [ ] Add workspace_id propagation through request context
- [ ] Performance testing
- [ ] Security validation
- [ ] Documentation

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
- Code validation in Modal sandbox

### 5. Environment Variables
- WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE=modal
- MODAL_TOKEN_ID
- MODAL_TOKEN_SECRET  
- MODAL_WORKSPACE_NAME

## File Structure
```
inference/
├── core/workflows/execution_engine/v1/dynamic_blocks/
│   ├── block_scaffolding.py (modified)
│   ├── modal_executor.py (new)
│   └── serializers.py (new)
├── core/env.py (modified)
├── requirements/requirements.modal.txt (new)
├── docker/dockerfiles/Dockerfile.onnx.gpu (modified)
└── modal/
    ├── build_modal_image.py (new)
    └── test_modal_blocks.py (new)
```

## Notes for Next Developer

The core implementation is complete but needs adjustment in how Modal Functions are created dynamically. The current approach tries to create functions on-the-fly, but Modal's architecture expects functions to be defined at deployment time.

Consider these alternatives:
1. **Use Modal Sandboxes directly** - More flexible for dynamic code execution
2. **Pre-deploy generic executor** - Deploy a single function that accepts code as input
3. **Use Modal's experimental features** - Check if they have new APIs for dynamic execution

The serialization layer is complete and tested. The integration with block_scaffolding.py properly routes to Modal when configured. The main blocker is the Modal Function creation strategy.

Workspace ID needs to be threaded through from the HTTP request context to the block execution layer.
