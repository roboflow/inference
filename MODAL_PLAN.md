# Modal Custom Python Blocks Implementation Plan

## Current Status

### ✅ Implementation Complete
The Modal Custom Python Blocks implementation is now complete with:
- Parameterized Modal Functions for workspace isolation  
- Integration with existing inference serializers
- Proper workspace_id threading through the system
- Deployment and testing scripts ready

### 🚀 Ready for Testing
To deploy and test:

1. **Set Modal Credentials**:
```bash
export MODAL_TOKEN_ID="your_token_id"
export MODAL_TOKEN_SECRET="your_token_secret"
```

2. **Deploy Modal App**:
```bash
python modal/deploy_modal_app.py
```

3. **Run Tests**:
```bash
export WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE="modal"
python modal/test_modal_blocks.py
```

4. **Use in Workflows**:
Set environment variable `WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE=modal` in deployment

### 📋 Remaining Tasks
- [ ] Deploy to production Modal environment
- [ ] Run end-to-end integration tests
- [ ] Performance benchmarking
- [ ] Security audit
- [ ] User documentation

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
- [x] ~~Created serializers.py for data serialization/deserialization~~ Removed - using existing serializers
- [x] Updated block_scaffolding.py to support Modal execution
- [x] Updated Serverless v2 Dockerfiles (both GPU and CPU) to include Modal SDK
- [x] Created test script (modal/test_modal_blocks.py)
- [x] Committed initial implementation
- [x] Refactored to use Modal Parameterized Functions for workspace-based isolation
- [x] Updated to use uv_pip_install for better optimization
- [x] Integrated with existing inference serializers (no new serializers.py)
- [x] Created deployment script (modal/deploy_modal_app.py)

### 🚧 Known Issues & Next Steps

1. **Modal App Deployment Strategy**: 
   - ✅ RESOLVED: Using Parameterized Functions with workspace_id parameter
   - Single Modal App "inference-custom-blocks" with parameterized executors
   - Each workspace gets its own executor instance via parameters
   
2. **Workspace ID Threading**: 
   - Already implemented in block_scaffolding.py
   - Falls back to "anonymous" for non-logged-in users
   
3. **Image Management**: 
   - Modal Image can be built on-demand using uv_pip_install
   - Pre-built images can be deployed for production

4. **Testing**: 
   - Requires Modal credentials to test
   - Need to set MODAL_TOKEN_ID and MODAL_TOKEN_SECRET environment variables

### 📝 TODO
- [ ] Deploy Modal App with parameterized executors
- [ ] Test end-to-end workflow with Modal credentials
- [ ] Performance testing
- [ ] Security validation
- [ ] Documentation for deployment and configuration
- [ ] Integration tests with various input types
- [ ] Load testing with multiple workspaces

## Implementation Details

### 1. Modal Image Creation
- Base: `modal.Image.debian_slim(python_version="3.11")`
- Install inference package via uv with version matching
- Pre-import common modules for faster cold starts
- Share across all workspace Apps

### 2. App Architecture
- Single Modal App: `inference-custom-blocks`
- Parameterized executors with workspace_id and code_hash
- Function caching per workspace+code combination
- Restricted mode with max_inputs=1, timeout=20
- Falls back to "anonymous" workspace for non-authenticated users

### 3. Serialization
- Use existing inference serializers from core_steps.common.serializers
- serialize_wildcard_kind handles WorkflowImageData, Batch, numpy arrays, CV2 images, sv.Detections
- No pickle for security (inputs passed as JSON-safe dicts)
- Modal transport uses standard JSON encoding

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
│   ├── block_scaffolding.py (modified - workspace_id support)
│   └── modal_executor.py (new - parameterized functions)
├── core/env.py (modified - Modal env vars)
├── requirements/requirements.modal.txt (new)
├── docker/dockerfiles/
│   ├── Dockerfile.onnx.gpu (modified - added Modal)
│   └── Dockerfile.onnx.cpu (modified - added Modal)
└── modal/
    ├── build_modal_image.py (new - uses uv_pip_install)
    └── test_modal_blocks.py (new)
```

## Notes for Next Developer

The implementation has been refactored to use Modal's Parameterized Functions feature, which allows us to create workspace-isolated executors without needing to dynamically deploy new Modal Apps. Key changes:

1. **Parameterized Executors**: Using `modal.parameter()` for workspace_id and code_hash, allowing each workspace to have its own isolated execution environment.

2. **Existing Serializers**: Removed the duplicate serializers.py and now using the existing `serialize_wildcard_kind` from `inference.core.workflows.core_steps.common.serializers`.

3. **Optimized Image Building**: Using `uv_pip_install` for faster and more reliable package installation in Modal Images.

4. **Fallback Support**: Non-authenticated users can use "anonymous" workspace for testing without a Roboflow account.

Next steps:
- Deploy the Modal App using `modal deploy`
- Set Modal credentials in environment
- Run integration tests with various input types
