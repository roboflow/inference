# 💾 Cache Management

Understanding how `inference-models` caches data to improve performance and reduce redundant operations.

## Overview

The library uses two types of caching:

- **Auto-Resolution Cache** - Stores backend selection decisions to avoid repeated API calls and package negotiation
- **Model Package Cache** - Stores downloaded model files (weights, configs, class names) to avoid re-downloading

Both caches are stored under `$INFERENCE_HOME` (defaults to `/tmp/cache/`).

## 🔄 Auto-Resolution Cache

### What Gets Cached

When you load a model with `AutoModel.from_pretrained()`, the library performs backend negotiation to select the optimal model package. This decision is cached to avoid repeating the process on subsequent loads.

All parameters that affect the negotiation are hashed together:

- Weights provider (Roboflow, local, etc.)
- Model ID
- API key (hashed for security)
- Requested backend preferences
- Requested quantization
- Requested batch size
- Device configuration
- ONNX execution providers
- Other negotiation parameters

If **any** of these parameters change, the cache is bypassed and a fresh negotiation occurs.

### Configuration

**Default location:** `$INFERENCE_HOME/auto-resolution-cache/` (defaults to `/tmp/cache/auto-resolution-cache/`)

**Cache expiration:** 24 hours (1440 minutes) by default

**Override cache location:**
```bash
export INFERENCE_HOME=/path/to/custom/cache
```

**Change expiration time:**
```bash
export AUTO_LOADER_CACHE_EXPIRATION_MINUTES=60  # 1 hour
```

**Disable caching:**
```python
model = AutoModel.from_pretrained(
    "rfdetr-base",
    use_auto_resolution_cache=False
)
```

**Purge cache from filesystem:**
```bash
rm -rf $INFERENCE_HOME/auto-resolution-cache/
# or if using default location:
rm -rf /tmp/cache/auto-resolution-cache/
```

## 📦 Model Package Cache

### What Gets Cached

Downloaded model files (weights, configs, class names, etc.) are cached locally to avoid re-downloading on subsequent loads.

!!! warning "Cache Access and API Key Assumptions"

    **Important:** The model package cache operates with an implicit assumption that **once a model is stored on the local filesystem, it can be accessed without requiring an API key**, even if the original download required authentication.

    This means:

    - If a model was downloaded using an API key and cached locally, subsequent loads from cache will succeed **even without providing the API key**
    - The library itself does **not implement access control guards** for cached files
    - In single-user environments, this is typically the desired behavior for convenience

    **Multi-tenant environments:**

    When running in multi-tenant or shared environments (e.g., on the Roboflow platform), an **upstream guard layer** should be implemented to ensure proper access control. The Roboflow platform ships with such guards that:

    - Verify user permissions before allowing cache access
    - Ensure client models remain isolated and secure
    - Prevent unauthorized access to cached models from other tenants

    If you're deploying `inference-models` in a multi-tenant environment, you are responsible for implementing appropriate access control mechanisms at the application layer.

### Directory Structure

**Default location:** `$INFERENCE_HOME/models-cache/` (defaults to `/tmp/cache/models-cache/`)

Model IDs are slugified and hashed to create safe, unique, yet human-readable directory names. Package IDs (provided by the weights provider) are used as subdirectory names within each model directory.

**Example structure:**
```
/tmp/cache/
├── models-cache/
│   ├── yolov8n-640-a1b2c3d4/          # Slugified model ID + hash
│   │   ├── onnxfp32/                   # Package ID from provider
│   │   │   ├── model.onnx -> ../../shared-blobs/e4f5a6b7...
│   │   │   └── class_names.txt
│   │   └── trtfp16/                    # Another package ID
│   │       └── model.engine -> ../../shared-blobs/c8d9e0f1...
│   └── rfdetr-base-e5f6g7h8/
│       └── torchfp32/
│           └── model.pt -> ../../shared-blobs/a2b3c4d5...
└── shared-blobs/                       # Content-addressed blob storage
    ├── e4f5a6b7...                     # MD5 hash of file content
    ├── c8d9e0f1...
    └── a2b3c4d5...
```

### 🔗 Shared Blob Storage

When the weights provider supplies a content hash (MD5) for a file, the library stores the actual file in `$INFERENCE_HOME/shared-blobs/` named after its hash, and creates symlinks from the model package directories.

**Benefits:**

- **Avoids duplicate downloads** - If multiple models or packages share the same file (e.g., same weights with different configs), it's only downloaded once
- **Saves disk space** - Shared files are stored once and linked multiple times
- **Helps in bandwidth-constrained environments** - Particularly useful when working with multiple model variants or in offline/air-gapped deployments

Files without content hashes are stored directly in the model package directory.

### Cache Expiration

Model package cache **does not expire automatically** - files remain until manually deleted.

**Purge model cache:**
```bash
rm -rf $INFERENCE_HOME/models-cache/
# or if using default location:
rm -rf /tmp/cache/models-cache/
```

**Purge shared blobs:**
```bash
rm -rf $INFERENCE_HOME/shared-blobs/
# or if using default location:
rm -rf /tmp/cache/shared-blobs/
```

## 🚀 Next Steps

- [Understand Core Concepts](understand-core-concepts.md) - Understand the design philosophy
- [Supported Models](../models/index.md) - Browse available models
- [How-To: Local Packages](../how-to/local-packages.md) - Working with local model packages

