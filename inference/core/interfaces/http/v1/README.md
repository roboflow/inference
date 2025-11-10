# V1 HTTP API Documentation

The V1 API provides a modern, efficient interface for the Roboflow Inference Server with significant performance and architectural improvements over the legacy API.

## Key Features

### 1. Header-Based Authentication
- **Old API**: API keys in request bodies (requires full body parsing before auth)
- **New API**: API keys in headers (auth before body parsing)

This enables:
- Standard reverse proxy authentication
- Better security isolation
- Faster request rejection for unauthorized requests
- Compatibility with OAuth2/Bearer token patterns

### 2. Multipart Form Data for Images
- **Old API**: Base64-encoded images in JSON (33% size overhead, slow JSON parsing)
- **New API**: Binary image uploads via multipart/form-data (no encoding overhead)

Performance improvements:
- ~25-50% faster for large images
- ~25% smaller payload sizes
- Lower memory usage during parsing
- Direct binary-to-numpy conversion

### 3. Clean RESTful Design
- Versioned endpoints (`/v1/...`)
- Consistent URL patterns
- Clear resource hierarchy
- Improved OpenAPI documentation

## Authentication

### Supported Methods

#### 1. Authorization Header (Recommended)
```bash
curl -X POST http://localhost:9001/v1/object-detection/my-model/1 \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -F "image=@photo.jpg" \
  -F 'config={"confidence": 0.5}'
```

#### 2. Custom Header
```bash
curl -X POST http://localhost:9001/v1/object-detection/my-model/1 \
  -H "X-Roboflow-Api-Key: YOUR_API_KEY" \
  -F "image=@photo.jpg" \
  -F 'config={"confidence": 0.5}'
```

#### 3. Query Parameter (Fallback)
```bash
curl -X POST "http://localhost:9001/v1/object-detection/my-model/1?api_key=YOUR_API_KEY" \
  -F "image=@photo.jpg" \
  -F 'config={"confidence": 0.5}'
```

## Model Inference Endpoints

### Object Detection

**Endpoint**: `POST /v1/object-detection/{model_id}`

**Request Format** (multipart/form-data):
```bash
curl -X POST http://localhost:9001/v1/object-detection/construction-safety/3 \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -F "image=@photo.jpg" \
  -F 'config={
    "confidence": 0.5,
    "iou_threshold": 0.4,
    "max_detections": 300,
    "visualize_predictions": false
  }'
```

**Config Options**:
- `confidence` (float): Detection confidence threshold (0-1)
- `iou_threshold` (float): IoU threshold for NMS (0-1)
- `max_detections` (int): Maximum number of detections
- `visualize_predictions` (bool): Return visualization image
- `visualization_labels` (bool): Show labels in visualization
- `visualization_stroke_width` (int): Stroke width for visualization

### Instance Segmentation

**Endpoint**: `POST /v1/instance-segmentation/{model_id}`

Same format as object detection, with additional config:
- `mask_decode_mode` (string): `"accurate"` or `"fast"`
- `tradeoff_factor` (float): Speed vs accuracy tradeoff (0-1)

### Classification

**Endpoint**: `POST /v1/classification/{model_id}`

**Request Format**:
```bash
curl -X POST http://localhost:9001/v1/classification/animal-classifier/2 \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -F "image=@photo.jpg" \
  -F 'config={"confidence": 0.5}'
```

### Keypoint Detection

**Endpoint**: `POST /v1/keypoint-detection/{model_id}`

Additional config:
- `keypoint_confidence` (float): Keypoint confidence threshold (0-1)

## Workflow Endpoints

### Run Predefined Workflow

**Endpoint**: `POST /v1/workflows/{workspace_id}/{workflow_id}`

**Convention-Based Image Matching**: Multipart field names should match workflow input names.

**Example**: Workflow with inputs `["image", "confidence", "prompt"]`

```bash
curl -X POST http://localhost:9001/v1/workflows/my-workspace/my-workflow \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -F "image=@photo.jpg" \
  -F 'inputs={"confidence": 0.5, "prompt": "detect people"}'
```

The `image` multipart field will be matched to the `image` workflow input.

**Multiple Images**:
```bash
curl -X POST http://localhost:9001/v1/workflows/my-workspace/my-workflow \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -F "image=@photo.jpg" \
  -F "mask=@mask.png" \
  -F 'inputs={"confidence": 0.5}'
```

### Run Inline Workflow

**Endpoint**: `POST /v1/workflows/run`

```bash
curl -X POST http://localhost:9001/v1/workflows/run \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -F "image=@photo.jpg" \
  -F 'specification={"version": "1.0", "steps": [...]}' \
  -F 'inputs={"confidence": 0.5}'
```

## Migration Guide

### From Legacy API to V1

#### Object Detection

**Old API** (`/infer/object_detection`):
```python
import requests
import base64

with open("image.jpg", "rb") as f:
    image_bytes = f.read()
    image_base64 = base64.b64encode(image_bytes).decode()

response = requests.post(
    "http://localhost:9001/infer/object_detection",
    json={
        "model_id": "my-project/1",
        "api_key": "YOUR_API_KEY",  # ❌ API key in body
        "image": {
            "type": "base64",
            "value": image_base64,  # ❌ Base64 encoded
        },
        "confidence": 0.5,
    }
)
```

**New V1 API** (`/v1/object-detection/{model_id}`):
```python
import requests

with open("image.jpg", "rb") as f:
    image_bytes = f.read()

response = requests.post(
    "http://localhost:9001/v1/object-detection/my-project/1",
    headers={
        "Authorization": f"Bearer YOUR_API_KEY",  # ✅ API key in header
    },
    files={
        "image": ("image.jpg", image_bytes, "image/jpeg"),  # ✅ Binary upload
        "config": (None, '{"confidence": 0.5}', "application/json"),
    }
)
```

#### Workflows

**Old API** (`/{workspace}/workflows/{workflow_id}`):
```python
response = requests.post(
    "http://localhost:9001/my-workspace/workflows/my-workflow",
    json={
        "api_key": "YOUR_API_KEY",  # ❌ API key in body
        "inputs": {
            "image": {
                "type": "base64",
                "value": image_base64,  # ❌ Base64 encoded
            },
            "confidence": 0.5,
        }
    }
)
```

**New V1 API** (`/v1/workflows/{workspace}/{workflow_id}`):
```python
response = requests.post(
    "http://localhost:9001/v1/workflows/my-workspace/my-workflow",
    headers={
        "Authorization": f"Bearer YOUR_API_KEY",  # ✅ API key in header
    },
    files={
        "image": ("image.jpg", image_bytes, "image/jpeg"),  # ✅ Binary upload
        "inputs": (None, '{"confidence": 0.5}', "application/json"),
    }
)
```

## Python Client Example

```python
import requests
from pathlib import Path


class RoboflowV1Client:
    """Simple client for Roboflow V1 API."""

    def __init__(self, host: str, api_key: str):
        self.host = host.rstrip("/")
        self.api_key = api_key
        self.headers = {"Authorization": f"Bearer {api_key}"}

    def object_detection(
        self,
        model_id: str,
        image_path: str,
        confidence: float = 0.5,
        **kwargs
    ):
        """Run object detection inference."""
        url = f"{self.host}/v1/object-detection/{model_id}"

        with open(image_path, "rb") as f:
            image_bytes = f.read()

        config = {"confidence": confidence, **kwargs}

        files = {
            "image": (Path(image_path).name, image_bytes, "image/jpeg"),
            "config": (None, str(config).replace("'", '"'), "application/json"),
        }

        response = requests.post(url, headers=self.headers, files=files)
        response.raise_for_status()
        return response.json()

    def workflow(
        self,
        workspace_id: str,
        workflow_id: str,
        images: dict,
        inputs: dict = None,
    ):
        """Run workflow inference."""
        url = f"{self.host}/v1/workflows/{workspace_id}/{workflow_id}"

        files = {}

        # Add images
        for name, path in images.items():
            with open(path, "rb") as f:
                files[name] = (Path(path).name, f.read(), "image/jpeg")

        # Add inputs
        if inputs:
            files["inputs"] = (None, str(inputs).replace("'", '"'), "application/json")

        response = requests.post(url, headers=self.headers, files=files)
        response.raise_for_status()
        return response.json()


# Usage
client = RoboflowV1Client(
    host="http://localhost:9001",
    api_key="YOUR_API_KEY"
)

# Object detection
result = client.object_detection(
    model_id="construction-safety/3",
    image_path="photo.jpg",
    confidence=0.6,
)

# Workflow with multiple images
result = client.workflow(
    workspace_id="my-workspace",
    workflow_id="my-workflow",
    images={
        "image": "photo.jpg",
        "mask": "mask.png",
    },
    inputs={"confidence": 0.5},
)
```

## Performance Benchmarking

Use the included benchmark tool to measure improvements:

```bash
python -m tests.benchmark.api_performance \
    --host http://localhost:9001 \
    --api-key YOUR_API_KEY \
    --model-id your-project/version \
    --image-path test_image.jpg \
    --runs 10
```

See `tests/benchmark/README.md` for detailed benchmarking documentation.

## Backwards Compatibility

**All legacy endpoints remain functional.** The v1 API is additive, not replacing.

- Legacy endpoints: Continue working as before
- V1 endpoints: New, optimized alternatives
- No breaking changes to existing integrations

## API Version Header

All v1 responses include:
```
X-Inference-Api-Version: v1.0
```

## Error Responses

V1 API uses standard HTTP status codes:

- `200 OK`: Success
- `400 Bad Request`: Invalid request (missing image, invalid JSON, etc.)
- `401 Unauthorized`: Missing or invalid API key
- `404 Not Found`: Model or workflow not found
- `500 Internal Server Error`: Server error

Error format:
```json
{
    "status": 401,
    "message": "Invalid or unauthorized API key"
}
```

## OpenAPI Documentation

Full interactive API documentation available at:
- Swagger UI: `http://localhost:9001/docs`
- ReDoc: `http://localhost:9001/redoc`

Filter by "v1" tag to see only v1 endpoints.
