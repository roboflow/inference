### v1 multipart endpoints (MVP)

These endpoints are available under `/v1` alongside existing APIs. They accept header-based auth and multipart image uploads to avoid base64-in-JSON overhead.

- Auth headers:
  - `Authorization: Bearer <api_key>` (preferred)
  - or `X-API-Key: <api_key>`

#### Object detection (multipart)

```bash
curl -sS -X POST "http://localhost:9001/v1/infer/object_detection" \
  -H "Authorization: Bearer $ROBOFLOW_API_KEY" \
  -F "model_id=workspace/model" \
  -F "confidence=0.4" \
  -F "iou_threshold=0.3" \
  -F "image=@/path/to/image.jpg"
```

Batch multiple images by repeating the `-F "image=@..."` field.

#### Classification (multipart)

```bash
curl -sS -X POST "http://localhost:9001/v1/infer/classification" \
  -H "Authorization: Bearer $ROBOFLOW_API_KEY" \
  -F "model_id=workspace/model" \
  -F "image=@/path/to/image.jpg"
```

#### Workflows (multipart)

```bash
curl -sS -X POST "http://localhost:9001/v1/workflows/run" \
  -H "Authorization: Bearer $ROBOFLOW_API_KEY" \
  -F 'spec={"nodes":[...],"inputs":{"image":"image"}}' \
  -F 'inputs={"param1":"value"}' \
  -F "image=@/path/to/image.jpg"
```

Notes:
- `spec` is a JSON string with your Workflow specification.
- Non-image parameters go in `inputs` (JSON string).
- Image is bound to the `image` input name by convention for MVP.


