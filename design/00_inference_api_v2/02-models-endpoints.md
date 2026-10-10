# Models endpoints

## `POST /v2/models/run`

Endpoint for running model inference.

### General ideas

A client should be able to drive the full action surface of any model through this single endpoint. The vast majority of models expose just one action — `infer` (a forward pass). Some have richer interfaces; CLIP, for example, can produce text embeddings, image embeddings, or compare a query against a subject and return similarities. In the workflows ecosystem these have historically been split across multiple blocks. To preserve that flexibility under a single endpoint, `v2` admits per-model action sets.

The API must serve two distinct audiences without compromising either:

* **Casual clients** — for whom a simple request shape and a verbose response shape lower the entry bar. Simple requests may be slower; that is acceptable.
* **Performance-sensitive clients** — for whom transport overhead matters and a compact, denser representation is worth the extra decoding effort.

The same idea applies on both sides of the wire: request and response.

### Request input formats

Three submission formats are supported.

**1. Query-only `POST`.** Every parameter is supplied as a URL-encoded query param. Useful for ~90% of casual users who are exploring the API. Constrained to simple, URL-representable types.

```bash
curl -X POST https://serverless.roboflow.com/v2/models/run \
  -H "Authorization: Bearer <token>" \
  --url-query 'model_id=whatever/model-id/we?can?figure-out' \
  --url-query 'image=https://images.com/my-super-awesome-image.jpg' \
  --url-query 'confidence=0.5'
```

Limitations:

* Most HTTP tooling imposes an upper bound on combined URL length (commonly 4–8 KiB); large parameter sets do not fit.
* Locally available files and in-memory image buffers cannot be transported this way; only references (URLs) are practical.

**2. JSON body `POST`.** All inputs encoded in a JSON document, equivalent in spirit to today's workflows request shape, with the single change that authentication moves to headers. Slow for large binary inputs (base64 overhead) but the most expressive option.

```bash
curl -X POST https://serverless.roboflow.com/v2/models/run \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  --url-query 'model_id=whatever/model-id/we?can?figure-out' \
  -d @- <<'EOF'
{
  "inputs": {
    "image": [
      {"type": "url", "value": "https://images.com/my-other-awesome-image.jpg"},
      {"type": "url", "value": "https://images.com/my-other-awesome-image.jpg"}
    ],
    "confidence": 0.3
  }
}
EOF
```

**3. Multipart `POST`.** Parameters are split into parts. A designated `inputs` part carries the JSON document with input definitions and may reference sibling parts as `$part.<name>`. Most efficient for large binary inputs (no base64 expansion, no JSON-parse overhead on the binary blob).

Plain form — every input named directly:

```bash
curl -X POST https://serverless.roboflow.com/v2/models/run \
  -H "Authorization: Bearer <token>" \
  --url-query 'model_id=whatever/model-id/we?can?figure-out' \
  -F 'inputs={"confidence": 0.5};type=application/json' \
  -F "image=@/tmp/photo1.jpg;type=image/jpeg" \
  -F "image=@/tmp/photo2.jpg;type=image/jpeg"
```

With explicit part references — `inputs` names other parts via `$part.<name>`:

```bash
curl -X POST https://serverless.roboflow.com/v2/models/run \
  -H "Authorization: Bearer <token>" \
  --url-query 'model_id=whatever/model-id/we?can?figure-out' \
  -F 'inputs={"image": [$part.image1, $part.image2], "confidence": 0.5};type=application/json' \
  -F "image1=@/tmp/photo1.jpg;type=image/jpeg" \
  -F "image2=@/tmp/photo2.jpg;type=image/jpeg"
```

### Responses

Two orthogonal needs pull the response shape in opposite directions:

* **Ease of understanding** — favours verbose, self-describing structures that a human or LLM can read directly.
* **Speed** — favours sparse, columnar representations that minimise byte count, even when they require extra decoding on the client.

To accommodate both, the client selects via the `response_style` query parameter:

* `rich` (default) — verbose, self-describing, easy to consume.
* `compact` — column-oriented, minimised payload.

Some prediction types (notably instance-segmentation masks and depth maps) produce dense outputs that do not compress cleanly inside JSON. For these, a multi-part response is selectable via `response_format`:

* `json` (default) — single JSON document.
* `multipart` — a multi-part HTTP response in which one part (`response`) carries a JSON envelope and additional parts carry binary blobs. The JSON envelope refers to binary parts as `$part.<name>`, mirroring the request-side convention.

**Binary part encoding (multipart responses).** The default binary encoding is NumPy's native `.npy` format — self-describing (shape, dtype, byte-order, fortran-vs-C order all in the part body), well-known, and trivially read with `numpy.load(io.BytesIO(part_bytes))`. The part carries `Content-Type: application/x-numpy-array`. For clients that cannot depend on NumPy, an alternative encoding is raw little-endian bytes with shape/dtype carried in part-level `X-Array-*` headers; this remains an open question (see "Open questions" below).

All responses are wrapped in a top-level envelope. This is intentional — it gives us a stable place to surface metadata (usage, server identity, deprecation notices) without changing the prediction payload itself.

```json
{
  "type": "roboflow-inference-server-response-v1",
  "outputs": [
    {
      "model_results": {},
      "inference_id": "..."
    },
    {
      "predictions": {},
      "inference_id": "..."
    }
  ]
}
```

Per-model response shapes are spelled out in the next section.

### Structured query params

The following query parameters are recognised across all `POST /v2/models/run` requests, regardless of the input format:

| Name | Required | Description |
|---|---|---|
| `model_id` | yes | Model identifier, URL-encoded if it contains reserved characters. |
| `model_package_id` | no | Identifier of a specific model package (e.g., a particular quantisation or backend build). |
| `response_style` | no | `rich` (default) or `compact`. |
| `response_format` | no | `json` (default) or `multipart`. |
| `requested_output` | no | Filter restricting which outputs are returned. Repeatable. |

### Representation of predictions

#### Classification (single-label)

Current representation:

```json
{
  "predictions": [
    {"class_name": "car", "class_id": 0, "confidence": 0.6},
    {"class_name": "cat", "class_id": 1, "confidence": 0.4}
  ],
  "confidence": 0.6,
  "top": "car",
  "parent_id": "<workflow-specific-metadata>"
}
```

Problems with the current shape:

* **No "nothing predicted" path.** Once a confidence threshold is applied, an empty result cannot be expressed without breaking the contract — `top` is typed as a non-null string and `confidence` as a non-null float. Returning a sentinel value like `"unknown"` only hides the design flaw.
* **Slow to construct on wide class sets.** For classifiers with many classes (e.g., a secondary classifier applied to cropped detections), building the per-class `predictions` list dominates per-inference cost — we have seen it become the FPS bottleneck on otherwise fast pipelines.
* **Payload size matters on serverless.** Network overhead is the dominant latency contributor for hosted serverless inference. The minimal information required to reconstruct a prediction is a single vector of confidences plus a class-name lookup, which is dramatically smaller than the current per-class object list.
* **No `top-n` trimming.** There is no way to ask for "the top 5 classes only".
* **`predictions` ambiguity in workflows.** Because every prediction type uses a top-level `predictions` key, workflow consumers end up with `data["predictions"]["predictions"]` constructions, which are confusing.

> [!NOTE]
> Compact representations are strictly better on bytes-on-the-wire. They are not strictly better on readability — verbose forms lower the entry bar for human users and (probably) for coding agents reading responses. Hence both styles are first-class.

**Proposed _compact_ representation:**

```json
{
  "type": "roboflow-classification-compact-v1",
  "class_names": ["cat", "dog"],
  "confidence": [0.6, 0.4],
  "confidence_threshold": 0.5,
  "predicted_class_ids": [0]
}
```

It is tempting to drop `class_names` and have the client fetch the class list once (e.g., via the interface endpoint) and reuse it across requests — that would shave further bytes but introduces a stateful client contract. For now, `class_names` is kept on every response.

**Proposed _rich_ representation:**

```json
{
  "type": "roboflow-classification-rich-v1",
  "candidates": [
    {"class_name": "car", "class_id": 0, "confidence": 0.08},
    {"class_name": "cat", "class_id": 1, "confidence": 0.92}
  ],
  "predicted_classes": [{"class_name": "cat", "class_id": 1, "confidence": 0.92}],
  "confidence_threshold": 0.5
}
```

#### Classification (multi-label)

Current representation:

```json
{
  "predictions": {
    "cat": {"class_id": 0, "confidence": 0.7},
    "dog": {"class_id": 1, "confidence": 0.7}
  },
  "predicted_classes": ["cat", "dog"],
  "parent_id": "<workflow-specific-metadata>"
}
```

This shape diverges from single-label classification without good reason. We unify the two: the compact and rich shapes proposed above are reused as-is. Single-label is the special case `len(predicted_class_ids) <= 1`; multi-label allows any length.

**Proposed _compact_ representation** (same `type` string as single-label by design):

```json
{
  "type": "roboflow-classification-compact-v1",
  "class_names": ["cat", "dog"],
  "confidence": [0.6, 0.4],
  "confidence_threshold": 0.5,
  "predicted_class_ids": [0]
}
```

**Proposed _rich_ representation:**

```json
{
  "type": "roboflow-classification-rich-v1",
  "candidates": [
    {"class_name": "car", "class_id": 0, "confidence": 0.08},
    {"class_name": "cat", "class_id": 1, "confidence": 0.92}
  ],
  "predicted_classes": [{"class_name": "cat", "class_id": 1, "confidence": 0.92}],
  "confidence_threshold": 0.5
}
```

#### Object detection

Current representation:

```json
{
  "predictions": [
    {
      "x": 10,
      "y": 20,
      "width": 100,
      "height": 200,
      "confidence": 0.3,
      "class_id": 0,
      "class_name": "some",
      "tracker_id": "x",
      "detection_id": "y"
    }
  ]
}
```

**Proposed _compact_ representation** (struct-of-arrays — parallel lists, all aligned by index):

```json
{
  "type": "roboflow-object-detection-compact-v1",
  "class_names": ["list", "of", "all", "classes"],
  "xyxy": [
    [0, 1, 2, 3],
    [0, 1, 2, 3]
  ],
  "class_id": [0, 1],
  "confidence": [0.33, 0.64],
  "tracker_id": [0, 1]
}
```

**Proposed _rich_ representation:**

```json
{
  "type": "roboflow-object-detection-rich-v1",
  "detections": [
    {
      "left_top": [10, 20],
      "right_bottom": [110, 220],
      "confidence": 0.3,
      "class_id": 0,
      "class_name": "some",
      "tracker_id": 10,
      "detection_id": "y"
    }
  ]
}
```

Note the move from `(x, y, width, height)` (centre + size) to `(left_top, right_bottom)` corner pairs in the rich form, and to a flat `xyxy` matrix in the compact form. This aligns with how `supervision` and most downstream consumers actually use detections.

#### Instance segmentation

Current representation:

```json
{
  "predictions": [
    {
      "x": 10,
      "y": 20,
      "width": 100,
      "height": 200,
      "confidence": 0.3,
      "class_id": 0,
      "class_name": "some",
      "tracker_id": "x",
      "detection_id": "y",
      "points": [{"x": 10, "y": 20}]
    }
  ]
}
```

The polygon representation (`points`) is problematic for certain shapes (concave masks, masks with holes, near-pixel-perfect masks). We adopt @Borda's compact cropped-RLE proposal: <https://github.com/roboflow/supervision/pull/2159>.

**Proposed _compact_ representation:**

```json
{
  "type": "roboflow-instance-segmentation-compact-v1",
  "class_names": ["list", "of", "all", "classes"],
  "xyxy": [
    [0, 1, 2, 3],
    [0, 1, 2, 3]
  ],
  "class_id": [0, 1],
  "confidence": [0.33, 0.64],
  "tracker_id": [0, 1],
  "mask": {
    "type": "roboflow-compact-cropped-rle-mask-v1",
    "image_size": [1920, 1080],
    "rles": [[1, 3, 2]],
    "crop_shapes": [[100, 100], [10, 200]],
    "offsets": [[100, 100], [10, 200]]
  }
}
```

**Proposed _rich_ representation:**

```json
{
  "type": "roboflow-instance-segmentation-rich-v1",
  "image_size": [1920, 1080],
  "detections": [
    {
      "left_top": [10, 20],
      "right_bottom": [110, 220],
      "confidence": 0.3,
      "class_id": 0,
      "class_name": "some",
      "tracker_id": 10,
      "detection_id": "y",
      "rle_mask": {}
    }
  ]
}
```

The `rle_mask` shape inside the rich representation is the current full-image RLE format, without the cropping optimisation — readable per-detection, at the cost of larger payloads when there are many small masks in a large image.

#### Semantic segmentation

Current representation:

```json
{
  "segmentation_mask": "base64-encoded PNG of predicted class label at each pixel",
  "class_map": {"0": "cat", "1": "car"},
  "confidence_mask": "base64-encoded PNG of predicted class confidence at each pixel"
}
```

Since `v2` already supports a multipart response format for dense outputs, base64-encoded PNGs are no longer the only option. Two response styles map naturally to the two response formats:

**Proposed representation (JSON response format — `rich` style):**

```json
{
  "type": "roboflow-semantic-segmentation-v1",
  "class_names": ["cat", "dog"],
  "segmentation_map": [[]],
  "pixel_scores": [[]]
}
```

`segmentation_map` and `pixel_scores` are dense per-pixel arrays (class id and class confidence respectively). They are JSON-encoded only when the client accepts the size cost — typically for small masks or for debugging.

**Proposed representation (multipart response format):** the `response` part holds the JSON envelope with `$part.segmentation_map` and `$part.pixel_scores` references, and two binary parts carry the arrays in the encoding described above.

#### Other types

* **Dense numeric outputs** — embeddings, similarity scores, depth maps — are returned as dense numeric arrays (JSON list-of-lists for `rich`/JSON; native binary array for `multipart`).
* **Text-only outputs** remain plain strings.
* **Structured OCR outputs** are treated as a special case of object detection. An extra field beyond `class_name` is required to distinguish the recognised text content from the class label (e.g., `class_name` = "word", plus a `text` field carrying the recognised characters).

### Serialisation of data types

Inputs and outputs need a wire representation. For some data types the representation depends on the transport: an `image`, for example, is sent as a URL when transported via query params, as raw JPEG bytes (or a C-order NumPy dump) when transported via a multipart part, and as a JSON object (`{"type": "url|base64", "value": "..."}`) when embedded in a JSON body.

`v2` commits to a fixed catalogue of opinionated representations per data type and per transport. The catalogue is part of the contract and is discoverable via `/v2/models/interface`.

**Open questions:**

* Exact byte-level format for multipart binary parts. Default proposal: NumPy native (`application/x-numpy-array`). Fallback for non-NumPy clients: raw little-endian bytes with `X-Array-Shape`, `X-Array-Dtype`, `X-Array-Byteorder` headers on the part. Decision deferred until we have at least one client implementation that needs the fallback.
* Whether `$part.<name>` references inside JSON bodies should support nested-attribute access (e.g., `$part.frame.timestamp`) or remain strictly part-level.
* Versioning policy for `type` strings — do we treat a `*-v1` → `*-v2` bump as breaking, or do we keep `*-v1` callable indefinitely with additive fields only?

## `GET /v2/models/interface`

The interface endpoint exists because Swagger is the wrong tool here: our contract is dynamic. The valid set of inputs (and their meanings) depends on which model the client is calling. Swagger assumes static schemas per endpoint.

The proposed approach is hybrid:

* Parts of the contract are **fixed** across models — the three request formats, the response envelope, the data-type representation catalogue.
* Parts are **per-model** — which named inputs the model accepts, what their data types are, which outputs come back, and how they are shaped.

`/v2/models/interface` returns both halves of the contract for a given model so that a client (human or coding agent) can:

* understand which parameters the model takes, what they mean, and how to pass them;
* generate a custom client for the model without reading prose docs.

Proposed response format:

```json
{
  "type": "roboflow-inference-server-model-interface-v1",
  "control_parameters": {
    "model_id": "...",
    "model_package_id": "..."
  },
  "request_formats": {
    "query_params_major": {
      "description": "Human-level explanation of how inputs are supplied as query parameters.",
      "technical_details": "..."
    },
    "json_payload": {
      "description": "Human-level explanation of the JSON body shape.",
      "technical_details": "..."
    },
    "multipart_request": {
      "description": "Human-level explanation of multipart parts and $part.<name> references.",
      "technical_details": "..."
    }
  },
  "model_inputs": {
    "<input_name>": {
      "description": "Human-level description of the input.",
      "type": "<type-identifier>",
      "representation": [
        {"$ref": "#/definitions/<defId>", "relevant_request_formats": ["json_payload", "multipart_request"]}
      ]
    }
  },
  "model_outputs": {
    "<output_name>": {
      "description": "Human-level description of the output.",
      "type": "<type-identifier>",
      "representation": [
        {"$ref": "#/definitions/<defId>", "response_style": "compact", "response_format": "json"}
      ]
    }
  },
  "definitions": {
    "<defId>": "OpenAPI/JSON-Schema fragment describing the representation. Referenced from elsewhere in the document via $ref: '#/definitions/<defId>'."
  }
}
```

### Query parameters

| Name | Required | Description |
|---|---|---|
| `model_id` | yes | Model identifier, URL-encoded if needed. |
| `response_style` | no | If set, the response filters representations down to the requested style (`rich`/`compact`). |
| `response_format` | no | If set, the response filters representations down to the requested format (`json`/`multipart`). |
| `request_format` | no | If set, the response filters representations down to the requested request transport (`query_params_major`/`json_payload`/`multipart_request`). |

Filtering trims the document; it does not change the contract. A client that wants the full picture omits these.

### Server-side implementation sketch

Because this is a custom interface format (not Swagger), the server has to build the document. The bulk of the schema is fixed and can be assembled by framework code; the per-model bits are the only thing model contributors should have to write.

Proposed shape:

* A registry of **known input/output types**, each declared once as a Python object that carries (a) a human-level description, (b) the OpenAPI/JSON-Schema fragment(s) for each representation it supports, and (c) the mapping from `(transport, response_style, response_format)` to which representation is used.
* A helper that takes such a type object and produces the right `model_inputs` / `model_outputs` entry on the fly.
* A per-model declaration that maps input/output names to types, e.g.:

  ```python
  ModelInterface(
      inputs={
          "image": IMAGE_INPUT_TYPE,
          "confidence": CONFIDENCE_INPUT_TYPE,
      },
      outputs={
          "predictions": OBJECT_DETECTION_OUTPUT_TYPE,
      },
  )
  ```

The goal is that a contributor adding a new model writes a small declarative block, not a schema document. The framework assembles the introspection response from that declaration.

**Open question:** how to version the catalogue of known types so that adding fields to a representation is non-breaking but renaming/removing them is explicitly breaking. Suggested starting point: every `type` string carries a `-vN` suffix; new fields are additive within a major version; field removal forces a `-v(N+1)` bump.
