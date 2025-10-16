### SAM3 Inference API – cURL Examples

This page documents practical cURL examples for using the SAM3 endpoints. It covers:

- SAM3 PVS (Promptable Visual Segmentation): `/sam3/embed_image`, `/sam3/visual_segment`
- SAM3 PCS (Promptable Concept Segmentation): `/sam3/concept_segment`

Replace `$API_KEY` with your Roboflow API key.

---

## SAM3 PVS (Promptable Visual Segmentation)

The PVS API mirrors the SAM2 interface and supports caching image embeddings for fast point/box segmentation.

#### 1) Embed Image (used for visual segmentation caching)

```bash
curl -X POST 'http://localhost:9001/sam3/embed_image?api_key=$API_KEY' \
  -H 'Content-Type: application/json' \
  -d '{
    "image": {
      "type": "url",
      "value": "https://images.pexels.com/photos/18001210/pexels-photo-18001210.jpeg"
    }
  }'
```

Notes:
- Response includes an `image_id` which can be reused to speed up subsequent visual segmentation calls.

#### 2) Visual Segmentation (points example; optionally include `image_id`)

```bash
curl -X POST 'http://localhost:9001/sam3/visual_segment?api_key=$API_KEY' \
  -H 'Content-Type: application/json' \
  -d '{
    "image": {
      "type": "url",
      "value": "https://images.pexels.com/photos/18001210/pexels-photo-18001210.jpeg"
    },
    "prompts": [
      { "points": [ { "x": 100, "y": 100, "positive": true } ] }
    ],
    "format": "rle"
  }'
```

---

## SAM3 PCS (Promptable Concept Segmentation)

The PCS API uses a unified `prompts` list. Each prompt can be:
- Text prompt: `{ "type": "text", "text": "dog" }`
- Visual prompt: `{ "type": "visual", "boxes": [ ... ], "box_labels": [ ... ], "text": "optional hint" }`

Boxes are absolute pixel coordinates and may be provided as:
- XYWH objects: `{ "x": <px>, "y": <px>, "width": <px>, "height": <px> }`
- XYXY objects: `{ "x0": <px>, "y0": <px>, "x1": <px>, "y1": <px> }`

You can mix XYWH and XYXY entries in the same list. `box_labels` must match the number of boxes; `1`=positive, `0`=negative.

The response returns `prompt_results[]`, one entry per prompt.

#### 1) Single Text Prompt (default polygon output)

```bash
curl -X POST 'http://localhost:9001/sam3/concept_segment?api_key=$API_KEY' \
  -H 'Content-Type: application/json' \
  -d '{
    "image": {
      "type": "url",
      "value": "https://images.pexels.com/photos/18001210/pexels-photo-18001210.jpeg"
    },
    "prompts": [ { "type": "text", "text": "raccoon" } ]
  }'
```

#### 2) Multiple Text Prompts (batched; one pass)

```bash
curl -X POST 'http://localhost:9001/sam3/concept_segment?api_key=$API_KEY' \
  -H 'Content-Type: application/json' \
  -d '{
    "image": { "type": "url", "value": "https://images.pexels.com/photos/777157/pexels-photo-777157.jpeg" },
    "prompts": [
      { "type": "text", "text": "car" },
      { "type": "text", "text": "person" },
      { "type": "text", "text": "bicycle" }
    ]
  }'
```

#### 3) Visual Prompt with XYWH Box

```bash
curl -X POST 'http://localhost:9001/sam3/concept_segment?api_key=$API_KEY' \
  -H 'Content-Type: application/json' \
  -d '{
    "image": { "type": "url", "value": "https://images.pexels.com/photos/18001210/pexels-photo-18001210.jpeg" },
    "prompts": [
      {
        "type": "visual",
        "boxes": [ { "x": 200, "y": 250, "width": 180, "height": 220 } ],
        "box_labels": [1]
      }
    ]
  }'
```

#### 4) Visual Prompt with XYXY Box

```bash
curl -X POST 'http://localhost:9001/sam3/concept_segment?api_key=$API_KEY' \
  -H 'Content-Type: application/json' \
  -d '{
    "image": { "type": "url", "value": "https://images.pexels.com/photos/18001210/pexels-photo-18001210.jpeg" },
    "prompts": [
      {
        "type": "visual",
        "boxes": [ { "x0": 220, "y0": 260, "x1": 380, "y1": 460 } ],
        "box_labels": [1]
      }
    ]
  }'
```

#### 5) Mixed Visual Boxes (XYWH and XYXY) in the Same Prompt, With Text Hint

```bash
curl -X POST 'http://localhost:9001/sam3/concept_segment?api_key=$API_KEY' \
  -H 'Content-Type: application/json' \
  -d '{
    "image": { "type": "url", "value": "https://images.pexels.com/photos/18001210/pexels-photo-18001210.jpeg" },
    "prompts": [
      {
        "type": "visual",
        "text": "dog",
        "boxes": [
          { "x": 120, "y": 180, "width": 160, "height": 140 },
          { "x0": 400, "y0": 200, "x1": 520, "y1": 360 }
        ],
        "box_labels": [1, 0]
      }
    ]
  }'
```

#### 6) Batch Mixed Prompts: Text + Visual in One Pass

```bash
curl -X POST 'http://localhost:9001/sam3/concept_segment?api_key=$API_KEY' \
  -H 'Content-Type: application/json' \
  -d '{
    "image": { "type": "url", "value": "https://images.pexels.com/photos/18001210/pexels-photo-18001210.jpeg" },
    "prompts": [
      { "type": "text", "text": "person" },
      {
        "type": "visual",
        "boxes": [ { "x": 50, "y": 60, "width": 120, "height": 150 } ],
        "box_labels": [1]
      }
    ]
  }'
```

#### 7) RLE Output Format

```bash
curl -X POST 'http://localhost:9001/sam3/concept_segment?api_key=$API_KEY' \
  -H 'Content-Type: application/json' \
  -d '{
    "image": { "type": "url", "value": "https://images.pexels.com/photos/18001210/pexels-photo-18001210.jpeg" },
    "prompts": [ { "type": "text", "text": "raccoon" } ],
    "format": "rle"
  }'
```

#### 8) Adjust Output Threshold

```bash
curl -X POST 'http://localhost:9001/sam3/concept_segment?api_key=$API_KEY' \
  -H 'Content-Type: application/json' \
  -d '{
    "image": { "type": "url", "value": "https://images.pexels.com/photos/18001210/pexels-photo-18001210.jpeg" },
    "prompts": [ { "type": "text", "text": "raccoon" } ],
    "output_prob_thresh": 0.35
  }'
```


