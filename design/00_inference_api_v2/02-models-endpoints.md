# Models endpoints

## POST `/v2/models/run`

Endpoint to run model inference.

### General ideas

User should be able to use full spectrum of actions available for all models through this endpoint. Vast majority of 
models would expose single action - `infer` (to simply run forward pass), other models, may actually have more 
elaborate interface - for instance CLIP which could produce text embeddings / image embeddings or compare query vs 
subject to return similarities. In such cases, in Workflows ecosystem, we usually had multiple blocks wrapping model 
functionalities - to reflect that we are going to have different actions possible for certain models.

User should be able to consume the API in multiple ways, depending on circumstances:

* simple, yet possibly slow requests methods should be exposed for the sake of simplicity and to avoid high technical 
entry bar

* more advanced, yet faster methods should be available for more advanced users when speed matters.

* both requests and responses should follow this idea


### Requests input formats

Three main ways of submitting requests should be possible:

1. `POST` request with query params specifying every parameter needed 
(constraint to simple types which can be represented this way, yet should probably be fine for 90% of users just 
trying things out).

```bash
curl -X POST https://serverless.roboflow.com/v2/models/run \
  -H "Authorization: Bearer <token>"
  --url-query 'model_id=whatever/model-id/we?can?figure-out' \
  --url-query 'image=https://images.com/my-super-awesome-image.jpg' \
  --url-query 'confidence=0.5'
```

Limitations:

* some artificial limit of query params size enforced by tools operating with HTTP requests 

* not able to represent locally available image as file, neither image available in-memory


2. `POST` request with JSON payload, encoding all inputs there (equivalent to the current state of workflows, where 
all the data is fed to payload and extensive decoding of all parts is needed - with the only exception being the fact 
that auth-related payload is pushed to headers) - effectively slow, but really handy - with all options regarding input 
formation available.

```bash
curl -X POST https://serverless.roboflow.com/v2/models/run \
  -H "Authorization: Bearer <token>"
  -H "Content-Type: application/json" \
  --url-query 'model_id=whatever/model-id/we?can?figure-out' \
  --d @- <<'EOF'
{
  "image": [
    {"type": "url", "value": "https://images.com/my-other-awesome-image.jpg"},
    {"type": "url", "value": "https://images.com/my-other-awesome-image.jpg"}
  ],
  "confidence": 0.3,
}
  EOF
```

3. `form/multipart` POST request with parameters stored in parts + optional designated part - `inputs` capable to store 
JSON document with definitions of inputs (and if needed, additional references)

Plain:
```bash
curl -X POST https://serverless.roboflow.com/v2/models/run \
  -H "Authorization: Bearer <token>"
  --url-query 'model_id=whatever/model-id/we?can?figure-out' \
  -F 'inputs={"confidence": 0.5};type=application/json' \
  -F "image=@/tmp/photo.jpg;type=image/jpeg" \
  -F "image=@/tmp/photo.jpg;type=image/jpeg"
```


With references:
```bash
curl -X POST https://serverless.roboflow.com/v2/models/run \
  -H "Authorization: Bearer <token>"
  --url-query 'model_id=whatever/model-id/we?can?figure-out' \
  -F 'inputs={"image": [$part.image1, $part.image2], "confidence": 0.5};type=application/json' \
  -F "image1=@/tmp/photo.jpg;type=image/jpeg" \
  -F "image2=@/tmp/photo.jpg;type=image/jpeg"
```


### Responses

Given that there are two orthogonal needs for response consumption:

* ease-of-understanding (which requires more verbose, easier to parse responses) 

* speed - which promotes sparse data representation (even if more extensive decoding is needed)

We should let clients choose what they want via: `response-style` (`rich` vs `compact`).

Default should be `rich`, `compact` should be selectable.

Some types of predictions (like instance segmentation maps or depth estimation maps), which produce dense responses, 
w/o clearly defined way to produce compacted / compressed form may additionally benefit from mechanism of multi-part response
which should be requestable via `response-format` (`json` - default, vs `multi-part`). In `multi-part` response, 
we propose having designated part (let's say `response`) which holds general JSON schema of the response, but additional
part (referred as `$part.name` in JSON document) would carry binary data relevant for specific data type - for instance
C-order dumped np array (tbd - specifics of this binary format, some header may be required).

Responses should come back to users within envelope, not in flat structure - that will help us over time to provide 
more clarity of what the API is doing (maybe usage reported?)

```json
{
  "type": "roboflow-inference-server-response-v1",
  "outputs": [  # workflows-style outputs
    {
      "model_results": {},
      "inference_id": "some"
    },
    {
      "predictions": {},
      "inference_id": "some"
    }
  ],
  # additional data in the envelope
}
```

Details of specific model responses outlined in the next sections.

### Structured query params

* `model_id` - provides model identifier, url-encoded if needed (required)
* `model_package_id` - provides identifier for specific model package to be used (optional)
* `response_style` (optional)
* `requested_output` (optional) - specifies filter for the model output to be provided
* `response_format`


### Representation of predictions

#### **classification (the classical one - single class)**

Our current response is the following:
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

There are a couple of issues with current representation:
* no way to apply confidence threshold - once done and "nothing is predicted" - we cannot return that to a client w/o breaking change - since our contract says `top` to be always provided as string (also `confidence` as `float`) - we may pretend we handle that using some constant out-of-class names value, but that's just hidding design flaw.
* current representation is slow to construct - in cases of dataset with large number of classes - construction of `predictions` takes ages - it's joke, especially when we consider using such classifier as secondary model (applied on top of cropped detections to map classes) - we've witnessed that being bottleneck of models running really fast, making substantial cut to processing FPS
* overhead in terms of size is also important - in the most minimalistic scenario possible, the information required to re-construct prediction is list of confidences - drastically smaller amount of bytes than we currently use - since network overheads seems to contribute to **vast majority of latency on serverless platform** - having slim entities representation is a desired end
* no way of trimming the representation to `top-n` classes
* `predictions` is everywhere - especially with workflows it leads to `data["predictions"]["predictions"]` very often which is confusing.

> [!NOTE]  
> While performance-wise, compact representations seems to be better - we should not forget another aspects - readability for users lowers entry-bar **and probably also makes predictions easier to digest for agents(?)**. Looks like another reason why to have both if feasable for given output type.

Proposed **_compact_** representation:
```json
{
    "type": "roboflow-classification-compact-v1",
    "class_names": ["cat", "dog"],
    "confidence": [0.6, 0.4],
    "confidence_threshold": 0.5,
    "predicted_class_ids": [0]
}
```
_it's tempting to remove `class_names` in favour of fetching this once by client and use for consecutive requests, but this may be inconvenient._

Proposed **_rich_** representation:
```json
{
    "type": "roboflow-classification-rich-v1",
    "candidates": [
         { "class_name": "car", "class_id": 0, "confidence": 0.08 },
         { "class_name": "cat", "class_id": 1, "confidence": 0.92 }
    ],
    "predicted_classes": [{ "class_name": "cat", "class_id": 1, "confidence": 0.92 }],
    "confidence_threshold": 0.5
}
```

#### **multi-label classification**

Our current response is the following:
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
This prediction divereges from single-label classification. There is a world where we have a common representation.

Proposed **_compact_** representation:
```json
{
    "type": "roboflow-classification-compact-v1",
    "class_names": ["cat", "dog"],
    "confidence": [0.6, 0.4],
    "confidence_threshold": 0.5,
    "predicted_class_ids": [0]
}
```
Proposed **_rich_** representation:
```json
{
    "type": "roboflow-classification-rich-v1",
    "candidates": [
         { "class_name": "car", "class_id": 0, "confidence": 0.08 },
         { "class_name": "cat", "class_id": 1, "confidence": 0.92 }
    ],
    "predicted_classes": [{ "class_name": "cat", "class_id": 1, "confidence": 0.92 }]
}
```

#### **object detection**

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
       }
    ]
}
```

Proposed **_compact_** representation:

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

Proposed **_rich_** representation:

```json
{
    "type": "roboflow-object-detection-rich-v1",
    "detections": [
        { 
            "left_top": [10, 20], 
            "right_bottom": [10, 20], 
            "confidence": 0.3, 
            "class_id": 0,
            "class_name": "some", 
            "tracker_id": 10, 
            "detection_id": "y"
       }
    ]
}
```

#### **instance segmentation**

Current representation
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
            "points": [{"x": 10, "y": 20}] # polygon representation
       }
    ]
}
```

Our current representation is problematic - namely certain shapes - we propose to double-down on @Borda proposal - compact cropped RLE representation: https://github.com/roboflow/supervision/pull/2159

Proposed **_compact_** representation:

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
        "offsets": [[100, 100], [10, 200]],
    }
}
```

Proposed **_rich_** representation:
```json
{
    "type": "roboflow-instance-segmentation-rich-v1",
    "image_size": [1920, 1080],
    "detections": [
        { 
            "left_top": [10, 20], 
            "right_bottom": [10, 20], 
            "confidence": 0.3, 
            "class_id": 0,
            "class_name": "some", 
            "tracker_id": 10, 
            "detection_id": "y",
            "rle_mask": {}  # current RLE format, w/o compaction
       }
    ]
}
```

##### Semantic segmentation

Current representation:
```json
{
    "segmentation_mask": "base64-encoded PNG of predicted class label at each pixel",
    "class_map": {"0": "cat", "1": "car"},
    "confidence_mask": "base64-encoded PNG of predicted class confidence at each pixel",
}
```

Since we are producing fully decompressed representation of responses, maybe base64 is actualy not needed.

Proposed representation:

```json
{
    "type": "roboflow-semantic-segmentation-v1",
    "pixels_scores": [[0.3, 0.4, ...]],
    "segmentation_map": [[]], # just dump of numpy mask (or alternativelly - only compavt format if we want to keep payload size regime)
    "class_names": ["cat", "dog"],
}
```

#### Other types

* Dense values, such as embeddings, similarity measurements or depth-estimations - should be provided back as dense arrays.
* text-only outputs should remain just simple texts
* structured OCR outputs - should be treated as special case of object-detection (additional field beyond class may be needed to differentiate type of object vs ocred content)


### Serialisation of data types 

Both inputs and outputs must be represented somehow. In some cases the representation will be relevant for 
transfer encoding type - for instance `image` data, when shipped via query param will be represented as URL to the image,
when shipped in multi-part will be either raw JPEG bytes or C-order dumped np.array as set of integers. When embedded 
into JSON - would take standard `inference` format: `{"type": "url | base64", "value": "xxx"}`.

We propose that we maintain set of opinionated representation for certain data types while transferred though the network.
Details here - to be agreed.

## GET `/v2/models/interface`

Endpoint to run get model interface. This endpoint is supposed to help users understanding the format of specific model
inputs / outputs, given we do have a single endpoint to run predictions. 

We are not able to rely on Swagger, since our contract is dynamic - so the proposed solution is a hybrid approach, where
part of endpoint interface is fixed across models (representation of specific data types, general formats of requests 
and responses, ways of shipping certain information in request). The interface discovery should help clients:

* understand what are parameters of the model and how to use them

* build custom clients on their own (would be great if fully possible using coding agents).


Proposed response format is the following:
```json
{ 
  "type": "roboflow-inference-server-model-interface-v1",
  "control_parameters": {
    "model_id": "...",
    "model_package_id": "..."
  },
  "request_formats": {
    "query_params_major": {
      "description": "Explain how params can be used to specify input",
      "technical_details": "..."
    },
    "json_payload": {
       "description": "Explain how params can be used to specify input",
        "technical_details": "..."
    },
    "multipart_request": {
       "description": "Explain how params can be used to specify input",
        "technical_details": "..."
    }
  },
  "model_inputs": {
    "name": {
      "description": "human-level description of the parameter",
      "type": "identifier-of-the-type",
      "representation": [
        {"$ref": "#/definitions/myElement", "relevant_request_formats": ["json_payload", "multipart_request"]}
      ]
    }
  },
  "model_outputs": {
    "name": {
      "description": "human-level description of the parameter",
      "type": "identifier-of-the-type",
      "representation": [
        {"$ref": "#/definitions/myElement", "response_style": "compact", "response_format": "json"}
      ]
    }
  },
  "definitions": {
    "ref-id": "swagger object definition - used to provide representation of data, referred elsewhere in the payload via $ref: '#/definitions/myElement'"  
  }
}


```

Additionally, GET `/v2/models/interface` should support query params:
* `model_id` - provides model identifier, url-encoded if needed (required)
* `response_style` (optional)
* `response_format`
* `request_format`
to filter out pieces not relevant for client.


Now - since this is a custom format of API description we need to figure out how best to build the representation 
server side. Since the majority of schema is fixed and can be constructed by server framework, we can provide 
helper functions to make that happened - what is important is how to easily allow per-model type declaration of 
the contract for anyone contributing to the server (without hardening the entry barer for contribution).

Proposition is as follows:
* define set of known types as python constants (being objects defining descriptions and swagger defs)
* define mappings of those types into inputs and outputs representations, such that those can be generated in the fly once declared
* let people use those definitions to effectively build mapping between input/output name and type



