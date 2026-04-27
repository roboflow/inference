# Models endpoints

## POST `/v2/models/infer`

Endpoint to run model inference.

### Structured query params

* `model_id` - provides model identifier, url-encoded if needed (required)

This should be the way, we suggest people to encode `model_id` uisng `curl`

```bash
curl -X POST https://serverless.roboflow.com/v2/models/infer \
    --data-urlencode 'model_id=whatever/model-id/we?can?figure-out'
```

> [!IMPORTANT]  
> Since we have `inference-models` and one model may have multiple model-packages, `model_package_id` is natural candidate for structured query param - letting clients specify which exact model package they want, altering dafault auto-loading choice. We can decide also that certain parameters of auto-loading should be possible to be passed (although we need to decide on that relatively fast due to engineering work in progress and impact on model manager).


### Ways to deliver model inputs

Given the correspondence between standalone model inference and single-step workflow execution, for each model we can **deduce input interface**.
That ends up being list of inputs to the `run(...)` / `infer(...)`  method, each ether providing _scalars (like confidence thresholds, filtering params etc)_ or 
_batch-oriented_ data (for which model inference is applied as SIMD operation, or treated like auxiliary batched input).

Having such interface after determining model metadata (based on `model_id`) - makes it possible to retrieve / validate those parameters dynamically
in handler runtime, using at most 2 out of 3 sources at a time (query params alone, JSON body alone, form-multipart chunks alone, query + JSON body or query + form-multipart).

In general, regarding names of inputs we can:
* make **a gentelmens agreement we will re-use common, sensible names for things** like `image`, `confidence`, etc.
* expect names of all inputs will simply map to names of params in `run(...)` / `infer(...)`  method we use under the hood - exactly as we do in `workflows` (plus obviously validators provided to mark proper HTTP response codes)
* assume some precedence of param sources in case overlap is detected, or raise error on overlap - tbd
* for special types of data inputs (like images), assume each input source to have own representation, for instance `image` shipped as query param to only be shipped as URL, versus in `form-multipart` as encoded JPEG / PNG bytes or raw unit8 bytes we decode directly to tensor
* We should agree on list of _**structured query params**_, as those may require special treatment (modifiers not actual input-output parameters) - **potential risk of collision** (we need mitigation strategy, either formally enforcable or gentelments agreement)
* batch input definition should be done via multiplying params (like `image` part / query param duplicated) _TODO: verify all clients and servers we care about support that without issues_

> [!IMPORTANT]  
> I (@Paweł) suggest avoiding to many input formats, especially insecure ones - even at the expense of slight user incovenience - for instance, it's not needed to support pickled numpy, which is insecure and only exists in old inference for convenience. It's also not essential to load stringified base64 image from `form-multipart` requests, as more performant options are available - shortly speaking - support only what makes qualitative difference and be opinionated.

### Ways of requesting inference

#### Pure query params

```bash
curl -X POST https://serverless.roboflow.com/v2/models/infer \
    --data-urlencode 'model_id=whatever/model-id/we?can?figure-out' \
    --data-urlencode 'image=https://images.com/my-super-awesome-image.jpg' \
    --data-urlencode 'confidence=0.3' \
    -H "Authorization: Bearer <your-api-key>"
```

> [!IMPORTANT]  
> There is security issue **embedded into accepting URLs as inputs - especially on the platform.** We accepted the risk of being middle-man in DDoS attack so far, likely it is going to be the case in the future (for user convenience), but would be good for all of parties involved into discussion to recognize and acknowledge this risk - to avoid surprises in the future.

_Batched version_

```bash
curl -X POST https://serverless.roboflow.com/v2/models/infer \
    --data-urlencode 'model_id=whatever/model-id/we?can?figure-out' \
    --data-urlencode 'image=https://images.com/my-super-awesome-image.jpg' \
    --data-urlencode 'image=https://images.com/my-other-awesome-image.jpg' \
    --data-urlencode 'confidence=0.3' \
    -H "Authorization: Bearer <your-api-key>"
```


#### Query params and image bytes

```bash
curl -X POST https://serverless.roboflow.com/v2/models/infer \
    --data-urlencode 'model_id=whatever/model-id/we?can?figure-out' \
    --data-urlencode 'confidence=0.3' \
    -F "image=@photo.jpg;type=image/jpeg" \
    -H "Authorization: Bearer <your-api-key>"
```

_Batched version_
```bash
curl -X POST https://serverless.roboflow.com/v2/models/infer \
    --data-urlencode 'model_id=whatever/model-id/we?can?figure-out' \
    --data-urlencode 'confidence=0.3' \
    -F "image=@photo-1.jpg;type=image/jpeg" \
    -F "image=@photo-2.jpg;type=image/jpeg" \
    -H "Authorization: Bearer <your-api-key>"
```
> [!WARNING]  
> We should not support multiple sources for batch-oriented data (like query and form delivering `image`) as that creates ambiguity regarding the order.

#### KV parameters in form-data
```bash
curl -X POST https://serverless.roboflow.com/v2/models/infer \
    --data-urlencode 'model_id=whatever/model-id/we?can?figure-out' \
    -F "image=@photo.jpg;type=image/jpeg" \
    -F 'inputs={"confidence": 0.5};type=application/json'
    -H "Authorization: Bearer <your-api-key>"
```
> [!WARNING]  
> We must agree on some name for parameters `inputs` is good candidate. We shall also decide do we allow for batch-oriented parameters there - essentially this is JSON with base64 encoded images as in `Content-Type: application/json` scenario - I (@Paweł) would resign from that being supported.


#### `Content-Type: application/json` scenario
```bash
curl -X POST https://serverless.roboflow.com/v2/models/infer \
    --data-urlencode 'model_id=whatever/model-id/we?can?figure-out' \
    -H "Content-Type: application/json" \
      -d '{
        "inputs": {
          "image": {"type": "base64",  "value": "iVBORw0KGgo..."},
          "confidence": 0.5,
        }
      }'
```

### Response formats
Regarding responses from the endpoints, we should aim to achieve the following goals:
* unification with Workflows endpoints (following principle of model looking like single-block Workflow)
* maintaining readibility and understandability
* leaving space for execution-level metadata, without mixing them into the prediction content itself (let's say some time in the future we want to add usage structured data - currently execution time flag is passed - but that is not sustainable when we change billing strategy)
* _(maybe?)_ typed entities for easier parsing - would help us building clients for different languages in automated way
* _(maybe?)_ allowing different formants for certaoin types of responses where it makes sense (for example - performance vs easy to use tradeoff)
* _(maybe?)_ allowing different types of responses response representation - for some models the user should decide between _JSON with base64-encoded data (for ease of use)_ vs _multipart response with our properiatary way of decoding (for speed)_

### Envelope
Since flat response formats lead to mix of actual predictions with metadata, the following strategy for response envelope is proposed - below you see whole response, certain details may differ regarding type of the model.
```json
{
    "type": "roboflow-inference-server-response-v1",
    "model_info": { # metadata describing model used, to be decided },
    "usage": { # metadata describing model used, to be decided },
    "predictions": [ # here image-wise, dicts of specific model results, as proposed below]
}
```

#### Types of predictions
The following types of predictions are in use at the moment (and those are formats proposed to be used in API v2). List of output types cannot be fully defined at the moment (given the evolution of models and representation types), but we should have clear set of responses we know are used in this moment - which we should version and evolve over time. It should be possible to add non-breaking changes to the formats over time, as well as add new formats for the same types of predictions if needed (maybe it actually makes sense to have formally defined way in the API to specify variant of the output entity).

##### **classification (the classical one - single class)**

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

There are couple of issues with current representation:
* no way to apply confidence threshold - once done and "nothing is predicted" - we cannot return that to a client w/o breaking change - since our contract says `top` to be always provided as string (also `confidence` as `float`) - we may pretend we handle that using some constant out-of-class names value, but that's just hidding design flaw.
* current representation is slow to cunstruct - in cases of dataset with large number of classes - construction of `predictions` takes ages - it's joke, especially when we consider using such classifier as secondary model (applied on top of cropped detections to map classes) - we've witnessed that being bottleneck of models running really fast, making substantial cut to processing FPS
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
    "confidences": [0.6, 0.4],
    "top_classes_ids": [0]
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
    "top": [{ "class_name": "cat", "class_id": 1, "confidence": 0.92 }]
}
```
_Bonus question - how should filtering work here - just discard **top**, or alter **candidates**?_

##### **multi-label classification**

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
    "confidences": [0.6, 0.4],
    "detected_classes_ids": [0]
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
    "top": [{ "class_name": "cat", "class_id": 1, "confidence": 0.92 }]
}
```
Despite common representation, there should be differences in behaviour of multi-label classifiers:
* no `top_n` input parameter
* list of `top` entries to represent selected classes (above confidence threshold)


##### **object detection**

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
    "tracker_id": [0, 1],
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
            "detection_id": "y",
       }
    ]
}
```

> [!NOTE]  
> Similar format could be used to handle oriented bboxes - in that case, it's good to provide both OBB and standard bboxes, treating this format as extension of OD format.

##### **instance segmentation**

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

Proposed **_compact_** representation:

```json
{
    "type": "roboflow-semantic-segmentation-compact-v1",
    "pixels_scores": [[0.3, 0.4, ...]],  # maybe due to the size, available on demand only?
    "segmentation_map": [], # rle representation - either per class or with class ids encoded: [(0, 50), (3, 12), (0, 8), (7, 20), (3, 5), ...],
    "class_names": ["cat", "dog"],
}
```


Proposed **_rich_** representation:

```json
{
    "type": "roboflow-semantic-segmentation-rich-v1",
    "pixels_scores": [[0.3, 0.4, ...]],
    "segmentation_map": [[]], # just dump of numpy mask (or alternativelly - only compavt format if we want to keep payload size regime)
    "class_names": ["cat", "dog"],
}
```

##### Other types

* Dense values, such as embeddings, similarity measurements or depth-estimations - should be provided back as dense arrays.
* text-only outputs should remain just simple texts
* structured OCR outputs - should be treated as special case of object-detection (additional field beyond class may be needed to differentiate type of object vs ocred content)

#### Responses encoding
Not only response style (`compact` / `rich`) can be selected, but also HTTP response type. We can imagine delivering `Content-Type: application/json` as default (especially for `rich` style), but for performance (to avoid encoding overhead for sparse / dense arrays [and imgaes]), we could send back _multipart_ responses, then - first part, let's say `mainfest` would contain JSON document, which in positions where binary data is expected, contains `$parts.<part-name>`. This would let us delivering arrays w/o JSON encoding overhead - which is going to be very important to maximise performance. 


## GET `/v2/models/interface`

Endpoint to run get model interface. This endpoint should be used by client to discover:

* all parameters influencing predictions, and their accepted formats and defaults
* response formats (different options available, human- / agent- understandable description and schema definition)
* compatibility with the server (flag to tell if current setup is able to run the model)
* to be decided - do we want to provide here (maybe optional) additional information about model - like available model packages (seems resonable when we want to let people select them) - **alternative: additional endpoint regarding model info or rename of endpoint to match semantics of the operation**
