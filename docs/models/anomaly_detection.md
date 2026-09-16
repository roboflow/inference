# FoundAD and PatchCore

Anomaly projects store image labels as `normal` and `anomalous`. Both models learn normal
appearance, and their saved threshold is calibrated with validation images. They use the
classification model route and existing classification Workflow block.

Install the `roboflow-anomaly` wheel built from `roboflow-train/train/anomaly_runtime` alongside
this local inference checkout. The `anomaly` extra pins the runtime version for release, but that
wheel is not published yet. The runtime contains the pinned official implementations and the
single train/serve preprocessing, artifact, and scoring contract.

```python
from inference import get_model

model = get_model("workspace/anomaly-model-t1", api_key="YOUR_API_KEY")
result = model.infer("image.jpg", include_anomaly_map=True)[0]
print(result.top, result.anomaly_score, result.anomaly_threshold)
```

`predictions`, `top`, and `confidence` retain the classification response shape. Additional fields:

- `anomaly_score`: raw image anomaly score; larger means more anomalous.
- `anomaly_threshold`: the saved validation threshold.
- `is_anomalous`: whether the raw score meets or exceeds that threshold.
- `anomaly_map`: optional raw local anomaly evidence resized to the input image dimensions.

Confidence is a monotonic score transformation, not an estimated defect probability. A generic
classification confidence argument does not recalibrate the decision. Scores are independent of
which other images share a request. Heatmaps are evidence, not supervised segmentation masks.
Both normal and anomalous predictions are retained, including at the decision threshold.

Artifacts include all fitted backbone and model state; loading does not download pretrained
weights. PatchCore supports CPU inference; FoundAD benefits from a GPU. The CPU/GPU server image
must install the runtime wheel before these architectures are enabled.

Schema 2 platform artifacts also record version-export preprocessing. The runtime reproduces the
platform's AREA resize and JPEG encoding before feature extraction; optional online materialization
is recorded separately. Send original images to inference. Training marks exported images as already
materialized, so preprocessing is applied exactly once. Schema 1 artifacts retain their original
preprocessing; retrain through the platform to obtain the corrected contract rather than changing
their saved threshold.
