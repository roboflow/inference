# Tensor-native V2 examples

From the repository root, run the existing environment:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=workflows:inference_models:stream_vision \
  /Users/ppeczek/miniconda3/envs/roboflow-inference-new/bin/python \
  development/workflows-2.0/03-tensor-native/run_demo.py \
  --scenario all --output-dir /tmp/workflows-2.0-tensor-native
```

Open `/tmp/workflows-2.0-tensor-native/index.html`. All generated files stay under
`--output-dir`. These are deterministic offline fixtures, with no model download,
credentials or camera. The prediction block is explicitly synthetic.

| Scenario | Inspect |
| --- | --- |
| `geometry` | Actual compiler → session → nested workflow → output rows. Crop, anisotropic resize, another crop, native detections, confidence selection and a condition. Root/own overlays, IDs, transforms, native dtype/device and serialized rows. |
| `gallery` | All eleven native kinds using real `inference_models` carriers: empty float16 tensors/detections, dense masks, disconnected RLE/semantic regions, keypoints without boxes, both classification forms, detection iterator row, QR/barcodes and embeddings. Every fixture is serialized through output rows and decoded through compiled ingress, with equality checks. |
| `invalid` | HWC and wrong-dtype image tensors, unequal prediction row counts, unknown output coordinates, and ambiguous root restoration of populated/empty composite mosaics. |

The numerical geometry oracle is deliberately asymmetric: first crop offset
`(30,70)`, resize factors `(0.5,0.25)`, nested crop offset `(5,3)`, then local box
`[2,4,6,8]` → root box `[44,98,52,114]`. Both large roots are 300×400 pixels.
The middle rectangle falls outside the image, leaving crop indices `0,2`.
One root is entirely filtered; the small third root has a genuine empty group
and a blank composite mosaic. Requesting root rows does not rewrite the own
payload, even when both outputs select the same native object.

Start with [author_blocks.py](author_blocks.py) and
[the workflow JSON](workflows/nested_geometry.json). An author reads pixels from
`image.tensor_image`, attaches `image.prediction_metadata()` to native model
results, and calls `select_predictions(prediction, mask=...)` to keep aligned
fields together. Scalars and thresholds remain ordinary Python values. The
selection helper rejects a mismatched mask; `indices=[...]` can reorder rows.
It preserves tensor devices and may transfer selected indices for Python row
metadata/RLE, plus synchronize its all-true check. It does not export prediction
tensors to NumPy. PNG rendering and JSON serialization are explicit host
boundaries; decoding allocates on the receiving CPU.

Image transforms use `target_xy = local_xy * scale_xy + offset_xy` with `(x,y)`
vectors and `(height,width)` dimensions. The V1 prediction metadata bridge uses
reciprocal `scaling_relative_to_*` values. Output `coordinates_system="own"`
keeps local predictions; `"root"` and V1-compatible `"parent"` restore workflow
root coordinates. The no-option default is `"parent"`; `"parent"` does not
mean the immediate crop parent. That frame remains inspectable on the image.
A mosaic stores per-tile provenance and has no single contributing-source root.

The opt-in catalogue also handles native values at wildcard boundaries and the
legacy `numpy_array` label used by tensor-native depth blocks. Use `tensor` in
new declarations. These examples use the V2 catalogue explicitly and do not
change the V1 engine selection or block discovery.
