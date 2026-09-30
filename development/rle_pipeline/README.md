# SAM3 preview mask pipeline

The local SAM3 v3 tensor block requests `mask_format="rle"` from the model's
existing chunked postprocessor. Per-class thresholds, cross-prompt mask NMS,
tight boxes, and prediction packing operate on those RLEs. The model still
computes dense masks internally; the change avoids returning a full dense mask
stack to the workflow and encoding it again there. Model RLE strings are
normalized to the native carrier's byte counts without decoding pixels.

Stitch placement translates foreground runs directly into parent coordinates.
Masks with a large compressed representation use a vectorized fallback that
materializes one source slice at a time, never the full-frame instance stack.
This is deliberate: pure Python run iteration is slower for noise-like masks.
Identity placement reuses the compressed bytes.

Visualization clips compressed foreground runs directly to the detection crop.
The tensor polygon v2 annotator reads `CompactMask.crop(index)` and offsets the
contours. It must not index `CompactMask[index]`, which reconstructs a full-frame
mask even though the container itself is compact. Drawing order, colors,
thickness, holes, clipped boxes, and inclusive mask box coordinates are retained.
The non-tensor and polygon v1 annotators retain their existing behavior.

## Local measurements

Measured on the development Mac with three repeats per operation, against
`72b5c5407`. Input masks were reconstructed from saved polygon predictions for
Gustavo's 1920x1012 video. These are real prediction shapes, but not raw SAM3
logits or the exact pre-stitch model outputs. Placement uses a 640x640 carrier
containing each eligible prediction. The benchmark excludes corpus preparation.

| Frame | Instances | Placement before / after | Crop conversion before / after | Polygon drawing before / after |
| --- | ---: | ---: | ---: | ---: |
| 0 | 494 | 1055 / 18 ms | 687 / 24 ms | 237 / 11 ms |
| 240 | 572 | 1231 / 20 ms | 739 / 28 ms | 308 / 12 ms |
| 571 | 512 | 1106 / 20 ms | 694 / 27 ms | 246 / 12 ms |

Decoded crop masks, placement RLE bytes, and rendered image pixels match the
baseline exactly. These component results are not a prediction of GPU FPS.
The cost of the model's RLE encoder and final video throughput still need a
staging measurement with the optimized worker image.

The reproducible benchmark lives on `rafel/v1.6.0-stitch-compact`. Check out that
branch and run it with both repository packages on `PYTHONPATH` and its
Supervision dependency installed:

```sh
PYTHONPATH="$PWD:$PWD/inference_models" python development/rle_pipeline/benchmark.py \
  --results /path/to/saved/results.json --frames 0 240 571 --repeats 3
```

No customer video, predictions, credentials, or weights are included in the
repository. `--baseline` selects the Git revision for the old implementations.

## Verification

The unit tests compare against dense masks and Supervision's original polygon
annotator. They cover fragmented/all-zero/all-one masks, clipping, source and
canvas edges, translated runs spanning columns, NMS ties, per-prompt thresholds,
class mappings, bytes/string RLE counts, empty detections, and PIL inputs.
Guards prohibit pixel decoding in the compressed placement path, mask
encoding/decoding in the native RLE packer, and full-frame indexing in the new
polygon annotator.

`test_sam3_postprocessor_rle.py` additionally compares the real pinned SAM3
chunked postprocessor's dense and RLE outputs through workflow filtering and
packing. It requires the optional SAM3 package; it runs on CPU without weights.
CUDA model execution, batching, and deployment are outside these local checks.
