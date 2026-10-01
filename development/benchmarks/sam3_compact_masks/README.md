# SAM3 compact mask follow-up

This change builds on the combined #3073 + #3098 staging branch (`5ba0902dc`).
A SAM3 frame with hundreds of instances currently allocates one full-resolution
boolean mask per instance even when the server returns compressed RLE. The new
path converts the RLE directly to Supervision's existing `CompactMask` type.

## Adoption and compatibility

For a SAM3 v3 step, set:

```json
{
  "type": "roboflow_core/sam3@v3",
  "name": "sam3",
  "images": "$inputs.image",
  "class_names": ["pill bottle"],
  "output_format": "rle",
  "use_compact_masks": true
}
```

The default is false. Existing workflows continue returning dense NumPy masks.
The option applies to RLE output on the NumPy block's SDK, local, and proxy paths;
it does not change requests, polygon output, or tensor-native representations.
The tensor sibling accepts the field so definitions remain portable.

Enable only after checking downstream consumers. There is no automatic graph
inspection or conversion before unknown custom blocks. A consumer requiring a
NumPy array can explicitly use `np.asarray(predictions.mask)`, paying the dense
allocation cost at that boundary. Do not enable this globally in the worker.

- For sliced workflows, also set `use_compact_masks: true` on Detections Stitch.
  It preserves compact output only with this opt-in and when every non-empty
  mask input is compact. Default, dense, or mixed inputs retain dense output.
- NMS and unfiltered stitching remain compact. Supervision 0.30.6 NMM still
  densifies individual merge candidates internally; this change does not fix it.
- Mask visualization already supports compact crops; polygon visualization now
  reads them directly, retaining the existing bounding-box slicing semantics.
- RLE serialization retains the original response payload and does not densify.
  Consumers that serialize polygons or explicitly materialize masks may still
  allocate dense arrays.

Only `roboflow-workflows` implementation changes are required. No new dependency,
GPU server, or frontend change is needed. Worker image adoption and staging
workflow opt-in are separate rollout steps. Maintainer review of the opt-in
contract and staging validation remain necessary before enabling it broadly.

## Validation

The tests cover default and opt-in execution, real SDK request concurrency and
ordering with mocked HTTP, local/proxy routing, tensor portability, mask and RLE
serialization parity, empty masks, holes, disconnected regions, frame edges,
rendering parity, and stitch NMS/NMM/no-filter behavior including clipping and
mixed dense/compact inputs. Allocation guards reject full-frame decoding on the
compact conversion/render/NMS path.

Run the synthetic benchmark from the repository root in an environment with
repository dependencies:

```sh
PYTHONPATH=.:workflows:inference_models python development/benchmarks/sam3_compact_masks/benchmark.py
```

It creates 1080p masks containing small rectangles with holes, verifies every
mask, alternates dense/compact conversion order, discards a warm-up, and reports
three timings per mode. Dense allocations can exceed 2 GiB. No model, GPU, or
network is used. This is a mask-conversion benchmark, not end-to-end preview FPS.

Local Apple Silicon, Python 3.11.15, NumPy 2.4.6, Supervision 0.30.6:

| Instances | Dense median | Compact median | Conversion speedup |
| --- | ---: | ---: | ---: |
| 100 | 192.8 ms | 17.1 ms | 11.29x |
| 500 | 945.1 ms | 84.0 ms | 11.25x |

The 500-mask dense stack alone occupies 1,036,800,000 bytes. Performance depends
on mask complexity and crop coverage. These synthetic results do not establish
production latency or an 8-GiB worker's safe concurrency. Compare the same video
with both prerequisite PRs deployed, using identical worker limits, workflow,
and output boundaries before making a rollout performance claim.
