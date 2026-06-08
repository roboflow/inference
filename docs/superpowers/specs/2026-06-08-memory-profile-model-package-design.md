# MemoryProfile on ModelPackageMetadata — Design

Date: 2026-06-08
Package: `inference_models` (inner package `inference_models/inference_models/`)

## Goal

Attach GPU-memory measurements to per-model-package metadata so consumers can
reason about a package's VRAM footprint at a given batch size. Numbers are
produced offline (model_eval / conversion side — out of scope here) and served
by the Roboflow model-metadata API; this work only adds the field, its parsing,
and a lookup helper on the inference-models side.

## Scope

In scope:
- New `MemoryProfile` type carrying peak VRAM per batch size.
- New optional `memory_profile` field on `ModelPackageMetadata`.
- Parsing the field off the Roboflow API package payload.
- A `vram_for_batch()` lookup helper.
- Unit + parser tests.

Out of scope (explicitly deferred):
- Device/GPU keying — assume cross-device difference is small for now.
- Latency or allocated-vs-reserved memory split — single peak VRAM number only.
- Production of the measurements (model_eval / conversion pipeline).

## Why ModelPackageMetadata (granularity)

`ModelMetadata` (`weights_providers/entities.py:177`) holds one `model_id` and a
**list** of `ModelPackageMetadata`. A `ModelPackageMetadata`
(`weights_providers/entities.py:117`) is one concrete deployable: a specific
`backend` (TRT/ONNX/torch), `quantization`, and batch config. GPU footprint is a
function of exactly those package attributes — the same model served as TRT-int8
vs torch-fp16 uses different VRAM. So the measurement is only meaningful at
**package** granularity. `ModelMetadata` would be too coarse.

This also matches the existing pattern: `ModelPackageMetadata` already hosts
backend-detail sub-objects (`trt_package_details`, `onnx_package_details`,
`torch_script_package_details`) and a measurement-from-eval object
(`recommended_parameters`). `memory_profile` is the VRAM analog of
`recommended_parameters`.

## Data model

Each per-batch entry is a single integer: peak VRAM in MB. Keys are batch sizes.
Static-batch packages carry a single key; dynamic-batch packages carry several.
The same `Dict[int, int]` represents both — no separate structure needed.

Keys are **not assumed contiguous**. Measurement density (dense `1..N` vs sparse
`1,2,4,8,16,32`) is a measurement-production decision, not a schema constraint.
Consumers round **up** to the nearest measured key (ceiling), which slightly
overestimates VRAM on sparse keys — safe for packing/eviction.

### New type — `weights_providers/entities.py` (beside `RecommendedParameters`, ~line 114)

`MemoryProfile` is a pydantic `BaseModel` (not a frozen dataclass) so it
auto-parses from API JSON exactly like `RecommendedParameters`, including
coercing string JSON object keys (`"1"`, `"2"`) to `int`. A dataclass would force
manual construction in every backend parser.

```python
class MemoryProfile(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="ignore",
        populate_by_name=True,
        alias_generator=to_camel,
    )

    peak_vram_mb: Optional[Dict[int, int]] = None  # batch_size -> peak VRAM (MB)

    def vram_for_batch(self, batch_size: int) -> Optional[int]:
        """Peak VRAM (MB) for `batch_size`.

        Ceiling lookup: returns the value for the smallest measured batch size
        >= `batch_size`. If `batch_size` exceeds the largest measured key, clamp
        to the largest key. Returns None if no measurements are present.
        """
        if not self.peak_vram_mb:
            return None
        keys = sorted(self.peak_vram_mb)
        for k in keys:
            if k >= batch_size:
                return self.peak_vram_mb[k]
        return self.peak_vram_mb[keys[-1]]
```

### Host field — `ModelPackageMetadata` (`weights_providers/entities.py:117`)

New optional field, default `None` → existing callers and stored payloads stay
valid (backward compatible):

```python
memory_profile: Optional[MemoryProfile] = field(default=None)
```

## Population path

Mirrors `recommended_parameters`, which sits on the package wrapper
`RoboflowModelPackageV1` (not inside the per-backend manifest) and is threaded
unchanged into `ModelPackageMetadata` by each backend parser.

1. `RoboflowModelPackageV1` (`weights_providers/roboflow.py:66`) — add:
   ```python
   memory_profile: Optional[MemoryProfile] = Field(
       alias="memoryProfile", default=None
   )
   ```
2. Each per-backend parser that builds a `ModelPackageMetadata`
   (`parse_onnx_model_package` and the other ~6, all in `roboflow.py`) — add one
   line to the constructor call:
   ```python
   memory_profile=metadata.memory_profile,
   ```

No changes to `get_roboflow_model` / `ModelMetadata` — the field lives entirely
at package level.

## Testing

- `MemoryProfile.vram_for_batch`:
  - exact key hit
  - ceiling between two measured keys
  - request above largest key → clamp to largest
  - empty dict / `None` → returns `None`
- Parser test: a package payload with `memoryProfile` JSON parses into a
  populated `memory_profile` (string keys coerced to int).
- Back-compat test: a package payload without `memoryProfile` yields
  `memory_profile is None`.

## Backward compatibility

All new fields are optional with `None` defaults. Old API payloads (no
`memoryProfile`) parse unchanged; existing `ModelPackageMetadata` constructions
need no edits beyond the threading line.
