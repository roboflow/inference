# TDD sequence: `masks_resolution_factor`

Companion to `instance-seg-tradeoff-factor-v2.md`. Each phase is red → green → commit.

## Existing coverage, measured not assumed

`inference_models/tests/unit_tests/models/common/roboflow/test_post_processing.py` (28 tests) **does**
cover `align_instance_segmentation_results`, via `test_chunked_resize_matches_monolithic` (:502) and
`..._with_static_crop_canvas`. But both are **invariance** tests — they assert chunked output equals
monolithic output. Change the resize target and both sides move together, so **both still pass**.

Nothing pins output *geometry*: not the shape relative to the image, not coordinate correctness.
`crop_masks_to_boxes` is not imported and has no tests at all.

So phase 0 is not optional ceremony — it is the only thing that would catch a geometry regression.

---

## Phase 0 — characterization (GREEN on current code)

Pin today's behaviour before touching anything. These must pass unchanged at `t=1.0` forever after.

```python
class TestCurrentMaskGeometry:
    """Characterization. These encode today's contract; they must not change at t=1.0."""

    def test_masks_are_image_sized(self) -> None:
        _, masks = self._run()
        assert masks.shape[1:] == (self.size_after_pre_processing.height,
                                   self.size_after_pre_processing.width)

    def test_masks_are_bool(self) -> None:
        _, masks = self._run()
        assert masks.dtype == torch.bool

    def test_golden_output_for_fixed_seed(self) -> None:
        # seeded protos -> exact tensor. Any geometry change breaks this loudly.
        torch.manual_seed(0)
        _, masks = self._run()
        assert masks.sum().item() == GOLDEN_SET_PIXELS      # fill from first run
        assert masks[0, :8, :8].tolist() == GOLDEN_CORNER   # fill from first run

    def test_non_square_image_preserves_aspect(self) -> None:
        # the axis-swap guard. Square fixtures hide transposes.
        _, masks = self._run(image=(1080, 1920))
        assert masks.shape[1] == 1080 and masks.shape[2] == 1920
```

And the first tests `crop_masks_to_boxes` has ever had:

```python
class TestCropMasksToBoxes:
    def test_zeroes_outside_box(self) -> None: ...
    def test_scaling_default_is_proto_stride(self) -> None: ...
    def test_non_square_inference_size(self) -> None:
        # documents that a single scalar `scaling` is wrong when mask_w/inference_w != 0.25
        ...
```

**Commit 0.** No production change.

## Phase 1 — RED: the coordinate-scaling gap

The prerequisite from §2 of the plan. This test fails today and proves the gap exists.

```python
def test_polygon_coords_are_image_space_when_mask_is_smaller(self) -> None:
    # given: a mask deliberately smaller than the declared image
    masks = make_masks(n=2, h=540, w=960)          # half of 1080x1920
    meta = make_metadata(original_size=(1080, 1920))

    # when
    response = build_instance_segmentation_response(masks, meta)

    # then: coordinates must span the IMAGE, not the mask
    xs = [p.x for pred in response.predictions for p in pred.points]
    assert max(xs) > 960, "polygon coords left in mask space"
```

RED today — `inference_models_adapters.py:917` emits raw `masks2poly` output. Green once scaling is
applied at the mask→polygon boundary, mirroring legacy `post_process_polygons`
(`inference/core/utils/postprocess.py:449`).

**Commit 1.** Coordinate scaling only. No new parameter yet — and it is a latent-correctness fix on
its own merits.

## Phase 2 — RED: the parameter

```python
@pytest.mark.parametrize("t", [0.0, 0.1, 0.25, 0.5, 1.0])
def test_resolution_factor_interpolates_shape(self, t: float) -> None:
    _, masks = self._run(masks_resolution_factor=t)        # TypeError today -> RED
    mh, mw = PROTO_UNPADDED
    assert masks.shape[1] == max(1, round(mh * (1 - t) + 1080 * t))
    assert masks.shape[2] == max(1, round(mw * (1 - t) + 1920 * t))

def test_t1_is_bit_identical_to_baseline(self) -> None:
    _, ref = self._run()                                  # no kwarg
    _, out = self._run(masks_resolution_factor=1.0)
    assert torch.equal(out, ref)

def test_t0_is_proto_resolution(self) -> None:
    _, masks = self._run(masks_resolution_factor=0.0)
    assert masks.shape[1:] == PROTO_UNPADDED

@pytest.mark.parametrize("t", [0.1, 0.25, 0.5])
def test_coords_round_trip_against_t1(self, t: float) -> None:
    ref = polygons_from(self._run())
    out = polygons_from(self._run(masks_resolution_factor=t))
    # unpad rounding error is amplified as t falls, so tolerance scales with 1/t
    assert_polygons_close(out, ref, tol_px=2.0 / t)

def test_zero_instances(self) -> None:
    _, masks = self._run(n=0, masks_resolution_factor=0.25)
    assert masks.shape[0] == 0

@pytest.mark.parametrize("mode", ["letterbox", "stretch"])
def test_both_resize_modes(self, mode: str) -> None: ...

def test_static_crop_offset_is_scaled(self) -> None: ...
```

The `t=1.0` bit-identity test is the safety net for every later phase.

**Commit 2.** Dense path only.

## Phase 3 — RED: `mask_size` metadata (plan O1)

```python
def test_mask_size_defaults_to_image_size(self) -> None:
    # back-compat: existing construction sites must not change
    m = InstancesRLEMasks.from_coco_rle_masks(image_size=(1080, 1920), masks=[...])
    assert m.mask_size == (1080, 1920)

def test_mask_size_reflects_reduced_resolution(self) -> None:
    dets = self._run_full_pipeline(masks_resolution_factor=0.25)
    assert dets.mask_size == (270, 480)
    assert dets.image_size == (1080, 1920)      # unchanged meaning
```

**Commit 3.**

## Phase 4 — RED: adapter mapping

```python
@pytest.mark.parametrize("mode,expected", [("accurate", 1.0), ("fast", 0.0)])
def test_enum_maps_to_factor(self, mode: str, expected: float) -> None: ...

def test_tradeoff_reads_the_factor(self) -> None: ...

@pytest.mark.parametrize("bad", [-0.1, 1.1])
def test_out_of_range_raises(self, bad: float) -> None:
    with pytest.raises(InvalidMaskDecodeArgument):
        ...

def test_source_keys_do_not_leak_to_model_call(self) -> None:
    # map_inference_kwargs must consume both, not forward them
    mapped = adapter.map_inference_kwargs({"mask_decode_mode": "fast", "tradeoff_factor": 0.0})
    assert "mask_decode_mode" not in mapped and "tradeoff_factor" not in mapped
    assert mapped["masks_resolution_factor"] == 0.0
```

**Commit 4.**

## Phase 5 — RED: version skew (plan O2) and the Triton gate

```python
def test_absent_mask_size_defaults_to_input_size(self) -> None:
    # old server: field missing -> client assumes today's contract
    dets = parse_response(response_without_mask_size())
    assert dets.mask_size == dets.image_size

def test_unhonoured_hint_does_not_raise(self, caplog) -> None:
    # old server ignores the parameter: correct output, just slower
    dets = run_remote(masks_resolution_factor=0.25, server=OldServerStub())
    assert dets.mask_size == dets.image_size
    assert "not honoured" in caplog.text        # logged once, never raised


def test_reduced_resolution_forces_eager_fallback(self) -> None:
    # pure-python: no GPU, so it runs in x86 CI instead of being a dead GPU-only test
    reason = _unsupported_triton_postprocess_reason(..., masks_resolution_factor=0.25)
    assert reason is not None
```

**Commit 5.**

## Phase 6 — RLE parity

Blocked on the deferred wire-format decision. Until then the RLE path keeps `t=1.0` and phase 0's
characterization tests guard it. The evidence experiment (plan O3) belongs here:

```python
@pytest.mark.parametrize("decoder", ["pycocotools", "supervision"])
def test_reduced_size_rle_decode_behaviour(self, decoder: str) -> None:
    # not an assertion of correctness - a recorded observation of the failure mode,
    # to inform the deferred decision: raises? truncates silently? resizes cleanly?
    ...
```

## Running

```bash
cd inference_models && python -m pytest tests/unit_tests/models/common/roboflow/test_post_processing.py -v
python -m pytest tests/inference/models_predictions_tests
python -m pytest tests/workflows/unit_tests/
```

## Order and why

Phase 0 before everything, because the existing invariance tests would not catch a geometry
regression. Phase 1 before phase 2, because coordinate scaling is a correctness fix that stands
alone and must be in place before any reduced mask can exist. Phase 6 last, because it is the only
part gated on a decision the team deferred.
