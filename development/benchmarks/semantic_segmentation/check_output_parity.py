"""Output-parity check for the semantic segmentation post-processing PRs.

Asserts that the optimized `_convert_to_sv_detections` (present_class_ids
hint, single F-order conversion, optional in-process numpy masks) produces
outputs byte-identical to the pre-optimization implementation, which is
inlined here verbatim as the reference. Checked per detection:

* class_id order and values, class_name
* xyxy (exact)
* RLE counts strings (byte equality) and RLE size fields
* confidence (exact after the float32 storage cast the reference also gets)
* confidence_mask payload (array byte equality)

Input combinations exercised against the reference (base64/no-hint):

* base64 masks, no hint          (decode path, np.unique scan)
* base64 masks, hint             (decode path, hinted scan)
* numpy masks, hint              (fast path; auto-skipped when the tree
                                  lacks response_mask_format=numpy support)
* numpy masks, no hint           (fast path, np.unique fallback)

plus edge cases (all-background map, background-only hint, stale hint id,
hint-with-empty-list fallback) and, when torch is available, the
`present_class_ids_from_label_map` helper against `np.unique` ground truth.

By default the main comparison runs on a full-scale synthetic label map
matching the 16.1 MP workload that motivated the PRs (~30 s); use --quick
for a small map, or --label-map/--confidence-map for real data.

    PYTHONPATH=. python development/benchmarks/semantic_segmentation/check_output_parity.py

Exits non-zero on any mismatch.
"""

import argparse
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from benchmark_postprocessing import (
    CONVERT,
    baseline_convert_to_sv_detections,
    detect_numpy_mask_support,
    encode_mask_png_b64,
    load_grayscale,
    synthetic_confidence_map,
    synthetic_label_map,
)

FAILURES: List[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  PASS {label}")
    else:
        print(f"  FAIL {label} {detail}")
        FAILURES.append(f"{label} {detail}".strip())


def detections_match(label: str, reference: Any, candidate: Any) -> None:
    check(f"{label}: detection count", len(reference) == len(candidate))
    if len(reference) != len(candidate):
        return
    if len(reference) == 0:
        return
    check(
        f"{label}: class_id",
        reference.class_id.tolist() == candidate.class_id.tolist(),
        f"{reference.class_id.tolist()} vs {candidate.class_id.tolist()}",
    )
    check(
        f"{label}: class_name",
        reference.data["class_name"].tolist() == candidate.data["class_name"].tolist(),
    )
    check(
        f"{label}: xyxy exact",
        reference.xyxy.dtype == candidate.xyxy.dtype
        and np.array_equal(reference.xyxy, candidate.xyxy),
    )
    check(
        f"{label}: confidence exact (float32)",
        reference.confidence.dtype == candidate.confidence.dtype
        and np.array_equal(reference.confidence, candidate.confidence),
    )
    ref_rles = reference.data["rle_mask"]
    cand_rles = candidate.data["rle_mask"]
    counts_equal = all(r["counts"] == c["counts"] for r, c in zip(ref_rles, cand_rles))
    sizes_equal = all(r["size"] == c["size"] for r, c in zip(ref_rles, cand_rles))
    check(f"{label}: RLE counts byte-identical", counts_equal)
    check(f"{label}: RLE size fields", sizes_equal)
    ref_conf_mask = reference.data.get("confidence_mask")
    cand_conf_mask = candidate.data.get("confidence_mask")
    if ref_conf_mask is None and cand_conf_mask is None:
        check(f"{label}: confidence_mask (both absent)", True)
    else:
        check(
            f"{label}: confidence_mask byte-identical",
            ref_conf_mask is not None
            and cand_conf_mask is not None
            and np.array_equal(np.asarray(ref_conf_mask), np.asarray(cand_conf_mask)),
        )


def payload(
    seg: np.ndarray,
    conf: Optional[np.ndarray],
    class_map: Dict[str, str],
    transport: str,
    hint: Optional[List[int]] = None,
) -> Dict[str, Any]:
    if transport == "base64":
        body: Dict[str, Any] = {
            "segmentation_mask": encode_mask_png_b64(seg),
            "class_map": class_map,
        }
        if conf is not None:
            body["confidence_mask"] = encode_mask_png_b64(conf)
    else:
        body = {"segmentation_mask": seg, "class_map": class_map}
        if conf is not None:
            body["confidence_mask"] = conf
    if hint is not None:
        body["present_class_ids"] = hint
    return body


def build_maps(
    args: argparse.Namespace,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, str]]:
    if args.label_map:
        seg = load_grayscale(args.label_map)
    else:
        height, width = (303, 532) if args.quick else (3032, 5320)
        seg = synthetic_label_map(height, width, args.foreground_classes, args.seed)
    if args.confidence_map:
        conf = load_grayscale(args.confidence_map)
    else:
        conf = synthetic_confidence_map(seg, args.seed + 1)
    class_map = {
        str(cid): f"class_{cid}" for cid in np.unique(seg).tolist() if cid != 0
    }
    return seg, conf, class_map


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="small map (~0.2 MP)")
    parser.add_argument("--foreground-classes", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label-map", help="grayscale PNG or .npy uint8 label map")
    parser.add_argument("--confidence-map", help="grayscale PNG or .npy uint8 map")
    args = parser.parse_args()

    seg, conf, class_map = build_maps(args)
    height, width = seg.shape
    print(
        f"label map {width}x{height} ({height * width / 1e6:.2f} MP), "
        f"classes {np.unique(seg).tolist()}"
    )
    numpy_supported = detect_numpy_mask_support()
    print(f"numpy mask support in this tree: {numpy_supported}")
    true_hint = [int(v) for v in np.unique(seg)]

    print("\nreference (pre-optimization algorithm, base64 transport):")
    reference = baseline_convert_to_sv_detections(
        payload(seg, conf, class_map, "base64")
    )
    check("reference produced detections", len(reference) > 0)

    print("\nproduction, base64 masks, no hint:")
    detections_match(
        "base64/no-hint",
        reference,
        CONVERT(payload(seg, conf, class_map, "base64")),
    )
    print("\nproduction, base64 masks, hint:")
    detections_match(
        "base64/hint",
        reference,
        CONVERT(payload(seg, conf, class_map, "base64", hint=true_hint)),
    )
    if numpy_supported:
        print("\nproduction, numpy masks, hint:")
        detections_match(
            "numpy/hint",
            reference,
            CONVERT(payload(seg, conf, class_map, "numpy", hint=true_hint)),
        )
        print("\nproduction, numpy masks, no hint:")
        detections_match(
            "numpy/no-hint",
            reference,
            CONVERT(payload(seg, conf, class_map, "numpy")),
        )
    else:
        print("\nnumpy transport checks skipped (tree lacks support)")

    print("\nedge cases (small maps):")
    empty = np.zeros((64, 64), dtype=np.uint8)
    check(
        "all-background map -> empty detections",
        len(CONVERT(payload(empty, None, {}, "base64"))) == 0,
    )
    check(
        "background-only hint -> empty detections",
        len(CONVERT(payload(empty, None, {}, "base64", hint=[0]))) == 0,
    )
    small = np.zeros((64, 64), dtype=np.uint8)
    small[8:24, 8:24] = 3
    stale = CONVERT(
        payload(small, None, {"3": "a", "9": "ghost"}, "base64", hint=[0, 3, 9])
    )
    check(
        "stale hint id skipped, real class kept",
        len(stale) == 1 and stale.class_id.tolist() == [3],
    )
    empty_hint = CONVERT(payload(small, None, {"3": "a"}, "base64", hint=[]))
    check(
        "empty hint list falls back to scanning",
        len(empty_hint) == 1 and empty_hint.class_id.tolist() == [3],
    )

    try:
        import torch

        from inference.core.models.semantic_segmentation_utils import (
            present_class_ids_from_label_map,
        )

        print("\npresent_class_ids_from_label_map vs np.unique:")
        for name, array in (
            ("main map", seg),
            ("all-background", empty),
            ("id 255", np.full((32, 32), 255, dtype=np.uint8)),
        ):
            helper_ids = present_class_ids_from_label_map(torch.from_numpy(array))
            check(
                f"helper matches np.unique ({name})",
                helper_ids == [int(v) for v in np.unique(array)],
            )
    except ImportError:
        print("\ntorch unavailable - helper check skipped")

    print()
    if FAILURES:
        print(f"PARITY FAILURES ({len(FAILURES)}):")
        for failure in FAILURES:
            print(f"  - {failure}")
        return 1
    print("ALL PARITY CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
