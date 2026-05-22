
# `detections_overlaps` Kind

List of per-pair detection overlap records

## Data representation



### External

External data representation is relevant for Workflows clients - it dictates what is the input and output format of
data.

Type: `List[Dict[str, Any]]`

### Internal

Internal data representation is relevant for Workflows blocks creators - this is the type that will be provided
by Execution Engine in runtime to the block that consumes input of this kind.

Type: `List[Dict[str, Any]]`

## Details


List of per-pair overlap records computed between two sets of detections at
the same dimensionality. Internally and on the wire: `List[Dict[str, Any]]`.

Each record is a dict with the following keys.

**Always present:**

* `reference_class: Optional[str]` —
    `class_name` of the reference-side detection. `None` when the source
    `sv.Detections` does not carry `class_name`.

* `reference_confidence: Optional[float]` —
    Confidence of the reference-side detection. `None` when the source
    `sv.Detections.confidence` is `None`.

* `candidate_class: Optional[str]` —
    `class_name` of the candidate-side detection (same nullability rules).

* `candidate_confidence: Optional[float]` —
    Confidence of the candidate-side detection (same nullability rules).

* `overlap_ratio: float` —
    `intersection_area(reference_polygon, candidate_polygon) / reference_polygon.area`.
    Range `[0.0, 1.0]`. The denominator is *always* the reference detection's
    area, so the relation is not symmetric across the two inputs. Polygons
    come from masks when available (longest contour, validated), otherwise
    from the bounding-box rectangle.

**Conditionally present** (only when the source `sv.Detections.data` carries
`detection_id` for that side):

* `reference_detection_id: Optional[str]`
* `candidate_detection_id: Optional[str]`

Records below the configured `min_overlap` threshold are not emitted. The
top-level list is unordered with respect to pair identity; do not rely on
positional indexing to cross-reference back into the input batches — use
`reference_detection_id` / `candidate_detection_id` instead.


<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>
