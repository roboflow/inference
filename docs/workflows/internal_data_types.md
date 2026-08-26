---
template: redirect.html
redirect_url: https://docs.roboflow.com/workflows/developer-guide/developer-guide/data-representations
---

## VideoSegmentClassification

`VideoSegmentClassification` represents one classified frame range in a
video. It stores `start_frame_idx`, `end_frame_idx`, `class_name`, and
`class_id`. Serialization changes `class_name` to `class`.

A range that contains the last classified frame is provisional. Its end frame
advances with the stream until a later classification window closes it.
