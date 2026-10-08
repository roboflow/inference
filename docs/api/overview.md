# Inference conceptual API

This initial section covers action recognition in video Workflows. Other
Inference capabilities are outside this document's current scope.

## Value objects

- **Video frame:** an image and its position within one source stream.
- **Analysis window:** the latest analysis state, reported classes and the
  actual sampled coverage in that stream.
- **Action interval:** a class over an inclusive range of source frames.
- **Timeline:** accumulated action intervals; overlapping evidence may merge.

## Operations

| Operation | Input | Output |
| --- | --- | --- |
| `actionRecognition.analyze` | Video frame, model, class filter, analysis cadence | Analysis window, timeline, current-call error |
| `actionRecognition.visualize` | Matching video frame and analysis window, display mode, optional timeline | Annotated video frame |

## Invariants and effects

- The model owns sampling, analysis state and accumulated history per stream.
- Window classes describe all reported actions in that window, not the action
  at the displayed frame. Confidence invented for compatibility is not evidence.
- Collection, a successful empty result, and failure are distinct states.
  A failed call invalidates the displayed result until a successful call.
- Visualization changes pixels only. It neither runs analysis nor retains
  history, buffers replay, creates event identities, or counts movements.
- Timeline rendering clips existing history to the analyzed coverage. It does
  not extend an interval into frames with no result.
- Frame/FPS time is approximate. An assumed FPS cannot be presented as a
  declared source clock. Sampling coverage is not a latency measurement.

Action Recognition Model v2 and Action Recognition Visualization v1 implement
these operations. Model v1 retains its original timeline and error outputs.
See [connections and value-object fields](../workflows/action_recognition_visualization.md)
for their concrete Workflow interface.
