# V2 API design review comment follow-up

This companion to the [gap report](REPORT.md) checks whether Damian Kosowski's review comments on [PR #2277](https://github.com/roboflow/inference/pull/2277) were incorporated into subsequent design commits. Several ideas were incorporated, particularly the classification changes, while other questions remain unanswered.

## Scope and chronology

The three reviews were submitted on 27–28 April 2026. The subsequent design changes landed on 19 May 2026. This assessment compares the original design at `a7ef0df05` with the current PR head, `de634b98bac204c96caa98a15dd7559dded361d5`, including these relevant commits:

- [efa47b82e — Add chenges to model inference endpoint](https://github.com/roboflow/inference/commit/efa47b82e6a6a9001f3493c2f0550a07e72a0c3d): classification decisions, thresholds, request formats and response conventions.
- [11315f692 — Formalise ideas regarding interface discovery for models](https://github.com/roboflow/inference/commit/11315f6925850a45ff151107882ca71e4c6e7392): DELETE unload, explicit loaded-model route and interface discovery.
- [de634b98b — Polish the content with Claude](https://github.com/roboflow/inference/commit/de634b98bac204c96caa98a15dd7559dded361d5): further explanation and example changes, including multi-label thresholds and dense-output transport.

On 30 September 2026, all 14 inline review threads were still unresolved, 13 were marked outdated, and none contained a reply. GitHub's outdated marker records a changed diff location; it does not establish that the concern was addressed. The assessments below are based on the subsequent diffs and final document content, not inferred author intent.

## Comment assessment

| Review comment | Assessment | What changed or remains |
|---|---|---|
| [Use DELETE for unloading](https://github.com/roboflow/inference/pull/2277#discussion_r3146553921) | **Incorporated** | `11315f692` changes the operation to `DELETE /v2/models/unload`, covering one or all models. |
| [Clarify loaded versus compatible models](https://github.com/roboflow/inference/pull/2277#discussion_r3146611532) | **Addressed differently** | `11315f692` introduces `/models/loaded` and clarifies that compatibility lists architectures. The proposed `?state=` approach was not adopted. |
| [Separate management from prediction; disable automatic loading](https://github.com/roboflow/inference/pull/2277#discussion_r3146641940) | **Still open** | The design specifies neither a separate-port policy nor an explicit automatic-management switch. |
| [Per-item batch parameters plus shared defaults](https://github.com/roboflow/inference/pull/2277#discussion_r3146674058) | **Still open** | Named multipart references were added, but no per-item parameter/defaults contract. Examples still use one confidence value for multiple images. |
| [Expose all relevant model-loading parameters](https://github.com/roboflow/inference/pull/2277#discussion_r3146711145) | **Still open** | `model_package_id` becomes explicit, but full control over loading parameters is not specified. General model-input discovery does not settle this concern. |
| [URL safeguards, timeouts and image-size checks](https://github.com/roboflow/inference/pull/2277#discussion_r3146716276) | **Not documented** | The original security discussion was removed; the safeguards endorsed in the comment were not written into the revised contract. This was an endorsement of safeguards, rather than a request for a particular code change. |
| [Separate classification scores from threshold decisions](https://github.com/roboflow/inference/pull/2277#discussion_r3148829157) | **Incorporated** | `efa47b82e` replaces `top_classes_ids` with `predicted_class_ids` and adds `confidence_threshold`. |
| [Keep candidates; filter only the decision](https://github.com/roboflow/inference/pull/2277#discussion_r3148958925) | **Incorporated in examples** | Candidates retain below-threshold scores, while `predicted_classes` contains selected classes. An explicit normative statement would still help. |
| [Use “predicted,” not “detected,” for classification](https://github.com/roboflow/inference/pull/2277#discussion_r3148967856) | **Incorporated** | Multi-label uses `predicted_class_ids`/`predicted_classes`; the final revision includes thresholds in both styles. |
| [Clarify the purpose of rich representation](https://github.com/roboflow/inference/pull/2277#discussion_r3150595842) | **Clarified** | Later prose defines rich as readable and self-describing, versus compact for smaller payloads. This does not adopt the separate debugging-metadata proposal below. |
| [Plural `class_ids` and `confidences`](https://github.com/roboflow/inference/pull/2277#discussion_r3150602768) | **Not adopted** | Detection still uses singular `class_id` and `confidence` for arrays. Classification was also changed from `confidences` to `confidence`. |
| [Include effective thresholds and NMS settings in rich output](https://github.com/roboflow/inference/pull/2277#discussion_r3150719048) | **Still open** | Detection responses still omit `confidence_threshold`, `iou_threshold`, and `max_detections`. This is the one thread not marked outdated. |
| [Is `tracker_id` optional?](https://github.com/roboflow/inference/pull/2277#discussion_r3150719841) | **Still open** | Examples include it, but optionality is not specified. |
| [Avoid enormous C×H×W pixel-score responses](https://github.com/roboflow/inference/pull/2277#discussion_r3150798500) | **Partially addressed** | Final prose describes per-pixel class confidence and binary transport, acknowledging JSON size costs. It does not explicitly define tensor dimensions or move full class-score output into an optional workflow. |

The final [API structure](https://github.com/roboflow/inference/blob/de634b98bac204c96caa98a15dd7559dded361d5/design/00_inference_api_v2/01-general-api-structure.md) and [model endpoint specification](https://github.com/roboflow/inference/blob/de634b98bac204c96caa98a15dd7559dded361d5/design/00_inference_api_v2/02-models-endpoints.md) are the pinned sources for the current-design assessments above.

## Suggested follow-up

The unload and three classification threads are candidates for resolution after reviewer confirmation. The remaining threads need agreement on the alternative design, confirmation that the clarification answers the question, or an explicit decision on the outstanding concern. This document does not change any GitHub thread state.

Incorporation into the design does not establish implementation in `feat/new-model-manager`. The [gap report](REPORT.md) separately records implementation gaps, including DELETE unload and classification response fields that remain absent from the audited implementation.
