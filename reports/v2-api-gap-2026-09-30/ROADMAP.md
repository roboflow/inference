# V2 API completion roadmap

This roadmap proposes a sequence of focused PRs for completing the model and server portions of the V2 API on `feat/new-model-manager`. It is for Damian and the inference team to review before preparing individual implementation plans. Start with the shared contract, then work through one PR at a time: investigate its open questions, agree its contract and plan, implement, validate, and review before starting the next.

The recommended order puts model-loading controls and input correctness early, gives classification decisions their own PR, and separates compatibility discovery from the larger interface schema work. Binary output and management/prediction separation are distinct follow-ups rather than additions to earlier PRs.

**Workflows on hold:** The new Workflows functionality is not yet included, as clarified by Damian. All six V2 Workflows routes, their route-specific contract decisions, schemas, executable fixtures, execution semantics and live direct-inference parity checks are offloaded to separate roadmap PRs 12 and 13 and are **on hold**. They are not prerequisites or acceptance criteria for PR 01 or the active model/server sequence. Resume their planning only after the new functionality is included and the team explicitly agrees to resume; recheck its contract before implementing adapters. Existing workflow consumers still need protection from regressions in shared code. The original design's shared input/result requirement still constrains the model contract: an equivalent single-step workflow should be able to accept the same runtime inputs and return the same flat list of model results. PR 01 reviews that constraint with illustrative examples; it does not implement workflow routes or run workflow parity tests.

**V2 compatibility decision:** Damian confirmed V2 is not deployed. The active PRs may replace existing V2 paths and request/response behavior directly; no old-V2 aliases, compatibility adapters, migration window or version-selection mechanism are required. Update affected integration tests to assert the agreed contract, investigating failures rather than preserving outdated expectations. This does not waive V1 compatibility, shared-code regression checks or public/private package coordination.

**D3 response shape:** Damian chose a flat `outputs` list for the current model actions: one result per input item, preserving input order and a one-element list for a singleton batch. Do not add `name`/`value` entries, a dictionary keyed by `predictions`/`embeddings`, the draft's `model_results`/`predictions` wrappers, or a `batch` key. Result-specific arrays such as detections and embedding dimensions remain intact. The response example in PR 2277's model document is incorrect and must be replaced during contract reconciliation. Inference-ID placement remains open. A simple single-step workflow must follow the same result shape; general workflow output contracts remain on hold.

## Sources and decision status

The sources are the [gap report](REPORT.md) and [review comment follow-up](REVIEW_COMMENT_FOLLOWUP.md), as recorded at public report commit `45622ba08116babdbd71f1d2504fcc839822c10c`. The implementation audit covers public commit `3d45b8712cc428eb01b714f3609346be7c92acc4` and private plugins commit `18f2c22543b499727093cf180d45e835ba1c911f`. The draft design is [PR 2277](https://github.com/roboflow/inference/pull/2277), inspected at `de634b98bac204c96caa98a15dd7559dded361d5`.

The follow-up records Paweł's guidance from 30 September, including supported directions and unresolved details. It does not establish final team agreement. The ordering, PR boundaries and acceptance conditions below are recommendations. No new contract choice becomes agreed merely by appearing here. Refresh implementation and discussion evidence when preparing each plan; this roadmap is not a new code audit.

## How each PR proceeds

1. Prepare only the next PR's plan using the [repository implementation plan template](../../.github/implementation-plan-template.md). Recheck relevant public/private code and existing clients against current commits. Include a concrete before/after example, compatibility effects and the proposed tests.
2. Resolve that PR's blocking questions. Present the evidence, one recommended choice and the specific input needed. Record the decision and update the authoritative design, schemas and valid example fixtures before substantial implementation. Later PRs retain their own questions; do not require every future detail to be settled now.
3. Obtain maintainer agreement through the repository's existing process where applicable. The contributor shares the plan in `#discuss-inference-release`; this roadmap does not post to Slack or create implementation PRs.
4. Implement the agreed scope in a draft PR, including focused contract tests, documentation and required package/version metadata. Each implementation PR must remain usable with the dependencies it declares. An unsupported capability must be explicit rather than silently ignored.
5. Complete the relevant checks and review, record evidence and remaining limitations, and update roadmap status. Then prepare the next plan. If investigation changes a boundary or dependency, revise the roadmap before expanding implementation scope.

PR 01 establishes the shared conventions that would otherwise cause repeated rework. Subsequent PRs refine only their own contracts within those conventions. If a later decision changes an earlier shared contract, record and review that amendment first. Decide the canonical home of the evolving design in PR 01, since the proposal currently lives on a separate draft branch.

All implementation PRs should target `feat/new-model-manager` in the public repository unless the team changes the integration strategy. PR 03 belongs in `inference-closed-plugins`; choose its target from that repository's current integration policy during planning. The audit found no private branch named `feat/new-model-manager`.

## Make each plan easy to decide

Use plain language and concrete examples in every PR plan. Lead with what changes for the caller or contributor, then show the choice the reviewer needs to make. Keep the repository plan template's problem, before/after behavior, evidence, recommendation and open questions, but avoid making readers decode abstract terminology first.

- For each blocking decision, show a small example of the proposed behavior and ask a direct question about it. Explain only the trade-off needed to answer that question.
- Use JSON and HTTP request/response examples for API choices. Include an invalid request and its expected error when validation or conflicting inputs are the issue.
- Use a short Pydantic model or dataclass when fields, defaults, optional values or a Python interface are easier to understand in code. Label sketches as illustrative; they do not select an implementation or replace the agreed wire contract.
- Use Mermaid for routing, request handling or control-flow choices. Show the affected path and meaningful branches rather than a generic project process diagram. Skip diagrams that add no information.
- Put current and proposed examples next to each other. Mark recommendations and unresolved details explicitly so a plausible example cannot be mistaken for a settled decision.
- Keep source links close to factual claims. Check JSON/Python example syntax and diagram consistency before publishing. Save exhaustive cases for the later contract tests rather than making the plan a full specification.

These are writing guidelines for the existing scope. Workflows contracts and implementation remain on hold in PRs 12 and 13.

## Recommended sequence

The identifiers below are roadmap identifiers, not GitHub PR numbers. PRs 01–11 and 14 are proposed active-scope items; PRs 12 and 13 are on hold. Dependencies describe technical prerequisites; they are not instructions to start parallel work.

| PR | Scope | Depends on | Reason for this boundary |
|---|---|---|---|
| 01 | Shared model/server contract and executable examples | None | Resolve model/server conventions before behavior changes; Workflows contracts are on hold. |
| 02 | Public model loading and lifecycle controls | 01 | Platform migration needs explicit loading controls early. |
| 03 | Private MMP loading and package selection | 02 | Separate repository and release coordination; required to complete the loading feature. |
| 04 | Input validation, transports and V1 safeguard parity | 01 | Fix reproduced parser defects and make all input paths obey one policy. |
| 05 | Model execution envelope and detection output | 01, 04 | Establish common output behavior with one concrete family. |
| 06 | Classification scores and threshold decisions | 05 | Classification semantics need dedicated review and tests. |
| 07 | Architecture compatibility discovery | 02, 03 | Replace the stub without waiting for the full interface catalogue. |
| 08 | Segmentation and dense JSON representations | 05 | Resolve shape and mask semantics independently of binary encoding. |
| 09 | Text and structured OCR representations | 05 | Text and OCR have distinct structure and batching questions. |
| 10 | Model interface schemas and discovery consistency | 03, 05, 06, 08, 09 | Publish accurate schemas for the established representations and loading behavior. |
| 11 | Binary multipart output | 08, 09, 10 | Add transport efficiency after output semantics are settled. |
| 12 | **On hold —** Workflow metadata, interface and validation endpoints | New Workflows functionality included, explicit resumption, then 10 | Own the workflow discovery/validation contracts and implementation in a separate PR. |
| 13 | **On hold —** Workflow execution and direct inference parity | New Workflows functionality included, explicit resumption, then 04, 11, 12 | Own workflow execution contracts and parity in a separate PR. |
| 14 | Server metadata and Prometheus metrics | 01 | Complete operations endpoints independently of model representations. |

The active serial order is 01–11, then 14; skip held PRs 12 and 13. This is not a deadline or effort estimate. PR 14 can move earlier if deployment requires it and does not wait for Workflows. The management/prediction separation follow-up below is intentionally outside the initial completion sequence pending agreement on release scope.

## PR scope and acceptance

### PR 01 Shared V2 contract

Establish the model/server route/method inventory (six model and four server routes), direct replacement of undeployed experimental routes, common error and response conventions, flat result-list structure and input/result alignment, request control parameters, rich/compact defaults, inference-ID semantics, and schema/type version policy. Define the discovery schema structure and extension points now; populate concrete family definitions with later PRs. Correct invalid JSON and inconsistent mask examples before using them as fixtures. Workflow identifiers, workflow discovery/validation, skipped-output semantics, step tracing and live direct/workflow parity tests remain in the held PRs. PR 01 must nevertheless check that its proposed model input/result structure can also describe an equivalent single-step workflow; do not finalize a model-only structure that contradicts the original design.

Use the proposed `/run`, `/loaded` and DELETE unload paths, following Damian's updated D2 direction. Replace the existing V2 routes directly and adjust affected integration tests; no experimental V2 migration work is needed. Loading/error-state listing is a PR 02 planning question and may be deferred further. Resolve public health/readiness, control-plane defaults, and model-metadata authorization policy. Decide the common placement of effective parameters and the treatment of absent optional metadata. Classification field naming and mask encoding remain decisions for PRs 06 and 08; PR 01 should identify any shared naming constraints they must respect.

**Done when:** the agreed common contract has valid request/response/error examples and schema checks, the documented pre-deployment V2 replacement policy, and explicit unsupported/deferred features. This is a contract/documentation PR; passing its fixture checks does not claim the server conforms yet. Usage metadata must have an agreed integration boundary with the separate usage workstream, without pulling that implementation into this PR.

### PR 02 Public loading and lifecycle controls

Expose the agreed loading controls through explicit loading and automatic loading, and carry package selection through the public gateway and manager path. Align listing and unloading with PR 01. Inventory supported loader options rather than assuming `device`, `instance` and `model_package_id` are the complete set.

Resolve parameter validation, precedence, defaults, package/instance/cache identity, warm reuse versus reconfiguration, load status/errors, and partial unload-all failure behavior. Listing or filtering models in loading/error states is optional scope to assess during PR 02 planning; that plan may postpone it to a later PR. This does not defer correct reporting of load-operation errors or the underlying lifecycle state handling. Define what happens when an older plugin lacks the new capability. Specify direct and MMP gateway contracts together, even though the private implementation follows separately.

**Done when:** direct-backend checks demonstrate that distinct package/options requests select the intended configuration on cold and warm paths; lifecycle tests cover states and failures. Existing plugins retain a deliberate compatibility path or receive an explicit unsupported-capability error. No accepted option may be silently discarded. Full MMP support is a required PR 03 dependency for declaring this feature complete.

### PR 03 Private loading support

Implement the PR 02 contract in `rf-serving`, including MMP request propagation, worker loading and result/error mapping. Coordinate public package versions, private dependency pins and CI source pins so the tested pair matches the deliverable.

Resolve protocol/version compatibility and rollout/rollback order before changing the private transport. Keep binary HTTP responses out of this PR; MMP shared-memory transport is a different layer.

**Done when:** both gateway contract suites pass for the same declared interface, and real MMP tests verify package/option selection, reuse, failure and unload behavior with representative model loads. A coherent public/private install is verified. Mock-only HTTP observations cannot close this item.

### PR 04 Input correctness and safeguards

Fix non-object JSON returning 500 and malformed multipart `inputs` being ignored. Add JSON URL images and basic named multipart references such as `"$part.frame"`, sharing the established input normalization and guarded fetching paths.

Resolve malformed-input statuses, duplicate/missing references, parameter precedence and batch/image limits. Compare against the **current V1 safeguard baseline at planning time**, including destinations, redirects, timeouts, byte budgets, decoded-image dimensions and relevant configuration. Document intentional differences; the existing V2 fetcher alone does not establish parity.

**Done when:** equivalent query/JSON/multipart inputs normalize consistently, invalid structures fail before inference, and adversarial URL/size cases meet the agreed V1 baseline. Tests use controlled fixtures. Nested part traversal, per-item parameter overrides and additional safeguards beyond the agreed baseline remain separate extensions.

### PR 05 Common execution responses and detection

Apply the agreed execution route, envelope, output identity, batch structure and rich/compact default. Consume reserved HTTP controls instead of forwarding them to model methods. Reassess `requested_output` during planning: the flat model result list has no named output slots, so its model-side meaning or deferral needs an explicit decision; do not implement filtering based on the discarded named-output proposal. Provide the detection adapter as the first complete family, including class metadata and agreed effective-threshold metadata.

Resolve whether model output selection is needed, inference-ID scope, available versus missing class names, optional tracker/detection IDs, and the exact effective-parameter fields. The recommendation is to emit tracking metadata only when available, with omission/null behavior settled in this plan; this does not add tracking computation. Decide explicitly whether IoU and max detections join confidence thresholds.

**Done when:** valid detection requests satisfy the new contract across batch/style cases and any explicitly agreed output-selection behavior, with affected V2 integration expectations updated to the agreed behavior. Coordinate unfinished family changes explicitly as the sequence lands; there is no requirement to preserve their old V2 representation for clients. Unsupported multipart output must not be accepted and silently returned as JSON while PR 11 is pending.

### PR 06 Classification semantics

Implement the supported separation between raw scores and threshold-based decisions for single-label and multi-label classification. Add genuine rich multi-label output, effective thresholds and available class names, using the common response contract.

Resolve single-label empty results, threshold boundary/ties, scalar versus class-specific thresholds, defaults and candidate preservation. Decide classification `predicted`/`detected` naming and singular/plural array fields explicitly: the follow-up reopens these questions despite the draft's examples. Top-N remains a separate optional feature unless needed to define the agreed decision behavior.

**Done when:** raw candidates remain available under the agreed policy, selected results and effective thresholds are consistent in both styles, and real model outputs as well as synthetic boundary cases verify the adapter. Any changes to shared model behavior must account for V1/workflow consumers rather than unintentionally changing them.

### PR 07 Compatibility discovery

Replace the 501 stub with the agreed architecture compatibility contract, keeping it distinct from loaded-model inventory and detailed model interface discovery.

Resolve what compatibility means for installed dependencies, device/runtime capabilities, available packages and permissions. State whether compatibility is static support or demonstrated runtime readiness, and how uncertainty is represented. It must not imply that every compatible model has been loaded or executed.

**Done when:** direct and MMP configurations report capabilities consistently, unsupported/unknown combinations have documented semantics, and discovery does not require loading all models. Interface representation filters and the complete type catalogue stay in PR 10.

### PR 08 Segmentation and dense representations

Align instance/semantic segmentation, embeddings and depth with the agreed JSON output contract. Define mask representation and class metadata, dense tensor dimensions/dtypes, and rich/compact behavior where meaningful. Keep binary transport in PR 11.

Resolve cropped RLE versus other mask encoding, offsets, image/crop dimensions and consistent array lengths. Treat existing H×W semantic class/confidence maps as the **recommended baseline to confirm**, not an agreed decision or a reason to introduce C×H×W scores. Resolve type names and raw-array fallback/versioning before publishing schemas.

**Done when:** masks reconstruct correctly in round-trip tests, dense shapes and metadata satisfy schemas, and representative actual model results validate numerical/shape assumptions. Full class-score tensors remain a separate opt-in proposal if justified by a use case.

### PR 09 Text and OCR representations

Define text-only values and structured OCR inside the common envelope, including region/text relationships, output naming, style support and batching. Preserve existing text-producing model actions while adapting their HTTP output.

Resolve string versus typed text representation, OCR detection fields and optional region text, empty results and batch placement. Avoid importing a second incompatible batch convention from the current OCR wrapper.

**Done when:** text and OCR examples validate and real representative outputs preserve content, region alignment and batch order. Adding new OCR/model actions remains outside this PR.

### PR 10 Interface discovery

Implement the shared representation catalogue, `$ref`/definitions, control parameters, request formats and input/output schemas. Make loaded and unloaded discovery conform to the same contract, including the `image`/`images` mapping, agreed filters and model-metadata authorization.

Resolve concrete-model versus task-level action discovery without fabricating model capabilities or requiring model weights to be loaded. Audit all registered families, including VLM, keypoints, interactive segmentation, gaze and passthrough: each needs an accurate supported representation or an explicit limitation. If that audit identifies substantial new family behavior, extract a focused follow-up rather than inventing it inside discovery.

**Done when:** advertised inputs/outputs match executable examples, supported filters work, and loading does not silently change schema semantics or introduce falsely advertised actions. Check both backends and permissions. Do not advertise binary transport as available before PR 11; extend discovery atomically with that implementation.

### PR 11 Binary multipart output

Implement HTTP multipart responses using the agreed JSON response part and named binary data parts, for dense model outputs. Future workflow reuse will be assessed in the held workflow PRs; it is not an acceptance requirement here. Extend discovery and documentation at the same time.

Resolve part references, MIME types, binary dtype/shape/byte-order encoding, content negotiation, errors and resource limits. Keep nested part traversal and alternative dense encodings out unless the core format requires them.

**Done when:** a representative client decodes responses and round-trips JSON/multipart equivalents for masks, depth and embeddings; any agreed output-selection behavior works consistently in both formats; malformed and oversized payload handling is covered. Record payload/memory behavior on representative sizes without claiming a performance improvement from encoding alone.

### PR 12 Workflow discovery and validation on hold

**Status: On hold.** Start a separate contract/implementation plan after the new Workflows functionality is included and resumption is explicitly agreed. The scope below is provisional and must be rechecked then; it does not block the model/server work.

Add the five non-execution V2 workflow endpoints: interface, validation, blocks, definition schema and engine versions. Reuse the type catalogue, authentication and response/error conventions while retaining the existing execution engine semantics.

Resolve inline/predefined definition handling, workflow input/output discovery and structural validation boundaries. Optional runtime-readiness validation is a separate extension unless explicitly required in this plan.

**Done when:** fixtures verify all five endpoints with the workflows extra installed, including invalid definitions and authorization failures. Interface responses accurately describe workflow I/O even while V2 execution is pending. Changes to shared engine behavior, if discovered to be necessary, require explicit replanning and package compatibility/version review.

### PR 13 Workflow execution on hold

**Status: On hold.** Plan this separately after the new Workflows functionality is included, resumption is explicitly agreed and PR 12 establishes its interface. The scope below is provisional; detailed workflow batching, null positions, step IDs, usage and live parity checks are not settled by the model contract. It must honor the shared input/result requirement considered in PR 01.

Add V2 workflow execution using shared request/response codecs and the established workflow interface. Cover inline and predefined workflows, named outputs, batch ordering, null positions and short-circuit behavior.

Resolve how model invocation metadata and usage connect to workflow responses, and exactly which single-step workflows must match direct inference. Preserve scalar-versus-batch semantics; do not smuggle per-item parameter overrides into this work.

**Done when:** equivalent direct and single-step workflow executions compare successfully across the agreed transports/styles, with real representative models and workflow execution. Exercise multi-output, empty/null and short-circuit cases. Coordinate with the separate usage and legacy-parity workstreams; this PR does not replace their release checks.

### PR 14 Server observability

Complete server info and Prometheus-compatible metrics while retaining the health/readiness policy settled in PR 01. This can move earlier if required for deployment.

Resolve safe build/configuration fields, metric names/types/labels and cardinality, content negotiation, default access, and readiness meaning with/without preloads and backend failures.

**Done when:** direct/MMP fixtures verify health/readiness transitions and safe info output, and a Prometheus parser/scrape check validates the metrics contract. Comprehensive monitoring dashboards and changes to usage accounting remain independent.

## Explicit follow-ups and exclusions

| Work | Recommendation and condition for returning to it |
|---|---|
| Workflows routes and parity | **On hold in separate PRs 12 and 13.** Resume only when the new functionality is included and the team explicitly agrees; revisit contracts and acceptance criteria first. |
| Management/prediction separation | Keep as a separate follow-up PR after loading behavior is stable. Before initial release, explicitly confirm whether separation can wait. Its plan must settle ports/process boundaries, credentials, automatic-loading policy and prediction behavior when a model is absent. The control-plane gate is not equivalent to manual-loading-only operation. |
| Per-item batch parameters with shared defaults | Omit from the initial contract provisionally, reflecting the tentative guidance. Revisit with a concrete use case or comparative evidence and an explicit workflow scalar/batch design. |
| Full C×H×W semantic scores | No implicit expansion from the H×W baseline. Require an opt-in use case, size/resource assessment and a separate contract decision. |
| V1 exposure of new loading controls | Separate migration decision and PR if required; V2 loading controls are not deferred with it. |
| Top-N classification, nested part traversal and optional workflow runtime-readiness validation | Separate extensions after the basic contracts work, unless a specific requirement makes one essential. |
| New tracking, model actions, video streaming, pipeline migration and usage implementation | Preserve relevant integration boundaries, but keep these in their own workstreams. Legacy parity remains a separate release obligation. |

## Coverage against the reports

| Finding | Roadmap coverage |
|---|---|
| Route/method differences, defaults and envelope G1 | Contract 01; lifecycle 02; execution 05 |
| JSON URL/named-part inputs G2 and both parser defects | 04 |
| Binary output G3 and reserved parameter forwarding | Reserved controls 05; multipart 11 |
| Package selection G4 and full loading controls | Public 02 and required private 03 |
| Output selection G5 | Reassess model meaning or deferral in 01/05 after the flat-list decision; no assumed named model slots. Any agreed behavior extends to 11; workflow behavior in 13 is on hold. |
| Interface discovery G6 | Structure 01; family schemas 05/06/08/09; discovery 10 |
| Compatibility stub and loaded/compatible distinction | Lifecycle 02; compatibility 07 |
| Prediction representation gaps | Detection 05; classification 06; segmentation/dense 08; text/OCR 09; remaining-family inventory 10 |
| Workflow endpoints and parity G7 | **On hold:** five non-execution routes and contracts in 12; execution contract/parity in 13 |
| Server info/metrics, public probes and control-plane policy | Policy 01; observability 14 |
| Safeguard parity and effective parameters from review | Safeguards 04; detection metadata 05; classification thresholds 06 |
| Open naming, optional IDs, tensor shape and separation | Naming 01/06/08; IDs 05; shape 08; explicitly deferred separation above |

## Completion and next step

Each PR carries its own tests; validation does not wait until the final PR. Before declaring the active model/server scope complete, run its contract suite against both supported backends using a coherent public/private package pair, exercise representative real model families, verify SDK/client decoding, and cover required CPU/GPU and deployment configurations. Workflow execution and direct/workflow parity are acceptance gates only for the held PRs after resumption. Completing the active scope does not mean the full proposed V2 Workflows surface is complete. Record any untested configurations and explicitly deferred capabilities in the release scope. The earlier 318 passing tests and 51 observations are audit evidence, not acceptance of the future contract.

The next deliverable is **the plan for PR 01 only**. It should recommend shared choices with concrete examples, identify the decisions requiring input, and establish where the authoritative contract will live. This roadmap does not start that plan or implement any API change.
