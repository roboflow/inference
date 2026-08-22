# Review skill dispatch INDEX

Canonical dispatch map for the PR-review skill suite. Consumers read THIS file
instead of carrying their own copy of the tables:

- **CI** — `.github/prompts/claude-pr-review.md` (Skill Dispatch section).
- **Local review tooling** — any agent or script reviewing this repo can
  resolve skills through this file from its own checkout.

Editing rule: this file is the single source of truth for path→skill and
signal→skill routing and for the shared review contract below. Do not restate
the tables anywhere else; skills keep only their own trigger description.

This file is repo-opinionated: it encodes the project's review standards and
nothing else. Reviewer-personal tooling, preferences, and overlays live outside
this repo and reference this file — never the other way around.

## 1. Surface skills — dispatch by changed path

Load every surface skill whose paths the PR touches:

| Changed path (glob) | Surface skill |
| --- | --- |
| `inference/core/workflows/core_steps/**` | `review-workflows-blocks` |
| `inference/core/workflows/execution_engine/**`, other `inference/core/workflows/**` | `review-workflows-execution-engine` |
| `inference_models/**` | `review-inference-models-pkg` |
| `inference/models/**`, `inference/core/models/**`, `inference/core/registries/**` | `review-legacy-models-registries` |
| `inference/core/interfaces/http/**`, `inference_cli/server.py` | `review-http-api-server` |
| `inference_sdk/**` | `review-sdk` |
| `inference_cli/**` (except `server.py`) | `review-cli-cloud-tooling` |
| `docker/**`, `.github/**`, `requirements/**`, `.release/**`, `Makefile`, repo-root `setup.py`/`pyproject.toml`/`mkdocs.yml`, `build_scripts/**`, `app_bundles/**` | `review-packaging-ci` |
| other `inference/core/**` (env, version, entities, utils, roboflow_api, exceptions) | `review-core-infra` |

A changed **test** file dispatches to the same surface skill as the product code
it exercises (`tests/workflows/**` → the workflow skills;
`tests/inference/**/http/**` → `review-http-api-server`;
`tests/inference_models/**` → `review-inference-models-pkg`;
`tests/inference_sdk/**` → `review-sdk`) — in addition to
`review-topic-test-hygiene`.

## 2. Topic skills — dispatch by contribution signal

Independently of path, load each topic skill whose signal the PR exhibits
(confirm via the skill's `description`):

| If the PR… | Topic skill |
| --- | --- |
| holds per-video / per-session / per-user state, trackers, caches across frames, TTL/reattach | `review-topic-workflow-state-management` |
| changes local vs remote / hosted / serverless execution, or backend / runtime routing | `review-topic-local-vs-remote-execution` |
| touches boxes / masks / keypoints / coordinate transforms / pre- or post-processing / serialization | `review-topic-prediction-integrity` |
| changes any public contract **or the user-visible behavior of one** — HTTP route/entity/behavior, SDK, CLI, workflow block schema / a new `vN.py` block version / kinds / loader registration, compiled-workflow format, `inference_models` API, persisted/cache format — or is release-bound | `review-topic-backward-compat-and-versioning` |
| adds threads / async / background work, touches caches / locks / model lifecycle / temp dirs, or long-running resources | `review-topic-concurrency-and-resource-safety` |
| calls an external / platform API, changes an SDK↔server contract, or adds fallback / auto-conversion | `review-topic-external-contract-and-silent-fallback` |
| touches auth / api-key / workspace-tenant scoping / permissions / secrets | `review-topic-auth-and-tenant-security` |
| ingests external / user input — a URL / file path / uploaded image, `torch.load` / pickle / weights load, or zip/tar extraction (SSRF, path traversal, unsafe deserialization, decompression bombs) | `review-topic-input-boundary-security` |
| adds/modifies an outbound HTTP call, builds a URL from `API_BASE_URL` / `HOSTED_*_URL` / a `*.roboflow.com` host, adds an endpoint-URL env var or setting, constructs an `InferenceHTTPClient`, or touches `wrap_url` / `SECURE_GATEWAY` | `review-topic-secure-gateway-url-wrapping` |
| **(every substantive PR)** — verify changed behavior is covered by a real CI test, tests are isolated, selectors exercised | `review-topic-test-hygiene` |

## 3. Surfaces with NO dedicated skill — visibility rule

These surfaces have no surface skill. Apply the generic review plus matching
topic skills:

- `inference/core/interfaces/{stream*,camera,udp,webrtc_worker,sam3_video_session}/**` (streaming pipeline)
- `inference/core/{active_learning,cache,managers}/**`
- `inference/usage_tracking/**`
- `inference/enterprise/**`
- `modal/**`, `development/**`, `signatures/**`
- `docs/**`, `examples/**`

When a PR touches one of these, the review summary MUST state:
"No dedicated surface skill covers `<path>` — generic review plus topic skills
only." Never let the coverage gap pass silently.

## 4. Shared review contract

Applies whenever one or more skills load.

**Severity vocabulary.** Skills use BLOCK / FLAG / NIT; reported findings use
Critical / High / Medium. Mapping:

- BLOCK → **Critical** (likely production breakage, data loss, or security
  exposure) or **High** (significant bug or contract break under realistic
  usage). Fix before merge.
- FLAG → **Medium** — meaningful risk worth addressing before merge.
- NIT → **never posted as a finding.** At most one collapsed one-line footnote
  across the whole review, or dropped entirely.

**Multi-skill arbitration.** The surface skill owns severity calibration for
its surface; the topic skill owns the deep rule text. If a surface and a topic
skill both flag the same defect, post ONE finding citing the topic skill's
rule. De-duplicate all findings across loaded skills before posting.

**Version-bump carve-out (canonical statement).** Contributors never select or
bump versions; maintainers own release PRs (version constants, lock-step pins,
final changelog headings). From a feature/fix contributor require only the
user-facing entry under `## Unreleased` in the relevant changelog.
