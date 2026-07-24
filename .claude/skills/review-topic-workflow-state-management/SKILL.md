---
name: review-topic-workflow-state-management
description: Load when a diff in inference/core/workflows/core_steps/** or prototypes/block.py seeds cross-run() state in a WorkflowBlock __init__ (self._trackers, self._sessions, self._previous_positions, self.cache, dicts/deques/sets), keys state by video_metadata.video_identifier / session / api_key, edits trackers/_base.py InstanceCache, touches analytics/time_in_zone|velocity or flow_control/delta_filter, or adds/edits get_restrictions() / STATEFUL_VIDEO_HTTP_SOFT_RESTRICTION / a remote guard.
---

# Review topic: Workflow / block state management

## When this applies
Load this skill when a diff in `inference/core/workflows/core_steps/**` (or `inference/core/workflows/prototypes/block.py`) does any of:
- Seeds instance state in a `WorkflowBlock.__init__` that persists across `run()` calls (`self._trackers`, `self._per_video_cache`, `self._sessions`, `self._previous_positions`, `self.cache`, any `Dict`/`deque`/`set`/`list`).
- Keys state by `image.video_metadata.video_identifier`, session id, or API key; or tracks objects across frames (bytetrack, botsort, sort, ocsort under `core_steps/trackers/`).
- Touches temporal-analytics blocks: `analytics/time_in_zone`, `analytics/line_counter`, `analytics/velocity`, `analytics/path_deviation`, `transformations/stabilize_detections`, `flow_control/delta_filter`.
- Manages TTL / eviction / cooldown / reattach windows, or reads/writes `inference/core/cache` (`BaseCache`, `MemoryCache`) with `expire=`, or `deque(maxlen=...)` / FIFO / LRU buffers in a block or sink.
- Adds or edits `get_restrictions()`, `STATEFUL_VIDEO_HTTP_SOFT_RESTRICTION`, `COOLDOWN_HTTP_SOFT_RESTRICTION`, or a `StepExecutionMode` / `NotImplementedError` remote guard.

Paths are hints; the trigger is the *behaviour* — a block holding state that must outlive a single `run()` yet must not silently outlive its scope or cross a request/shard boundary.

## Review checklist

### BLOCK — must be fixed before merge
- [ ] **Remote-execution contract.** A block holding cross-frame or cross-request state MUST either return a SOFT restriction from `get_restrictions()` (e.g. `STATEFUL_VIDEO_HTTP_SOFT_RESTRICTION` / `COOLDOWN_HTTP_SOFT_RESTRICTION`) OR hard-fail: `run()` raises `NotImplementedError` when `step_execution_mode is not StepExecutionMode.LOCAL`. A stateful block with **neither** produces silently-wrong tracks/counts/cooldowns on hosted-serverless (frames land on different workers).
- [ ] **Keying / isolation.** Per-video/per-session state must be keyed by a real discriminator (`video_identifier`, session id, API-key hash), guarded with `setdefault(video_id, ...)` / `if video_id not in self._x`. A shared or empty-default namespace cross-contaminates tracks/counts/dwell-times and can leak one tenant's data into another's output (cross-session bleed regressed in #632).

### FLAG — reviewer should raise
- [ ] **Bounded state.** Every mutable `__init__` attribute that grows per-video or per-track needs a bound: `deque(maxlen=...)`, an LRU / `WithFixedSizeCache`, capacity + FIFO/LRU eviction, or `cache.set(..., expire=...)`. A bare `Dict`/`set`/`list` keyed by `tracker_id` (unbounded — new ids arrive forever) is a memory leak on a long-lived server. Per-`video_identifier` dicts are bounded by distinct videos (softer, but still flag on a hosted process).
- [ ] **Eviction correctness.** A bounded cache must evict from *all* backing structures. `InstanceCache` pops `_cache_inserts_track` (the deque) AND `.remove()`s from `_cache` (the set) — dropping one leaks the other (the model-manager analogue regressed in #1212 / #1526).
- [ ] **State cleanup on invalidation.** When config the state depends on changes (e.g. model switch), stale state must be cleared, not reused against a mismatched config — e.g. `self._sessions.clear()` in the SAM2-video block.
- [ ] **Cache-key TTL & namespacing.** A cache key holding a transient (last-report-time, cooldown, workspace lookup) must set `expire=` OR be namespaced by workflow/step/aggregator id so it can't grow unbounded or collide across workflows/tenants. A durable-looking `cache.set(key, value)` with no `expire=` on transient data is an unbounded key.
- [ ] **New stateful transformation blocks.** Treat brand-new stateful transformation blocks with extra scrutiny — the tracklet-recognition transformation was reverted (#2527 reverting #2497).

### NIT — optional
- [ ] **Window consistency.** TTL / reattach / lost-track-buffer / cooldown windows should be internally consistent — a reattach window shorter than the eviction interval silently drops tracks that should survive.
- [ ] **Metadata fallbacks.** Missing `fps` / `video_identifier` should degrade gracefully (default + warn), never crash the pipeline nor collapse all streams under one empty identifier.
- [ ] **Required stateful manifest fields.** Fields that seed state (zones, lines) should not carry `default=None` — a silent default yields empty state instead of a clear error (#658 removed `default=None` from line-counter / time-in-zone manifests).

### Not blocking
- A block keyed only by `video_identifier` with a plain `Dict` (e.g. `delta_filter`'s `self.cache`) is acceptable for a minimal per-stream skeleton — do NOT demand a `deque`/LRU when the key space is distinct videos, not per-track ids.
- Do NOT demand a hard `NotImplementedError` guard when the block already declares a SOFT restriction via `get_restrictions()` — SOFT (degraded output) and HARD (fail-closed) are alternatives, not both.
- Do NOT flag missing eviction on state that is flushed each `run()` (e.g. the model-monitoring `PredictionsAggregator` resets `_raw_predictions` on `get_and_flush`).
- Do NOT require Redis-offloaded state for a block explicitly scoped to `InferencePipeline` / local execution and honestly declaring its remote restriction.

## Standards

- **State must declare its remote/serverless contract.** In-memory state cannot cross a request or shard boundary; on stateless / multi-replica HTTP runtimes successive frames may hit different workers, so in-memory tracking/counting/cooldown is meaningless. The block declares this via `get_restrictions()` (SOFT: degraded output, surfaces at compile) or fails closed by raising `NotImplementedError` when `step_execution_mode is not StepExecutionMode.LOCAL` (HARD: surfaces at first frame). The `RuntimeRestriction` / `Severity` / `Runtime` framework in `inference/core/workflows/prototypes/block.py` exists to make this declarable — new stateful blocks must opt in.
- **State must be scoped and isolated.** Per-video/per-session state is keyed by a real discriminator and guarded before first use; nothing is shared under an anonymous/default namespace.
- **State must be bounded and evicted.** A block instance can live for the whole process. Anything keyed by `tracker_id` is effectively unbounded over a long stream and needs a cap + eviction that touches every backing structure; `maxlen`/size clamped `>= 1`.
- **Transient cache keys carry a TTL and a namespace.** `expire=` on transient values, namespaced keys so two workflows/tenants can't collide.
- **State is cleared when its config is invalidated**, and window parameters (TTL / reattach / cooldown) are internally consistent.

## Key files (canonical patterns)

- `inference/core/workflows/core_steps/trackers/_base.py` — canonical bounded per-video state. `TrackerBlockBase` keys `self._trackers` / `self._per_video_cache` by `video_id`; `InstanceCache` is a FIFO `deque(maxlen)` (`_cache_inserts_track`) + `set` (`_cache`) with correct dual-structure eviction in `_cache_new_tracker_id` and a `size = max(1, size)` clamp. New trackers subclass this instead of re-rolling state (#642, later #705).
- `inference/core/workflows/core_steps/transformations/byte_tracker/v3.py` — shipped per-video tracker: `self._trackers` + `InstanceCache`, `get_restrictions()` returning `STATEFUL_VIDEO_HTTP_SOFT_RESTRICTION` + `STILL_IMAGE_INPUT_SOFT_RESTRICTION`.
- `inference/core/workflows/core_steps/models/foundation/segment_anything2_video/v1.py` — the **fail-closed** reference: `run()` raises `NotImplementedError` when `self._step_execution_mode is not StepExecutionMode.LOCAL`, keys `self._sessions` by `video_id` via `setdefault`, and `self._sessions.clear()`s on model switch. Copy this for any block whose state genuinely cannot survive remote execution.
- `inference/core/workflows/core_steps/analytics/time_in_zone/v3.py` (and `v2.py`) — per-video zone state keyed by `video_identifier`, requires `tracker_id` (raises if absent). The tracked-ids map is **unbounded** — cite as the shape to bound in new work, not as a bound reference.
- `inference/core/workflows/core_steps/analytics/velocity/v1.py` — `self._previous_positions` / `self._smoothed_velocities` are `setdefault(video_id, {})` dicts that are **unbounded per track**; the anti-pattern new analytics code copies. Cite when reviewing a new analytics block with no bound.
- `inference/core/workflows/core_steps/flow_control/delta_filter/v1.py` — smallest correct stateful skeleton: plain `self.cache` keyed by `video_identifier` + `STATEFUL_VIDEO_HTTP_SOFT_RESTRICTION`.
- `inference/core/workflows/prototypes/block.py` — the contract source: `Severity`, `Runtime`, `StepExecutionMode`, `RuntimeRestriction`, and the `STATEFUL_VIDEO_HTTP_SOFT_RESTRICTION` / `COOLDOWN_HTTP_SOFT_RESTRICTION` / `STILL_IMAGE_INPUT_SOFT_RESTRICTION` presets. (`common/entities.py` re-exports `StepExecutionMode` for back-compat.)
- `inference/core/workflows/core_steps/sinks/roboflow/model_monitoring_inference_aggregator/v1.py` — **mixed** example, read carefully. Good: `get_restrictions()` declares a SOFT restriction; the workspace-lookup key in `get_workspace_name` uses `cache.set(..., expire=900)` (15-min TTL) — the correct pattern for a durable/TTL'd cross-frame value. Anti-pattern: `_last_report_time_cache_key` (`workflows:steps_cache:...:{unique_aggregator_key}:last_report_time`) is written with `cache.set(key, value)` and **no `expire=`** — a per-aggregator-key cache entry with no eviction. Do NOT cite `last_report_time` as the TTL example; it is the "transient key missing `expire=`" anti-pattern this rule catches.

## Reference PRs
- #642 / #705 — bounded `InstanceCache` for tracker seen-ids sets.
- #658 — removed `default=None` from line-counter / time-in-zone manifests.
- #632 — kept distinct exec sessions for InferencePipeline usage tracking (cross-session bleed).
- #1212 / #1526 — fixed-size-cache eviction bugs in the model manager (same class as block-level FIFO).
- #2527 (reverting #2497) — reverted the tracklet-recognition transformation block.
- #2387 — graceful fallback on ephemeral cache failure (state must not assume the cache is durable on serverless).
