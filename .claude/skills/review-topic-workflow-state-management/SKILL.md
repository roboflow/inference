---
name: review-topic-workflow-state-management
description: Load when a PR adds or changes a Workflow block (or engine path) that holds per-video / per-session / per-user state in process memory — trackers, time-in-zone / line-counter / velocity analytics, streaming SAM sessions, delta/cooldown filters, or sinks with in-memory buffers — or that manages caches, TTLs, or reattach windows across frames. Reviews unbounded state, missing eviction, keying/isolation, and the fail-on-remote contract for stateful blocks.
---

# Review topic: Workflow / block state management

## When this applies
Load this skill when a diff, in `inference/core/workflows/core_steps/**` or an engine path, does any of:
- Adds a `WorkflowBlock` whose `__init__` seeds instance dicts/deques/sets that persist across `run()` calls (`self._trackers`, `self._batch_of_*`, `self._per_video_cache`, `self._sessions`, `self._previous_positions`, `self.cache`).
- Keys state by `image.video_metadata.video_identifier` (or by session/user/API key), or tracks objects across frames (byte_tracker, botsort, sort, ocsort, `trackers/_base.py`).
- Touches temporal-analytics blocks: `analytics/time_in_zone`, `analytics/line_counter`, `analytics/velocity`, `analytics/path_deviation`, `transformations/stabilize_detections`, `flow_control/delta_filter`.
- Manages TTL / eviction / cooldown / reattach windows, or reads/writes `inference/core/cache` (`BaseCache`, `MemoryCache`, `redis`) with `expire=`, or FIFO/LRU/`deque(maxlen=...)` buffers in a block or sink.
- Adds or edits `get_restrictions()`, `STATEFUL_VIDEO_HTTP_SOFT_RESTRICTION`, `COOLDOWN_HTTP_SOFT_RESTRICTION`, or a `StepExecutionMode.REMOTE` / `NotImplementedError` guard.
Paths are hints; the trigger is the *behaviour* — a block holding state that must outlive a single `run()` yet must not silently outlive its scope or cross a request/shard boundary.

## What to protect
- **State must be scoped and isolated.** Per-video/per-session state has to be keyed by a real discriminator (`video_identifier`, session id, API-key hash). Sharing an anonymous/default namespace across videos or tenants cross-contaminates tracks, counts, and dwell times, and can leak one tenant's data into another's output.
- **State must be bounded and evicted.** A block instance can live for the whole process. Any `Dict[video_id, ...]` or `Dict[tracker_id, ...]` that only ever grows is an unbounded memory leak on a long-lived server (one entry per video ever seen, per track ever created). New track ids are effectively unbounded over a long stream.
- **Stateful blocks must be honest about remote/serverless execution.** State cannot cross a request or shard boundary. On stateless / multi-replica HTTP runtimes successive frames may hit different workers, so in-memory tracking/counting/cooldown is meaningless. A block must either declare this via `get_restrictions()` (SOFT: degraded output) or **fail closed** — raise `NotImplementedError` at the top of `run()` when `step_execution_mode is not LOCAL` (HARD: cannot produce a usable result). The failure surfaces at compile/first-frame, not as silently-wrong analytics in production.
- **TTL / reattach / cooldown windows must be internally consistent.** A reattach window shorter than the eviction interval drops tracks that should survive; a cooldown stored in per-process memory does not throttle behind a load balancer.

## What to check
1. **`__init__` state inventory.** List every mutable attribute seeded in `__init__` (dicts, deques, sets, lists, cache handles).
2. For each attribute, ask two questions: is it **keyed** by a discriminator, and is it **bounded**?
3. **Bound check.** A bound is `deque(maxlen=...)`, an LRU/`WithFixedSizeCache`, a capacity + FIFO/LRU eviction, or `cache.set(..., expire=...)`. Flag any bare `Dict`/`set`/`list` that grows per-video or per-track with no eviction path as an unbounded leak.
4. **Keying / isolation.** Confirm state is keyed by `metadata.video_identifier` (or session/tenant), not a shared global. Check `setdefault(video_id, {})` / `if video_id not in self._x` guards exist before first use. Reject state shared across videos or default-keyed under an empty/missing identifier.
5. **Remote-execution contract.** If the block holds cross-frame or cross-request state it MUST do one of: return `STATEFUL_VIDEO_HTTP_SOFT_RESTRICTION` (or `COOLDOWN_HTTP_SOFT_RESTRICTION`) from `get_restrictions()`, OR hard-guard `run()` with `NotImplementedError` when `step_execution_mode is not StepExecutionMode.LOCAL` (SAM2-video pattern).
6. A stateful block with **neither** a declared restriction nor a fail-closed guard is a defect — it produces silently-wrong output on hosted-serverless.
7. **Eviction correctness.** For bounded caches, verify eviction removes from *all* backing structures (e.g. `InstanceCache` pops the deque AND `.remove()`s from the set — dropping one leaks the other) and that `maxlen`/size is clamped `>= 1`.
8. **Window consistency.** Verify TTL / reattach / lost-track-buffer / cooldown windows are internally consistent — a reattach window shorter than the eviction interval silently drops tracks that should survive.
9. **Cache-block scoping.** For `cache_get`/`cache_set` and sinks using `BaseCache`, verify keys are namespaced (workflow/step/aggregator id) so two workflows or tenants can't collide.
10. **Transient TTLs.** Verify `expire=` is set where the cached value is a transient (last-report-time, cooldown, workspace lookup) rather than persisted forever.
11. **State cleanup on invalidation.** When a config the state depends on changes (e.g. model switch), stale state must be cleared (`self._sessions.clear()`), not silently reused against a mismatched config.
12. **Metadata fallbacks.** Missing `fps` / `video_identifier` must degrade gracefully (default + warn), never crash the pipeline nor collapse all streams under a single empty identifier.
13. **Required stateful fields.** Manifest fields that seed state (zones, lines) must not carry `default=None` — a silent default produces empty state instead of a clear error.

## Common failure modes
- **Unbounded per-track/per-video growth** — a `Dict[tracker_id, ...]` or `Dict[video_id, ...]` with no eviction. `velocity/v1.py` (`self._previous_positions`, `self._smoothed_velocities`) and `time_in_zone` (`self._batch_of_tracked_ids_in_zone`) are inherently unbounded; the tracker blocks added the bounded `InstanceCache` (PR #642, later #705) precisely to cap the seen-ids set. New code copying the analytics pattern without a bound is the highest-value catch.
- **Missing remote guard** — stateful block with no `get_restrictions()` restriction and no `NotImplementedError` remote guard, so it produces silently-wrong tracks/counts on hosted-serverless. The `RuntimeRestriction`/`Severity` framework was added to make this declarable; new stateful blocks must opt in.
- **`default=None` on required stateful manifest fields** — PR #658 removed `default=None` from line-counter / time-in-zone manifests because a defaulted zone silently produced no state instead of erroring.
- **Cross-session state bleed in shared aggregators** — PR #632 kept *distinct* exec sessions for InferencePipeline usage tracking; collapsing sessions into one namespace mixed unrelated streams' data.
- **Eviction bug in fixed-size caches** — PR #1212 (removal from an empty key_queue) and PR #1526 (fixed-size cache model-acquisition lock) are the model-manager analogue: bounded caches whose eviction path was unsafe. Same class of bug applies to block-level FIFO caches.
- **Reverted risky stateful block** — PR #2527 reverted the tracklet-recognition transformation block (#2497); treat brand-new stateful transformation blocks with extra scrutiny.
- **Ephemeral-cache assumptions** — PR #2387 added graceful fallback on ephemeral cache failure; state that assumes the cache is always present/durable breaks on serverless.

## Example implementations (point here)
- `inference/core/workflows/core_steps/trackers/_base.py` — canonical bounded per-video state: `TrackerBlockBase` keys `self._trackers` / `self._per_video_cache` by `video_identifier`, and `InstanceCache` is a FIFO `deque(maxlen)` + set with correct dual-structure eviction and `size = max(1, size)` clamp. New trackers subclass this instead of re-rolling state.
- `inference/core/workflows/core_steps/transformations/byte_tracker/v3.py` — the shipped per-video tracker + `InstanceCache` + `get_restrictions()` returning `STATEFUL_VIDEO_HTTP_SOFT_RESTRICTION` and `STILL_IMAGE_INPUT_SOFT_RESTRICTION`. Establishes the pattern (PR #642).
- `inference/core/workflows/core_steps/models/foundation/segment_anything2_video/v1.py` — the **fail-closed** reference: `run()` raises `NotImplementedError` when `step_execution_mode is not StepExecutionMode.LOCAL`, keys `self._sessions` by `video_identifier`, and `.clear()`s sessions on model switch. Copy this for any block whose state genuinely cannot survive remote execution.
- `inference/core/workflows/core_steps/analytics/time_in_zone/v2.py` — per-video zone + `tracked_ids_in_zone` state via `setdefault(video_identifier, {})`, with entry-timestamp cleanup on zone exit; requires `tracker_id` (raises if absent). Note: the id map is unbounded — good to cite as the shape to bound in new work.
- `inference/core/workflows/core_steps/flow_control/delta_filter/v1.py` — minimal per-`video_identifier` cache + `STATEFUL_VIDEO_HTTP_SOFT_RESTRICTION`; the smallest correct stateful-block skeleton.
- `inference/core/workflows/prototypes/block.py` — the contract source: `Severity`, `Runtime`, `StepExecutionMode`, `RuntimeRestriction`, and the `STATEFUL_VIDEO_HTTP_SOFT_RESTRICTION` / `COOLDOWN_HTTP_SOFT_RESTRICTION` presets a stateful block declares.
- `inference/core/workflows/core_steps/sinks/roboflow/model_monitoring_inference_aggregator/v1.py` — sink using `BaseCache` with a namespaced key (`workflows:steps_cache:...:{unique_aggregator_key}:last_report_time`) and `expire=900`; the pattern for durable/TTL'd cross-frame state instead of a raw instance dict.

## Severity guidance
- **Critical** — stateful block that produces silently-wrong output on remote/serverless with no restriction and no fail-closed guard (frames land on different workers, tracking/counting/cooldown is meaningless); or cross-tenant/cross-video state bleed from a shared/default namespace.
- **High** — unbounded per-track or per-video state with no eviction on a long-lived process (memory leak); eviction that only clears one of several backing structures; stale state reused after a config change it depends on (e.g. model switch without `.clear()`).
- **Medium** — TTL/reattach/cooldown window inconsistent but self-recovering; `default=None` on a stateful manifest field that should be required; un-namespaced cache key that could collide; missing `fps`/`video_identifier` fallback that degrades rather than crashes.
