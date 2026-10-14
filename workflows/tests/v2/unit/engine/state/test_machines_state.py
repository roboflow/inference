"""Machine controller semantics on the in-memory backend.

Backend-independent cases live in ``tests/v2/support/reactions/machine_state_contract.py``;
``integration/external/redis/reactions/test_machines_state.py`` runs them on Redis.
"""

import uuid
from typing import Iterator

import pytest
from roboflow_workflows.v2.blocks.state.state_machine_set import v1 as state_machine
from roboflow_workflows.v2.blocks.state.state_machine_set.v1 import StateMachineSetBlock
from roboflow_workflows.v2.engine.reactions.machines import TransitionResult
from roboflow_workflows.v2.engine.state import InMemoryStateBackend, ManagedState

from tests.v2.support.reactions.machine_state_contract import (  # noqa: F401  collected here
    test_compiled_plan_matches_stamps_by_machine_and_source,
    test_concurrent_events_apply_exactly_one_transition,
    test_corrupt_record_raises_explicitly,
    test_counters_are_independent_snapshots,
    test_endless_cas_misses_stop_with_explicit_error,
    test_fixed_transition_applies_once_and_emits_exact_event,
    test_foreign_source_stamp_never_selects_its_record,
    test_global_stamp_is_matched_through_source_causes,
    test_handler_decision_with_matching_stamp_applies_and_emits,
    test_invalid_source_id_is_rejected_before_any_record,
    test_lost_cas_on_stamped_decision_is_stale_without_retry,
    test_lost_cas_re_reads_and_ignores_when_state_moved,
    test_machines_are_visited_and_emitted_in_declaration_order,
    test_matching_stamp_behind_a_foreign_stamp_decides,
    test_matching_stamp_still_needs_a_legal_from_state,
    test_one_event_selects_at_most_one_transition_per_machine,
    test_reserved_record_never_aliases_public_state_keys,
    test_restart_keeps_record_and_deleted_record_is_explicit,
    test_scopes_sources_and_nested_identities_are_isolated,
    test_stale_stamp_after_a_b_a_is_rejected_without_emission,
    test_unauthorized_handler_and_illegal_target_raise,
    test_unknown_cas_outcome_propagates_without_retry_or_emission,
    test_unstamped_setter_uses_current_state,
    test_version_overflow_raises_without_change,
)


@pytest.fixture(params=["memory"])
def state(request) -> Iterator[ManagedState]:
    namespace = f"test-{uuid.uuid4().hex}"
    managed = ManagedState(InMemoryStateBackend(), namespace=namespace)
    try:
        yield managed
    finally:
        managed.backend.close()


def test_setter_block_returns_result_fields(monkeypatch) -> None:
    calls = []

    class _Context:
        def set_machine_state(self, machine, transition, next_state):
            calls.append((machine, transition, next_state))
            return TransitionResult("stale", state="rejected")

    monkeypatch.setattr(state_machine, "get_execution_context", _Context)

    outputs = StateMachineSetBlock().run(
        trigger=None,
        machine="inspection",
        transition="finish_review",
        next_state="approved",
    )

    assert calls == [("inspection", "finish_review", "approved")]
    assert outputs == {"state": "rejected", "applied": False, "outcome": "stale"}
