"""Observation of event reactions.

``ExecutionObserver`` (``plan.py``) describes main-flow execution. Handler
runs report here instead, because they run on handler threads (asynchronous)
or inside the emitting call (synchronous), outside the main flow's callback
serialization::

    class Log(ReactionObserver):
        def on_reaction_finished(self, *, outcome, error, result):
            print(outcome.handler, outcome.status, outcome.cause.source_id)

    session = plan.create_session(reaction_observer=Log())

Callbacks of one reaction runtime are serialized by its own lock, never by
the main flow's. A callback must not wait for the run it observes.
"""

from typing import Any, Optional

__all__ = ["NULL_REACTION_OBSERVER", "ReactionObserver"]


class ReactionObserver:
    """No-op base receiving one notification per handled or lost event."""

    def on_reaction_finished(
        self,
        *,
        outcome: Any,
        error: Optional[BaseException],
        result: Optional[Any],
    ) -> None:
        """Called once per event a handler completed, failed, dropped or discarded.

        Args:
            outcome: The ``ReactionOutcome`` (metadata only).
            error: The handler's exception for ``failed``; ``None`` otherwise.
                Not retained by the engine. For an asynchronous handler the
                engine clears the frames of its traceback chain after this
                call, releasing the event's payload; the text stays printable.
            result: The handler run's ``RunResult`` for ``completed``;
                ``None`` otherwise. Not retained by the engine.
        """


NULL_REACTION_OBSERVER = ReactionObserver()
