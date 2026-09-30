"""Active execution of compiled V2 plans that declare sources.

``ExecutionSession.start`` delegates here. The runtime opens every declared
source on its own reader thread, admits emissions under a bounded per-source
guard, executes one pulse at a time on one processor thread over the shared
block instances, and delivers each registered output group to its handler::

    run = session.start({"path": "temps.csv"}, handlers={"temperatures": on_group})
    run.wait()

``execution`` holds the per-pulse primitives (``begin_pulse``,
``execute_pulse``, ``build_group_result``); ``runtime`` holds the readers,
processor, ``ActiveRun`` and its lifecycle. The public V2 API imports these
lightweight modules; source acquisition starts only when a session is started.
"""
