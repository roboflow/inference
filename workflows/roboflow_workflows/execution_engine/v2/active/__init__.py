"""Active execution of compiled V2 plans that declare sources.

``ExecutionSession.start`` delegates here. The runtime opens every declared
source on its own reader thread, admits emissions under a bounded per-source
guard and delivers each registered output group to its handler::

    run = session.start({"path": "temps.csv"}, handlers={"temperatures": on_group})
    run.wait()

By default one processor thread executes one pulse at a time over the shared
block instances (the serial reference). ``pipeline=PipelineOptions(...)``
runs pulses on a fixed pool of workers instead, overlapping different pulses
at different stages, each stage in sequence order per domain.

``execution`` holds the per-pulse primitives (``begin_pulse``,
``execute_pulse``, ``build_group_result``); ``pulses`` runs one pulse with
its deliveries and operator work, the same in both modes; ``runtime`` holds
the readers, the serial processor, ``ActiveRun`` and its lifecycle;
``pipeline`` holds the pipelined driver, imported only by pipelined runs.
The public V2 API imports these lightweight modules; source acquisition
starts only when a session is started.
"""
