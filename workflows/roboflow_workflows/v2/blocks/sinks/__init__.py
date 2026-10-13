"""Sink blocks: effects outside the workflow, such as files and HTTP requests.

Sinks are never pruned: their steps run even when no output is demanded.
``runtime`` holds the shared outcome helpers and ``BackgroundEffects``, the
host-owned resource behind ``fire_and_forget``.
"""
