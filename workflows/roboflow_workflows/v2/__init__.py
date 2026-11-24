"""Opt-in Workflows V2: ``engine``, ``data`` and ``blocks``.

Import the part you need::

    from roboflow_workflows.v2.engine import Catalogue, compile_workflow
    from roboflow_workflows.v2.data import Batch, ImageData, Kind
    from roboflow_workflows.v2.blocks import create_catalogue

``engine`` compiles and runs workflows and imports no block family. ``data``
holds the shared data contracts and does not import the engine runtime.
``blocks`` composes the packaged block families. Importing this package loads
none of them.
"""
