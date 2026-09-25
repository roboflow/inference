"""Small helpers the stream runtime owns instead of importing `inference.core.utils`.

Each module is a behavior-preserving copy of the historical helper it
replaces; the originals stay in place for their other callers. Nothing here
reads configuration.
"""
