"""Native V2 blocks that choose members of a group.

These are ordinary blocks: they receive a ``Batch`` for a ``Group`` field and
return which members they chose. The engine takes each result's source and
temporal context from the chosen member, so a chosen frame keeps its own
timestamp. Blocks never build contexts or layouts themselves.

* ``roboflow_core/best_frame@v1``: frames of one time window and the parent's reference
  image -> the ``frame`` most similar to the reference and its ``difference``.
  ``[N, T]`` frames with an ``[N]`` reference give ``[N]`` results; each
  parent may choose a different time position.
* ``roboflow_core/top_k_brightest@v1``: a group of images -> its ``k`` brightest images,
  brightest first, as a new collection ``[..., ranked]``, even when ``k`` is 1.

The choice is written with the member's full logical index, taken from the
delivered batch::

    best = min(range(len(frames)), key=differences.__getitem__)
    return {"frame": Selected(frames.indices[best])}
"""
