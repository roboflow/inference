# Execution Engine Changelog

Below you can find the changelog for Execution Engine.

## Execution Engine `v1.2.0` | inference `v0.23.0`

* The [`video_metadata` kind](/workflows/kinds/video_metadata/) has been deprecated, and we **strongly recommend discontinuing its use for building 
blocks moving forward**. As an alternative, the [`image` kind](/workflows/kinds/image/) has been extended to support the same metadata as 
[`video_metadata` kind](/workflows/kinds/video_metadata/), which can now be provided optionally. This update is 
**non-breaking** for existing blocks, but **some older blocks** that produce images **may become incompatible** with 
**future** video processing blocks.

??? warning "Potential blocks incompatibility"

    As previously mentioned, adding `video_metadata` as an optional field to the internal representation of 
    [`image` kind](/workflows/kinds/image/) (`WorkflowImageData` class) 
    may introduce some friction between existing blocks that output the [`image` kind](/workflows/kinds/image/) and 
    future video processing blocks that rely on `video_metadata` being part of `image` representation. 
    
    The issue arises because, while we can provide **default** values for `video_metadata` in `image` without 
    explicitly copying them from the input, any non-default metadata that was added upstream may be lost. 
    This can lead to downstream blocks that depend on the `video_metadata` not functioning as expected.

    We've updated all existing `roboflow_core` blocks to account for this, but blocks created before this change in
    external repositories may cause issues in workflows where their output images are used by video processing blocks.


* While the deprecated [`video_metadata` kind](/workflows/kinds/video_metadata/) is still available for use, it will be fully removed in 
Execution Engine version `v2.0.0`.

!!! warning "Breaking change planned - Execution Engine `v2.0.0`"

    [`video_metadata` kind](/workflows/kinds/video_metadata/) got deprecated and will be removed in `v2.0.0`


* As a result of the changes mentioned above, the internal representation of the [`image` kind](/workflows/kinds/image/) has been updated to 
include a new `video_metadata` property. This property can be optionally set in the constructor; if not provided, 
a default value with reasonable defaults will be used. To simplify metadata manipulation within blocks, we have 
introduced two new class methods: `WorkflowImageData.copy_and_replace(...)` and `WorkflowImageData.create_crop(...)`. 
For more details, refer to the updated [`WoorkflowImageData` usage guide](/workflows/internal_data_types/#workflowimagedata).
