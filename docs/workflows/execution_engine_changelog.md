# Execution Engine Changelog

Below you can find the changelog for Execution Engine.

## Execution Engine `v1.2.0` | inference `v0.23.0`

* [`video_metadata` kind](/workflows/kinds/video_metadata/) got deprecated - **we advise not using this kind for 
building blocks from now on.** As an alternative approach, [`image` kind](/workflows/kinds/image/) internal and
external representation got extended with the same metadata that 
[`video_metadata` kind](/workflows/kinds/video_metadata/) has, which can be provided optionally - making this 
change **non-breaking** for all old blocks, yet **making some old blocks producing images** incompatible with 
video processing blocks that will come in the future. Old blocks may also continue using 
[`video_metadata` kind](/workflows/kinds/video_metadata/), as it is left for now and will be removed in `v2.0.0`.

??? warning "Potential blocks incompatibility"

    As it was mentioned, adding `video_metadata` to `image` kind as optional set of metadata may create a friction 
    between already existing blocks that output `image` kind and future video processing blocks relying on 
    `video_metadata` that will be created. The reason is the following - since we are able to provide
    default value for `video_metadata` in `image`, without explicitly copying them from the input image in the block, 
    non-default metadata are lost and downstream blocks relying on the metadata may not work 100% as they should. 
    We adjusted all existing `roboflow_core` blocks to follow, but all blocks created prior this change in external 
    repositories may produce miss-behaviong Workflows when the output images are used by video-processing blocks.    

* As a result of the above, [`image` kind](/workflows/kinds/image/) got updated internal representation - there is 
new property `video_metadata` which can be optionally set in constructor, if not - default value with reasonable 
defaults will be provided. Additionally, to make it easier to manipulate metadata in blocks, `update_image(...)` and
`build_crop(...)` methods were added - see updates in 
[`WoorkflowImageData` usage guide](/workflows/internal_data_types/#workflowimagedata)


!!! warning "Breaking change planned - `v2.0.0`"

    [`video_metadata` kind](/workflows/kinds/video_metadata/) got deprecated and will be removed in `v2.0.0`


    