# Execution Engine Changelog

Below you can find the changelog for Execution Engine.

## Execution Engine `v1.2.0` | inference `v0.23.0`

* The [`video_metadata` kind](/workflows/kinds/video_metadata.md) has been deprecated, and we **strongly recommend discontinuing its use for building 
blocks moving forward**. As an alternative, the [`image` kind](/workflows/kinds/image.md) has been extended to support the same metadata as 
[`video_metadata` kind](/workflows/kinds/video_metadata.md), which can now be provided optionally. This update is 
**non-breaking** for existing blocks, but **some older blocks** that produce images **may become incompatible** with 
**future** video processing blocks.

??? warning "Potential blocks incompatibility"

    As previously mentioned, adding `video_metadata` as an optional field to the internal representation of 
    [`image` kind](/workflows/kinds/image.md) (`WorkflowImageData` class) 
    may introduce some friction between existing blocks that output the [`image` kind](/workflows/kinds/image.md) and 
    future video processing blocks that rely on `video_metadata` being part of `image` representation. 
    
    The issue arises because, while we can provide **default** values for `video_metadata` in `image` without 
    explicitly copying them from the input, any non-default metadata that was added upstream may be lost. 
    This can lead to downstream blocks that depend on the `video_metadata` not functioning as expected.

    We've updated all existing `roboflow_core` blocks to account for this, but blocks created before this change in
    external repositories may cause issues in workflows where their output images are used by video processing blocks.


* While the deprecated [`video_metadata` kind](/workflows/kinds/video_metadata.md) is still available for use, it will be fully removed in 
Execution Engine version `v2.0.0`.

!!! warning "Breaking change planned - Execution Engine `v2.0.0`"

    [`video_metadata` kind](/workflows/kinds/video_metadata.md) got deprecated and will be removed in `v2.0.0`


* As a result of the changes mentioned above, the internal representation of the [`image` kind](/workflows/kinds/image.md) has been updated to 
include a new `video_metadata` property. This property can be optionally set in the constructor; if not provided, 
a default value with reasonable defaults will be used. To simplify metadata manipulation within blocks, we have 
introduced two new class methods: `WorkflowImageData.copy_and_replace(...)` and `WorkflowImageData.create_crop(...)`. 
For more details, refer to the updated [`WoorkflowImageData` usage guide](/workflows/internal_data_types.md#workflowimagedata).


## Execution Engine `v1.3.0` | inference `v0.27.0`

* Introduced the change that let each kind have serializer and deserializer defined. The change decouples Workflows 
plugins with Execution Engine and make it possible to integrate the ecosystem with external systems that 
require data transfer through the wire. [Blocks bundling](/workflows/blocks_bundling.md) page was updated to reflect 
that change.

* *Kinds* defined in `roboflow_core` plugin were provided with suitable serializers and deserializers

* Workflows Compiler and Execution Engine were enhanced to **support batch-oriented inputs of 
any *kind***, contrary to versions prior `v1.3.0`, which could only take `image` and `video_metadata` kinds
as batch-oriented inputs (as a result of unfortunate and not-needed coupling of kind to internal data 
format introduced **at the level of Execution Engine**). As a result of the change:

    * **new input type was introduced:** `WorkflowBatchInput` should be used from now on to denote 
    batch-oriented inputs (and clearly separate them from `WorkflowParameters`). `WorkflowBatchInput` 
    let users define both *[kind](/workflows/kinds.md)* of the data and it's 
    *[dimensionality](/workflows/workflow_execution.md#steps-interactions-with-data)*.
    New input type is effectively a superset of all previous batch-oriented inputs: `WorkflowImage` and
    `WorkflowVideoMetadata`, which **remain supported**, but **will be removed in Execution Engine `v2`**. 
    We advise adjusting to the new input format, yet the requirement is not strict at the moment - as 
    Execution Engine requires now explicit definition of input data *kind* to select data deserializer
    properly. This may not be the case in the future, as in most cases batch-oriented data *kind* may
    be inferred by compiler (yet this feature is not implemented for now).

    * **new selector type annotation was introduced** - named simply `Selector(...)`.
    `Selector(...)` is supposed to replace `StepOutputSelector`, `WorkflowImageSelector`, `StepOutputImageSelector`, 
    `WorkflowVideoMetadataSelector` and `WorkflowParameterSelector` in block manifests, 
    letting developers express that specific step manifest property is able to hold either selector of specific *kind*.
    Mentioned old annotation types **should be assumed deprecated**, we advise to migrate into `Selector(...)`. 

    * as a result of simplification in the selectors type annotations, the old selector will no 
    longer be providing the information on which parameter of blocks' `run(...)` method is 
    shipped by Execution Engine wrapped into [`Batch[X]` container](/workflows/internal_data_types.md#batch).
    Instead of old selectors type annotations and `block_manifest.accepts_batch_input()` method, 
    we propose the switch into two methods explicitly defining the parameters that are expected to 
    be fed with batch-oriented data (`block_manifest.get_parameters_accepting_batches()`) and 
    parameters capable of taking both *batches* and *scalar* values 
    (`block_manifest.get_parameters_accepting_batches_and_scalars()`). Return value of `block_manifest.accepts_batch_input()`
    is built upon the results of two new methods. The change is **non-breaking**, as any existing block which
    was capable of processing batches must have implemented `block_manifest.accepts_batch_input()` method returning
    `True` and use appropriate selector type annotation which indicated batch-oriented data.

* As a result of the changes, it is now possible to **split any arbitrary workflows into multiple ones executing 
subsets of steps**, enabling building such tools as debuggers.

!!! warning "Breaking change planned - Execution Engine `v2.0.0`"

    * `WorkflowImage` and `WorkflowVideoMetadata` inputs will be removed from Workflows ecosystem.

    * `StepOutputSelector, `WorkflowImageSelector`, `StepOutputImageSelector`, `WorkflowVideoMetadataSelector`
    and `WorkflowParameterSelector` type annotations used in block manifests will be removed from Workflows ecosystem.


### Migration guide

??? Hint "Kinds' serializers and deserializers" 

    Creating your Workflows plugin you may introduce custom serializers and deserializers
    for Workflows *kinds*. To achieve that end, simply place the following dictionaries
    in the main module of the plugin (the same where you place `load_blocks(...)` function):
    
    ```python
    from typing import Any
    
    def serialize_kind(value: Any) -> Any:
      # place here the code that will be used to
      # transform internal Workflows data representation into 
      # the external one (that can be sent through the wire in JSON, using
      # default JSON encoder for Python).
      pass
    
    
    def deserialize_kind(parameter_name: str, value: Any) -> Any:
      # place here the code that will be used to decode 
      # data sent through the wire into the Execution Engine
      # and transform it into proper internal Workflows data representation
      # which is understood by the blocks.
      pass
    
    
    KINDS_SERIALIZERS = {
        "name_of_the_kind": serialize_kind,
    }
    KINDS_DESERIALIZERS = {
        "name_of_the_kind": deserialize_kind,
    }
    ```

??? Hint "New type annotation for selectors - blocks without `Batch[X]` inputs"

    Blocks manifest may  **optionally** be updated to use `Selector` in the following way:
    
    ```python
    from typing import Union
    from inference.core.workflows.prototypes.block import WorkflowBlockManifest
    from inference.core.workflows.execution_engine.entities.types import (
        INSTANCE_SEGMENTATION_PREDICTION_KIND,
        OBJECT_DETECTION_PREDICTION_KIND,
        FLOAT_KIND,
        WorkflowImageSelector,
        StepOutputImageSelector,
        StepOutputSelector,
        WorkflowParameterSelector,
    )
    
    
    class BlockManifest(WorkflowBlockManifest):
    
        reference_image: Union[WorkflowImageSelector, StepOutputImageSelector]
        predictions: StepOutputSelector(
            kind=[
                OBJECT_DETECTION_PREDICTION_KIND,
                INSTANCE_SEGMENTATION_PREDICTION_KIND,
            ]
        )
        confidence: WorkflowParameterSelector(kind=[FLOAT_KIND]) 
    ```
    
    should just be changed into:
    
    ```{ .py linenums="1" hl_lines="7 12 13 19"}
    from inference.core.workflows.prototypes.block import WorkflowBlockManifest
    from inference.core.workflows.execution_engine.entities.types import (
        INSTANCE_SEGMENTATION_PREDICTION_KIND,
        OBJECT_DETECTION_PREDICTION_KIND,
        FLOAT_KIND,
        IMAGE_KIND,
        Selector,
    )
    
    
    class BlockManifest(WorkflowBlockManifest):
        reference_image: Selector(kind=[IMAGE_KIND])
        predictions: Selector(
            kind=[
                OBJECT_DETECTION_PREDICTION_KIND,
                INSTANCE_SEGMENTATION_PREDICTION_KIND,
            ]
        )
        confidence: Selector(kind=[FLOAT_KIND]) 
    ```

??? Hint "New type annotation for selectors - blocks with `Batch[X]` inputs"

    Blocks manifest may  **optionally** be updated to use `Selector` in the following way:
    
    ```python
    from typing import Union
    from inference.core.workflows.prototypes.block import WorkflowBlockManifest
    from inference.core.workflows.execution_engine.entities.types import (
        INSTANCE_SEGMENTATION_PREDICTION_KIND,
        OBJECT_DETECTION_PREDICTION_KIND,
        FLOAT_KIND,
        WorkflowImageSelector,
        StepOutputImageSelector,
        StepOutputSelector,
        WorkflowParameterSelector,
    )
    
    
    class BlockManifest(WorkflowBlockManifest):
    
        reference_image: Union[WorkflowImageSelector, StepOutputImageSelector]
        predictions: StepOutputSelector(
            kind=[
                OBJECT_DETECTION_PREDICTION_KIND,
                INSTANCE_SEGMENTATION_PREDICTION_KIND,
            ]
        )
        data: Dict[str, Union[StepOutputSelector(), WorkflowParameterSelector()]]
        confidence: WorkflowParameterSelector(kind=[FLOAT_KIND]) 

        @classmethod
        def accepts_batch_input(cls) -> bool:
            return True
    ```
    
    should be changed into:
    
    ```{ .py linenums="1" hl_lines="7 12 13 19 20 22-24 26-28"}
    from inference.core.workflows.prototypes.block import WorkflowBlockManifest
    from inference.core.workflows.execution_engine.entities.types import (
        INSTANCE_SEGMENTATION_PREDICTION_KIND,
        OBJECT_DETECTION_PREDICTION_KIND,
        FLOAT_KIND,
        IMAGE_KIND,
        Selector,
    )
    
    
    class BlockManifest(WorkflowBlockManifest):
        reference_image: Selector(kind=[IMAGE_KIND])
        predictions: Selector(
            kind=[
                OBJECT_DETECTION_PREDICTION_KIND,
                INSTANCE_SEGMENTATION_PREDICTION_KIND,
            ]
        )
        data: Dict[str, Selector()]
        confidence: Selector(kind=[FLOAT_KIND]) 

        @classmethod
        def get_parameters_accepting_batches(cls)W -> List[str]:
            return ["predictions"]
    
        @classmethod
        def get_parameters_accepting_batches_and_scalars(cls) -> List[str]:
            return ["data"]
    ```

    Please point out that:

    * the `data` property in the original example was able to accept both **batches** of data
    and **scalar** values due to selector of batch-orienetd data (`StepOutputSelector`) and 
    *scalar* data (`WorkflowParameterSelector`). Now the same is manifested by `Selector(...)` type 
    annotation and return value from `get_parameters_accepting_batches_and_scalars(...)` method.


??? Hint "New inputs in Workflows definitions"

    Anyone that used either `WorkflowImage` or `WorkflowVideoMetadata` inputs in their 
    Workflows definition may **optionally** migrate into `WorkflowBatchInput`. The transition
    is illustrated below:
    
    ```json
    {
      "inputs": [
        {"type": "WorkflowImage", "name": "image"},
        {"type": "WorkflowVideoMetadata", "name": "video_metadata"}
      ]
    }
    ```
    
    should be changed into:
    ```json
    {
      "inputs": [
        {
          "type": "WorkflowBatchInput",
          "name": "image",
          "kind": ["image"]
        },
        {
          "type": "WorkflowBatchInput",
          "name": "video_metadata",
          "kind": ["video_metadata"]
        }
      ]
    }
    ```
    
    **Leaving `kind` field empty may prevent some data - like images - from being deserialized properly.**
    
    
    !!! Note
    
        If you do not like the way how data is serialized in `roboflow_core` plugin, 
        feel free to alter the serialization methods for *kinds*, simply registering
        the function in your plugin and loading it to the Execution Engine - the 
        serializer/deserializer defined as the last one will be in use.


## Execution Engine `v1.4.0` | inference `v0.29.0`

* Added new kind - [`secret`](/workflows/kinds/secret.md) to represent credentials. **No action needed** for existing 
blocks, yet it is expected that over time blocks developers should use this kind, whenever block is to accept secret 
value as parameter.

* Fixed issue with results serialization introduced in `v1.3.0` - by mistake, Execution Engine was not serializing 
non-batch oriented outputs.

* Fixed Execution Engine bug with preparing inputs for steps. For non-SIMD steps before, while collecting inputs 
in runtime, `WorkflowBlockManifest.accepts_empty_input()` method result was being ignored - causing the bug when
one non-SIMD step was feeding empty values to downstream blocks. Additionally, in the light of changes made in `v1.3.0`,
thanks to which non-SIMD blocks can easily feed inputs for downstream SIMD steps - it is needed to check if 
upstream non-SIMD block yielded non-empty results (as SIMD block may not accept empty results). This check was added.
**No action needed** for existing blocks, but this fix may fix previously broken Workflows.


## Execution Engine `v1.5.0` | inference `v0.38.0`

!!! Note "Change does not require any action"
  
    This change does not require any change from Workflows users. This is just performance optimisation.

* Exposed new parameter in the init method of `BaseExecutionEngine` class - `executor` which can accept instance of 
Python `ThreadPoolExecutor` to be used by execution engine. Thanks to this change, processing should be faster, as 
each `BaseExecutionEngine.run(...)` will not require dedicated instance of `ThreadPoolExecutor` as it was so far.
Additionally, we are significantly limiting threads spawning which may also be a benefit in some installations.

* Despite the change, Execution Engine maintains the limit of concurrently executed steps - by limiting the number of
steps that run through the executor at a time (since  Execution Engine is no longer in control of `ThreadPoolExecutor` 
creation, and it is possible for the pool to have more workers available).

??? Hint "How to inject `ThreadPoolExecutor` to Execution Engine?"
    
    ```python
    from concurrent.futures import ThreadPoolExecutor
    workflow_init_parameters = { ... }
    with ThreadPoolExecutor(max_workers=...) as thread_pool_executor:
        execution_engine = ExecutionEngine.init(
            init_parameters=workflow_init_parameters,
            max_concurrent_steps=4,
            workflow_id="your-workflow-id",
            executor=thread_pool_executor,
        )
        runtime_parameters = {
          "image": cv2.imread("your-image-path")
        }
        results = execution_engine.run(runtime_parameters=runtime_parameters)
    ```
