# Execution Engine Changelog

Below you can find the changelog for Execution Engine.

## Execution Engine `v1.7.0` | inference `v0.59.0`

!!! warning "Breaking change regarding step errors in workflows"
  
    To fix a bug related to invalid HTTP responses codes in `inference-server` handling Workflows execution requests 
    we needed to alter the default mechanism responsible for handling errors in Execution Engine. As a result of change,
    effective immediately on Roboflow Hosted Platform and in `inference>=0.59.0`, Workflow blocks interacting with 
    Roboflow platform which fails due to client misconfiguration (invalid Roboflow API key, invalid model ID, etc.) 
    instead of raising `StepExecutionError` (and HTTP 500 response from the server) will raise 
    `ClientCausedStepExecutionError` (and relevant HTTP response codes, such as 400, 401, 403, 404).

List of scenarios affected with the change:

* Block using Roboflow model defines invalid model ID - now will raise `ClientCausedStepExecutionError` with status code 400

* Block using Roboflow model defines invalid API key - now will raise `ClientCausedStepExecutionError` with status code 401

* Block using Roboflow model defines invalid API key or missing valid key with scpe to access resource - now will raise 
`ClientCausedStepExecutionError` with status code 403

* Block using Roboflow model defines model which does not exist - now will raise 
`ClientCausedStepExecutionError` with status code 404


!!! Note "Bringing back `legacy` error handling"
    
    It is possible to bring back the legacy behaviour of error handler if needed, which may be halpful in transition 
    period - all it takes is setting environmental variable `DEFAULT_WORKFLOWS_STEP_ERROR_HANDLER=legacy`.


## Execution Engine `v1.6.0` | inference `v0.53.0`

!!! Note "Change may require attention"

    This release introduces upgrades and new features with **no changes required** to existing workflows. 
    Some blocks may need to be upgraded to take advantage of the latest Execution Engine capabilities.

Prior versions of the Execution Engine had significant limitations when interacting with certain types of 
blocks - specifically those operating in Single Instruction, Multiple Data (SIMD) mode. These blocks are designed to 
process batches of inputs at once, apply the same operation to each element, and return results for the entire batch.

For example, the `run(...)` method of such a block might look like:

```python
def run(self, image: Batch[WorkflowImageData], confidence: float):
    pass
```

In the manifest, the `image` field is declared as accepting batches.

The issue arose when the input image came from a block that did not operate on batches. In such cases, the 
Execution Engine was unable to construct a batch from individual images, which often resulted in frustrating 
compilation errors such as:

```
Detected invalid reference plugged into property `images` of step `$steps.model` - the step property 
strictly requires batch-oriented inputs, yet the input selector holds non-batch oriented input - this indicates 
the problem with construction of your Workflow - usually the problem occurs when non-batch oriented step inputs are 
filled with outputs of non batch-oriented steps or non batch-oriented inputs.
```

In Execution Engine `v1.6.0`, this limitation has been removed, introducing the following behaviour:

* When it is detected that a given input must be batch-oriented, a procedure called **Auto Batch Casting** is applied. 
This automatically converts the input into a `Batch[T]`. Since all batch-mode inputs were already explicitly denoted in 
manifests, most blocks (with exceptions noted below) benefit from this upgrade without requiring any internal changes.

* The dimensionality (level of nesting) of an auto-batch cast parameter is determined at compilation time, based on the 
context of the specific block in the workflow as well as its manifest. If other batch-oriented inputs are present 
(referred to as *lineage supports*), the Execution Engine uses them as references when constructing auto-casted 
batches. This ensures that the number of elements in each batch dimension matches the other data fed into the step 
(simulating what would have been asserted if an actual batch input had been provided). If there are no 
*lineage supports*, or if the block manifest requires it (e.g. input dimensionality offset is set), the missing 
dimensions are generated similarly to the
[`torch.unsqueeze(...)` operation](https://docs.pytorch.org/docs/stable/generated/torch.unsqueeze.html).

* Step outputs are then evaluated against the presence of an Auto Batch Casting context. Based on the evaluation, 
outputs are saved either as batches or as scalars, ensuring that the effect of casting remains local, with the only 
exception being output dimensionality changes introduced by the block itself. As a side effect, it is now possible to:

    * **create output batches from scalars** (when the step increases dimensionality), and

    * **collapse batches into scalars** (when the block decreases dimensionality).

* The two potential friction point arises - first **when a block that does not accept batches** (and thus does not denote 
batch-accepting inputs) **decreases output dimensionality**. In previous versions, the Execution Engine handled this by 
applying dimensionality wrapping: all batch-oriented inputs were wrapped with an additional `Batch[T]` dimension, 
allowing the block’s `run(...)` method to perform reduce operations across the list dimension. With Auto Batch Casting, 
however, such blocks no longer provide the Execution Engine with a clear signal about whether certain inputs are 
scalars or batches, making casting nondeterministic. To address this, a new manifest method was introduced: 
`get_parameters_enforcing_auto_batch_casting(...)`. This method must return the list of parameters for which batch 
casting should be enforced when dimensionality is decreased. It is not expected to be used in any other context.

!!! warning "Impact of new method on existing blocks"

    The requirement of defining `get_parameters_enforcing_auto_batch_casting(...)` method to fully use 
    Auto Batch Casting feature in the case described above is non-strict. If the block will not be changed,
    the only effect will be that workflows wchich were **previously failing** with compilation error may 
    work or fail with **runtime error**, dependent on the details of block `run(...)` method implementation.

* The second friction point arises when there is a block declaring input fields supporting batches and scalars using 
`get_parameters_accepting_batches_and_scalars(...)` - by default, Execution Engine will skip auto-casting for such 
parameters, as the method was historically **always a way to declare that block itself has ability to broadcast scalars 
into batches** - see 
[implementation of `roboflow_core/detections_transformation@v1`](/inference/core/workflows/core_steps/transformations/detections_transformation/v1.py) 
block. In a way, Auto Batch Casting is *redundant* for those blocks - so we propose leaving them as is and 
upgrade to use `get_parameters_enforcing_auto_batch_casting(...)` instead of 
`get_parameters_accepting_batches_and_scalars(...)` in new versions of such blocks.

* In earlier versions, a hard constraint existed: dimensionality collapse could only occur at levels ≥ 2 (i.e. only 
on nested batches). This limitation is now removed. Dimensionality collapse blocks may also operate on scalars, with 
the output dimensionality “bouncing off” the zero ground.


There is one **key change in how outputs are built.** In earlier versions of Execution Error, a block was not allowed 
to produce a `Batch[X]` directly at the first dimension level — that space was reserved for mapping onto input batches.
Starting with version `v1.6.0`, this restriction has been removed. 

Previously, outputs were always returned as a list of elements:

* aligned with the input batches, or

* a single-element list if only scalars were given as inputs.

This raised a question: what should happen if a block now produces a batch at the first dimension level?
We cannot simply `zip(...)` it with input-based outputs, since the size of these newly generated batches might not 
match the number of input elements — making the operation ambiguous.

To resolve this, we adopted the following rule:

* Treat the situation as if there were a **"dummy" input batch of size 1**.

* Consider all batches produced from scalar inputs as being one level deeper than they appear.

* This follows the principle of broadcasting, allowing such outputs to expand consistently across all elements.

* Input batch may vanish as a result of execution, but when this happens and new first-level dimension emerges, it 
is still going to be virtually nested to ensure outputs consistency.

**Example:**

```
(NO INPUTS)    IMAGE FETCHER BLOCK --> image --> OD MODEL --> predictons --> CROPS --> output will be: ["crops": [<crop>, <crop>, ...]] 
```

It is important to note that **results generated from previously created workflows valid will be the same** and the 
change will only affect new workflows created to utilise new functionalities.

### Migration guide

??? Hint "Adding `get_parameters_enforcing_auto_batch_casting(...)` method"

    Blocks which decrease output dimensionality and do not define batch-oriented inputs needs to 
    declare all inputs which implementation expects to have wrapped in `Batch[T]` with the new class 
    method of block manifest called `get_parameters_enforcing_auto_batch_casting(...)`

    ```{ .py linenums="1" hl_lines="34-36 53-54"}
    from typing import List, Literal, Type, Union

    import supervision as sv
    
    from inference.core.workflows.execution_engine.entities.base import (
        Batch,
        OutputDefinition,
        WorkflowImageData,
    )
    from inference.core.workflows.execution_engine.entities.types import (
        IMAGE_KIND,
        OBJECT_DETECTION_PREDICTION_KIND,
        Selector,
    )
    from inference.core.workflows.prototypes.block import (
        BlockResult,
        WorkflowBlock,
        WorkflowBlockManifest,
    )
    
    
    class BlockManifest(WorkflowBlockManifest):
        type: Literal["my_plugin/tile_detections@v1"]
        crops: Selector(kind=[IMAGE_KIND])
        crops_predictions: Selector(
            kind=[OBJECT_DETECTION_PREDICTION_KIND]
        )
        scalar_parameter: Union[float, Selector()]
    
        @classmethod
        def get_output_dimensionality_offset(cls) -> int:
            return -1
    
        @classmethod
        def get_parameters_enforcing_auto_batch_casting(cls) -> List[str]:
            return ["crops", "crops_predictions"]
        
        @classmethod
        def describe_outputs(cls) -> List[OutputDefinition]:
            return [
                OutputDefinition(name="visualisations", kind=[IMAGE_KIND]),
            ]
    
    
    class TileDetectionsBlock(WorkflowBlock):
    
        @classmethod
        def get_manifest(cls) -> Type[WorkflowBlockManifest]:
            return BlockManifest
    
        def run(
            self,
            crops: Batch[WorkflowImageData],
            crops_predictions: Batch[sv.Detections],
            scalar_parameter: float,
        ) -> BlockResult:
            print("This is parameter which will not be auto-batch cast!", scalar_parameter)
            annotator = sv.BoxAnnotator()
            visualisations = []
            for image, prediction in zip(crops, crops_predictions):
                annotated_image = annotator.annotate(
                    image.numpy_image.copy(),
                    prediction,
                )
                visualisations.append(annotated_image)
            tile = sv.create_tiles(visualisations)
            return {"visualisations": tile}
    ```

    * in lines `34-36` one needs to add declaration of fields that will be subject to enforced auto-batch casting

    * as a result of the above, input parameters of run method (lines `53-54`) will be wrapped into `Batch[T]` by 
    Execution Engine.

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
