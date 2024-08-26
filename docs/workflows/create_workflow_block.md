# Creating Workflow block

Workflows blocks development is the process that requires understanding of 
Workflow Ecosystem. Before diving deep in the details, let's summarise the 
knowledge that would be required:

Understanding of [Workflow execution](/workflows/workflow_execution/), in particular:
    
* what is the relation of Workflow blocks and steps in Workflow definition

* how Workflow blocks and their manifests are used by [Workflows Compiler](/workflows/workflows_compiler/)

* what is the `dimensionality level` of batch-oriented data passing through Workflow

* how [Execution Engine](/workflows/workflows_execution_engine/) interact with step, regarding 
its inputs and outputs

* what is the nature and role of [Workflow `kinds`](/workflows/kinds/)

* understands how [`pydantic`](https://docs.pydantic.dev/latest/) works

## Prototypes

To create a Workflow block you need some amount of imports from the core of Workflows library.
Here is the list of imports that you may find useful while creating a block:

```python
from inference.core.workflows.execution_engine.entities.base import (
    Batch,  # batches of data will come in Batch[X] containers
    OutputDefinition,  # class used to declare outputs in your manifest
    WorkflowImageData,  # internal representation of image
    # - use whenever your input kind is image
)

from inference.core.workflows.prototypes.block import (
    BlockResult,  # type alias for result of `run(...)` method
    WorkflowBlock,  # base class for your block
    WorkflowBlockManifest,  # base class for block manifest
)

from inference.core.workflows.execution_engine.entities.types import *  
# module with `kinds` from the core library
```

The most important are:

* `WorkflowBlock` - base class for your block

* `WorkflowBlockManifest` - base class for block manifest

## Block manifest

A manifest is a crucial component of a Workflow block that defines a prototype 
for step declaration that can be placed in Workflow definition to use the block. 
In particular it: 

* **Uses `pydantic` to power syntax parsing of Workflows definitions:** 
It inherits from  [`pydantic BaseModel`](https://docs.pydantic.dev/latest/api/base_model/) features to parse and 
validate Workflow definitions. This schema can also be automatically exported to a format compatible with the 
Workflows UI, thanks to `pydantic's` integration with the OpenAPI standard.

* **Defines Data Bindings:** It specifies which fields in the manifest are selectors for data flowing through 
the workflow during execution and indicates their kinds.

* **Describes Block Outputs:** It outlines the outputs that the block will produce.

* **Specifies Dimensionality:** It details the properties related to input and output dimensionality.

* **Indicates Batch Inputs and Empty Values:** It informs the Execution Engine whether the step accepts batch 
inputs and empty values.

* **Ensures Compatibility:** It dictates the compatibility with different Execution Engine versions to maintain 
stability. For more details, see [versioning](/workflows/versioning/).

### Scaffolding for manifest

To understand how manifests work, let's define one step-by-step. The example block that we build here will be 
calculating images similarity. We start from imports and class scaffolding:

```python
from typing import Literal
from inference.core.workflows.prototypes.block import (
    WorkflowBlockManifest,
)

class ImagesSimilarityManifest(WorkflowBlockManifest):
    type: Literal["my_plugin/images_similarity@v1"] 
    name: str
```

This is the minimal representation of manifest. It defines two special fields that are important for 
Compiler and Execution engine:

* `type` - required to parse syntax of Workflows definitions based on dynamic pool of blocks - this is the 
[`pydantic` type discriminator](https://docs.pydantic.dev/latest/concepts/unions/#discriminated-unions) that let
Compiler understand which block manifest is to be verified when parsing specific step in Workflow definition

* `name` - this property will be used to give the step unique name and let other steps selects it via selectors

### Adding batch-oriented inputs

We want our step to take two batch-oriented inputs with images to be compared - so effectively
we will be creating SIMD block. 

??? example "Adding batch-oriented inputs"
    
    Let's see how to add definitions of those inputs to manifest: 

    ```{ .py linenums="1" hl_lines="2 6 7 8 9 17 18 19 20 21 22"}
    from typing import Literal, Union
    from pydantic import Field
    from inference.core.workflows.prototypes.block import (
        WorkflowBlockManifest,
    )
    from inference.core.workflows.execution_engine.entities.types import (
        StepOutputImageSelector,
        WorkflowImageSelector,
    )
    
    class ImagesSimilarityManifest(WorkflowBlockManifest):
        type: Literal["my_plugin/images_similarity@v1"] 
        name: str
        # all properties apart from `type` and `name` are treated as either 
        # definitions of batch-oriented data to be processed by block or its 
        # parameters that influence execution of steps created based on block
        image_1: Union[WorkflowImageSelector, StepOutputImageSelector] = Field(
            description="First image to calculate similarity",
        )
        image_2: Union[WorkflowImageSelector, StepOutputImageSelector] = Field(
            description="Second image to calculate similarity",
        )
    ```
    
    * in the lines `2-9`, we've added a couple of imports ro ensure that we have everything needed
    
    * line `17` defines `image_1` parameter - as manifest is prototype for Workflow Definition, 
    the only way to tell about image to be used by step is to provide selector - we have 
    two specialised types in core library that can be used - `WorkflowImageSelector` and `StepOutputImageSelector`.
    If you look deeper into codebase, you will discover those are type aliases - telling `pydantic`
    to expect string matching `$inputs.{name}` and `$steps.{name}.*` patterns respectively, additionally providing 
    extra schema field metadata that tells Workflows ecosystem components that the `kind` of data behind selector is 
    [image](/workflows/kinds/batch_image/).
  
    * denoting `pydantic` `Field(...)` attribute in the last parts of line `17` is optional, yet appreciated, 
    especially for blocks intended to cooperate with Workflows UI 
  
    * starting in line `20`, you can find definition of `image_2` parameter which is very similar to `image_1`.


Such definition of manifest can handle the following step declaration in Workflow definition:

```json
{
  "type": "my_plugin/images_similarity@v1",
  "name": "my_step",
  "image_1": "$inputs.my_image",
  "image_2": "$steps.image_transformation.image"
}
```

This definition will make Compiler and Execution Engine to:

* select as a step prototype the block which declared manifest with type discriminator being 
`my_plugin/images_similarity@v1`

* ship into step run method two parameters:

  * `input_1` of type `WorkflowImageData` which will be filled with image submitted as Workflow execution input
  
  * `imput_2` of type `WorkflowImageData` which will be generated in the runtime, by other step called 
  `image_transformation`


### Adding parameter to the manifest

Let's now add the parameter that will influence step execution. The parameter is not assumed to be 
batch-oriented and will affect all batch elements passed to the step.

??? example "Adding parameter to the manifest"

    ```{ .py linenums="1" hl_lines="9 10 11 26 27 28 29 30 31 32"}
    from typing import Literal, Union
    from pydantic import Field
    from inference.core.workflows.prototypes.block import (
        WorkflowBlockManifest,
    )
    from inference.core.workflows.execution_engine.entities.types import (
        StepOutputImageSelector,
        WorkflowImageSelector,
        FloatZeroToOne,
        WorkflowParameterSelector,
        FLOAT_ZERO_TO_ONE_KIND,
    )
    
    class ImagesSimilarityManifest(WorkflowBlockManifest):
        type: Literal["my_plugin/images_similarity@v1"] 
        name: str
        # all properties apart from `type` and `name` are treated as either 
        # definitions of batch-oriented data to be processed by block or its 
        # parameters that influence execution of steps created based on block
        image_1: Union[WorkflowImageSelector, StepOutputImageSelector] = Field(
            description="First image to calculate similarity",
        )
        image_2: Union[WorkflowImageSelector, StepOutputImageSelector] = Field(
            description="Second image to calculate similarity",
        )
        similarity_threshold: Union[
            FloatZeroToOne,
            WorkflowParameterSelector(kind=[FLOAT_ZERO_TO_ONE_KIND]),
        ] = Field(
            default=0.4,
            description="Threshold to assume that images are similar",
        )
    ```
    
    * line `9` imports `FloatZeroToOne` which is type alias providing validation 
    for float values in range 0.0-1.0 - this is based on native `pydantic` mechanism and
    everyone could create this type annotation locally in module hosting block
    
    * line `10` imports function `WorkflowParameterSelector(...)` capable to dynamically create 
    `pydantic` type annotation for selector to workflow input parameter (matching format `$inputs.param_name`), 
    declaring union of kinds compatible with the field
  
    * line `11` imports [`float_zero_to_one`](/workflows/kinds/float_zero_to_one) `kind` definition which will be used later
  
    * in line `26` we start defining parameter called `similarity_threshold`. Manifest will accept 
    either float values (in range `[0.0-1.0]`) or selector to workflow input of `kind`
    [`float_zero_to_one`](/workflows/kinds/float_zero_to_one). Please point out on how 
    function creating type annotation (`WorkflowParameterSelector(...)`) is used - 
    in particular, expected `kind` of data is passed as list of `kinds` - representing union
    of expected data `kinds`.

Such definition of manifest can handle the following step declaration in Workflow definition:

```{ .json linenums="1" hl_lines="6"}
{
  "type": "my_plugin/images_similarity@v1",
  "name": "my_step",
  "image_1": "$inputs.my_image",
  "image_2": "$steps.image_transformation.image",
  "similarity_threshold": "$inputs.my_similarity_threshold"
}
```

or alternatively:

```{ .json linenums="1" hl_lines="6"}
{
  "type": "my_plugin/images_similarity@v1",
  "name": "my_step",
  "image_1": "$inputs.my_image",
  "image_2": "$steps.image_transformation.image",
  "similarity_threshold": "0.5"
}
```

??? hint "LEARN MORE: Selecting step outputs"

    Our siplified example showcased declaration of properties that accept selectors to
    images produced by other steps via `StepOutputImageSelector`.

    You can use function `StepOutputSelector(...)` creating field annotations dynamically
    to express the that block accepts batch-oriented outputs from other steps of specified
    kinds

    ```{ .py linenums="1" hl_lines="9 10 25"}
    from typing import Literal, Union
    from pydantic import Field
    from inference.core.workflows.prototypes.block import (
        WorkflowBlockManifest,
    )
    from inference.core.workflows.execution_engine.entities.types import (
        StepOutputImageSelector,
        WorkflowImageSelector,
        StepOutputSelector,
        NUMPY_ARRAY_KIND,
    )
    
    class ImagesSimilarityManifest(WorkflowBlockManifest):
        type: Literal["my_plugin/images_similarity@v1"] 
        name: str
        # all properties apart from `type` and `name` are treated as either 
        # definitions of batch-oriented data to be processed by block or its 
        # parameters that influence execution of steps created based on block
        image_1: Union[WorkflowImageSelector, StepOutputImageSelector] = Field(
            description="First image to calculate similarity",
        )
        image_2: Union[WorkflowImageSelector, StepOutputImageSelector] = Field(
            description="Second image to calculate similarity",
        )
        example: StepOutputSelector(kind=[NUMPY_ARRAY_KIND])
    ```

### Declaring block outputs

Our manifest is ready regarding properties that can be declared in Workflow definition, 
but we still need to provide additional information for Execution Engine to successfully 
run the block.

??? example "Declaring block outputs"

    Minimal set of information required is outputs description. Additionally, 
    to increase block stability, we advise to provide information about execution engine 
    compatibility.
    
    ```{ .py linenums="1" hl_lines="1 5 13 33-44 46-48"}
    from typing import Literal, Union, List, Optional
    from pydantic import Field
    from inference.core.workflows.prototypes.block import (
        WorkflowBlockManifest,
        OutputDefinition,
    )
    from inference.core.workflows.execution_engine.entities.types import (
        StepOutputImageSelector,
        WorkflowImageSelector,
        FloatZeroToOne,
        WorkflowParameterSelector,
        FLOAT_ZERO_TO_ONE_KIND,
        BOOLEAN_KIND,
    )
    
    class ImagesSimilarityManifest(WorkflowBlockManifest):
        type: Literal["my_plugin/images_similarity@v1"] 
        name: str
        image_1: Union[WorkflowImageSelector, StepOutputImageSelector] = Field(
            description="First image to calculate similarity",
        )
        image_2: Union[WorkflowImageSelector, StepOutputImageSelector] = Field(
            description="Second image to calculate similarity",
        )
        similarity_threshold: Union[
            FloatZeroToOne,
            WorkflowParameterSelector(kind=[FLOAT_ZERO_TO_ONE_KIND]),
        ] = Field(
            default=0.4,
            description="Threshold to assume that images are similar",
        )
        
        @classmethod
        def describe_outputs(cls) -> List[OutputDefinition]:
            return [
              OutputDefinition(
                name="images_match", 
                kind=[BOOLEAN_KIND],
              )
            ]
    
        @classmethod
        def get_execution_engine_compatibility(cls) -> Optional[str]:
            return ">=1.0.0,<2.0.0"
    ```
    
    * line `1` contains additional imports from `typing`
    
    * line `5` imports class that is used to describe step outputs
  
    * line `13` imports [`boolean`](/workflows/kinds/boolean) `kind` to be used 
    in outputs definitions
  
    * lines `33-44` declare class method to specify outputs from the block - 
    each entry in list declare one return property for each batch element and its `kind`.
    Our block will return boolean flag `images_match` for each pair of images.
  
    * lines `46-48` declare compatibility of the block with Execution Engine -
    see [versioning page](/workflows/versioning/) for more details

As a result of those changes:

* Execution Engine would understand that steps created based on this block 
are supposed to deliver specified outputs and other steps can refer to those outputs
in their inputs

* blocks loading mechanism will not load the block given that Execution Engine is not in version `v1`

??? hint "LEARN MORE: Dynamic outputs"

    Some blocks may not be able to arbitrailry define their outputs using 
    classmethod - regardless of the content of step manifest that is available after 
    parsing. To support this we introduced the following convention:

    * classmethod `describe_outputs(...)` shall return list with one element of 
    name `*` and kind `*` (aka `WILDCARD_KIND`)

    * additionally, block manifest should implement instance method `get_actual_outputs(...)`
    that provides list of actual outputs that can be generated based on filled manifest data 

    ```{ .py linenums="1" hl_lines="14 35-42 44-49"}
    from typing import Literal, Union, List, Optional
    from pydantic import Field
    from inference.core.workflows.prototypes.block import (
        WorkflowBlockManifest,
        OutputDefinition,
    )
    from inference.core.workflows.execution_engine.entities.types import (
        StepOutputImageSelector,
        WorkflowImageSelector,
        FloatZeroToOne,
        WorkflowParameterSelector,
        FLOAT_ZERO_TO_ONE_KIND,
        BOOLEAN_KIND,
        WILDCARD_KIND,
    )
    
    class ImagesSimilarityManifest(WorkflowBlockManifest):
        type: Literal["my_plugin/images_similarity@v1"] 
        name: str
        image_1: Union[WorkflowImageSelector, StepOutputImageSelector] = Field(
            description="First image to calculate similarity",
        )
        image_2: Union[WorkflowImageSelector, StepOutputImageSelector] = Field(
            description="Second image to calculate similarity",
        )
        similarity_threshold: Union[
            FloatZeroToOne,
            WorkflowParameterSelector(kind=[FLOAT_ZERO_TO_ONE_KIND]),
        ] = Field(
            default=0.4,
            description="Threshold to assume that images are similar",
        )
        outputs: List[str]
        
        @classmethod
        def describe_outputs(cls) -> List[OutputDefinition]:
            return [
              OutputDefinition(
                name="*", 
                kind=[WILDCARD_KIND],
              ),
            ]

        def get_actual_outputs(self) -> List[OutputDefinition]:
            # here you have access to `self`:
            return [
              OutputDefinition(name=e, kind=[BOOLEAN_KIND])
              for e in self.outputs
            ]
    ```


## Definition of block class

At this stage, manifest of our simple block is ready, we will continue 
with our example, letting [advanced topics](#advanced-topics) section to
cover more details that would be just distractions at this stage.

### Base implementation

Having the manifest ready, we can prepare baseline implementation of the 
block.

??? example "Block scaffolding"

    ```{ .py linenums="1" hl_lines="1 5 6 8-11 56-68"}
    from typing import Literal, Union, List, Optional, Type
    from pydantic import Field
    from inference.core.workflows.prototypes.block import (
        WorkflowBlockManifest,
        WorkflowBlock,
        BlockResult,
    )
    from inference.core.workflows.execution_engine.entities.base import (
        OutputDefinition,
        WorkflowImageData,
    )
    from inference.core.workflows.execution_engine.entities.types import (
        StepOutputImageSelector,
        WorkflowImageSelector,
        FloatZeroToOne,
        WorkflowParameterSelector,
        FLOAT_ZERO_TO_ONE_KIND,
        BOOLEAN_KIND,
    )
    
    class ImagesSimilarityManifest(WorkflowBlockManifest):
        type: Literal["my_plugin/images_similarity@v1"] 
        name: str
        image_1: Union[WorkflowImageSelector, StepOutputImageSelector] = Field(
            description="First image to calculate similarity",
        )
        image_2: Union[WorkflowImageSelector, StepOutputImageSelector] = Field(
            description="Second image to calculate similarity",
        )
        similarity_threshold: Union[
            FloatZeroToOne,
            WorkflowParameterSelector(kind=[FLOAT_ZERO_TO_ONE_KIND]),
        ] = Field(
            default=0.4,
            description="Threshold to assume that images are similar",
        )
        
        @classmethod
        def describe_outputs(cls) -> List[OutputDefinition]:
            return [
              OutputDefinition(
                name="images_match", 
                kind=[BOOLEAN_KIND],
              ),
            ]
    
        @classmethod
        def get_execution_engine_compatibility(cls) -> Optional[str]:
            return ">=1.0.0,<2.0.0"
    
        
    class ImagesSimilarityBlock(WorkflowBlock):
      
        @classmethod
        def get_manifest(cls) -> Type[WorkflowBlockManifest]:
            return ImagesSimilarityManifest
    
        def run(
            self,
            image_1: WorkflowImageData,
            image_2: WorkflowImageData,
            similarity_threshold: float,
        ) -> BlockResult:
            pass
    ```

    * lines `1`, `5-6` and `8-9` added changes into import surtucture to 
    provide additional symbols required to properly define block class and all
    of its methods signatures

    * line `59` defines class method `get_manifest(...)` to simply return 
    the manifest class we cretaed earlier

    * lines `62-68` define `run(...)` function, which Execution Engine
    will invoke with data to get desired results

### Providing implementation for block logic

Let's now add example implementation of `run(...)` method to our block, such that
it can produce meaningful results.

!!! Note
    
    Content of this section is supposed to provide example on how to interact 
    with Workflow ecosystem as block creator, rather than providing robust 
    implementation of the block.

??? example "Implementation of `run(...)` method"

    ```{ .py linenums="1" hl_lines="3 56-58 70-81"}
    from typing import Literal, Union, List, Optional, Type
    from pydantic import Field
    import cv2
    
    from inference.core.workflows.prototypes.block import (
        WorkflowBlockManifest,
        WorkflowBlock,
        BlockResult,
    )
    from inference.core.workflows.execution_engine.entities.base import (
        OutputDefinition,
        WorkflowImageData,
    )
    from inference.core.workflows.execution_engine.entities.types import (
        StepOutputImageSelector,
        WorkflowImageSelector,
        FloatZeroToOne,
        WorkflowParameterSelector,
        FLOAT_ZERO_TO_ONE_KIND,
        BOOLEAN_KIND,
    )
    
    class ImagesSimilarityManifest(WorkflowBlockManifest):
        type: Literal["my_plugin/images_similarity@v1"] 
        name: str
        image_1: Union[WorkflowImageSelector, StepOutputImageSelector] = Field(
            description="First image to calculate similarity",
        )
        image_2: Union[WorkflowImageSelector, StepOutputImageSelector] = Field(
            description="Second image to calculate similarity",
        )
        similarity_threshold: Union[
            FloatZeroToOne,
            WorkflowParameterSelector(kind=[FLOAT_ZERO_TO_ONE_KIND]),
        ] = Field(
            default=0.4,
            description="Threshold to assume that images are similar",
        )
        
        @classmethod
        def describe_outputs(cls) -> List[OutputDefinition]:
            return [
              OutputDefinition(
                name="images_match", 
                kind=[BOOLEAN_KIND],
              ),
            ]
    
        @classmethod
        def get_execution_engine_compatibility(cls) -> Optional[str]:
            return ">=1.0.0,<2.0.0"
    
        
    class ImagesSimilarityBlock(WorkflowBlock):
      
        def __init__(self):
            self._sift = cv2.SIFT_create()
            self._matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
          
        @classmethod
        def get_manifest(cls) -> Type[WorkflowBlockManifest]:
            return ImagesSimilarityManifest
    
        def run(
            self,
            image_1: WorkflowImageData,
            image_2: WorkflowImageData,
            similarity_threshold: float,
        ) -> BlockResult:
            image_1_gray = cv2.cvtColor(image_1.numpy_image, cv2.COLOR_BGR2GRAY)
            image_2_gray = cv2.cvtColor(image_2.numpy_image, cv2.COLOR_BGR2GRAY)
            kp_1, des_1 = self._sift.detectAndCompute(image_1_gray, None)
            kp_2, des_2 = self._sift.detectAndCompute(image_2_gray, None)
            matches = self._matcher.knnMatch(des_1, des_2, k=2)
            good_matches = []
            for m, n in matches:
                if m.distance < similarity_threshold * n.distance:
                    good_matches.append(m)
            return {
                "images_match": len(good_matches) > 0,
            }
    ```

    * in line `3` we import OpenCV

    * lines `56-58` defines block constructor, thanks to this - state of block 
    is initialised once and live through consecutive invocation of `run(...)` method - for 
    instance when Execution Engine runs on consecutive frames of video

    * lines `70-81` provide implementation of block functionality - the details are trully not
    important regarding Workflows ecosystem, but there are few details you should focus:
    
        * lines `70` and `71` make use of `WorkflowImageData` abstraction, showcasing how 
        `numpy_image` property can be used to get `np.ndarray` from internal representation of images
        in Workflows. We advise to expole remaining properties of `WorkflowImageData` to discover more.

        * result of workflow block execution, declared in lines `79-81` is in our case just a dictionary 
        **with the keys being the names of outputs declared in manifest**, in line `44`. Be sure to provide all
        declared outputs - otherwise Execution Engine will raise error.
        
You may ask yourself how it is possible that implemented block accepts batch-oriented workflow input, but do not 
operate on batches directly. This is due to the fact that the default block behaviour is to run one-by-one against
all elements of input batches. We will show how to change that in [advanced topics](#advanced-topics) section.

## Exposing block in `plugin`

Now, your block is ready to be used, but if you declared step using it in your Workflow definition you 
would see an error. This is because no plugin exports the block you just created. Details of blocks bundling 
will be covered in [separate page](/workflows/blocks_bundling/), but the remaining thing to do is to 
add block class into list returned from your plugind `load_blocks(...)` function:

```python
# __init__.py of your plugin

from my_plugin.images_similarity.v1 import  ImagesSimilarityBlock  
# this is example import! requires adjustment

def load_blocks():
    return [ImagesSimilarityBlock]
```


## Advanced topics

### Blocks processing batches of inputs

Sometimes, performance of your block may benefit if all input data is processed at once as batch. This may
happen for models running on GPU. Such mode of operation is supported for Workflows blocks - here is the example
on how to use it for your block.

??? example "Implementation of blocks accepting batches"

    ```{ .py linenums="1" hl_lines="13 41-43 71-72 75-78 86-87"}
    from typing import Literal, Union, List, Optional, Type
    from pydantic import Field
    import cv2
    
    from inference.core.workflows.prototypes.block import (
        WorkflowBlockManifest,
        WorkflowBlock,
        BlockResult,
    )
    from inference.core.workflows.execution_engine.entities.base import (
        OutputDefinition,
        WorkflowImageData,
        Batch,
    )
    from inference.core.workflows.execution_engine.entities.types import (
        StepOutputImageSelector,
        WorkflowImageSelector,
        FloatZeroToOne,
        WorkflowParameterSelector,
        FLOAT_ZERO_TO_ONE_KIND,
        BOOLEAN_KIND,
    )
    
    class ImagesSimilarityManifest(WorkflowBlockManifest):
        type: Literal["my_plugin/images_similarity@v1"] 
        name: str
        image_1: Union[WorkflowImageSelector, StepOutputImageSelector] = Field(
            description="First image to calculate similarity",
        )
        image_2: Union[WorkflowImageSelector, StepOutputImageSelector] = Field(
            description="Second image to calculate similarity",
        )
        similarity_threshold: Union[
            FloatZeroToOne,
            WorkflowParameterSelector(kind=[FLOAT_ZERO_TO_ONE_KIND]),
        ] = Field(
            default=0.4,
            description="Threshold to assume that images are similar",
        )

        @classmethod
        def accepts_batch_input(cls) -> bool:
            return True
        
        @classmethod
        def describe_outputs(cls) -> List[OutputDefinition]:
            return [
              OutputDefinition(
                name="images_match", 
                kind=[BOOLEAN_KIND],
              ),
            ]
    
        @classmethod
        def get_execution_engine_compatibility(cls) -> Optional[str]:
            return ">=1.0.0,<2.0.0"
    
        
    class ImagesSimilarityBlock(WorkflowBlock):
      
        def __init__(self):
            self._sift = cv2.SIFT_create()
            self._matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
          
        @classmethod
        def get_manifest(cls) -> Type[WorkflowBlockManifest]:
            return ImagesSimilarityManifest
    
        def run(
            self,
            image_1: Batch[WorkflowImageData],
            image_2: Batch[WorkflowImageData],
            similarity_threshold: float,
        ) -> BlockResult:
            results = []
            for image_1_element, image_2_element in zip(image_1, image_2): 
              image_1_gray = cv2.cvtColor(image_1_element.numpy_image, cv2.COLOR_BGR2GRAY)
              image_2_gray = cv2.cvtColor(image_2_element.numpy_image, cv2.COLOR_BGR2GRAY)
              kp_1, des_1 = self._sift.detectAndCompute(image_1_gray, None)
              kp_2, des_2 = self._sift.detectAndCompute(image_2_gray, None)
              matches = self._matcher.knnMatch(des_1, des_2, k=2)
              good_matches = []
              for m, n in matches:
                  if m.distance < similarity_threshold * n.distance:
                      good_matches.append(m)
              results.append({"images_match": len(good_matches) > 0})
            return results
    ```

    * line `13` imports `Batch` from core of workflows library - this class represent container which is 
    veri similar to list (but read-only) to keep batch elements

    * lines `41-43` define class method that changes default behaviour of the block and make it capable 
    to process batches

    * changes introduced above made the signature of `run(...)` method to change, now `image_1` and `image_2`
    are not instances of `WorkflowImageData`, but rather batches of elements of this type

    * lines `75-78`, `86-87` present changes that needed to be introduced to run processing across all batch 
    elements - showcasing how to iterate over batch elements if needed

    * it is important to note how outputs are constructed in line `86` - each element of batch will be given
    its entry in the list which is returned from `run(...)` method. Order must be aligned with order of batch 
    elements. Each output dictionary must provide all keys declared in block outputs.

### Implementation of flow-control block

Flow-control blocks differs quite substantially from other blocks that just process the data. Here we will show 
how to create a flow control block, but first - a little bit of theory:

* flow-control block is the block that declares compatibility with step selectors in their manifest (selector to step
is defined as `$steps.{step_name}` - similar to step output selector, but without specification of output name)

* flow-control blocks cannot register outputs, they are meant tu return `FlowControl` objects

* `FlowControl` object specify next steps (from selectors provided in step manifest) that for given 
batch element (SIMD flow-control) or whole workflow execution (non-SIMD flow-control) should pick up next

??? example "Implementation of flow-control - SIMD block"
    
    Example provides and comments out implementation of random continue block

    ```{ .py linenums="1" hl_lines="10 14 26 28-31 55-56"}
    from typing import List, Literal, Optional, Type, Union
    import random
    
    from pydantic import Field
    from inference.core.workflows.execution_engine.entities.base import (
      OutputDefinition,
      WorkflowImageData,
    )
    from inference.core.workflows.execution_engine.entities.types import (
        StepSelector,
        WorkflowImageSelector,
        StepOutputImageSelector,
    )
    from inference.core.workflows.execution_engine.v1.entities import FlowControl
    from inference.core.workflows.prototypes.block import (
        BlockResult,
        WorkflowBlock,
        WorkflowBlockManifest,
    )
    
    
    
    class BlockManifest(WorkflowBlockManifest):
        type: Literal["roboflow_core/random_continue@v1"]
        name: str
        image: Union[WorkflowImageSelector, StepOutputImageSelector] = ImageInputField
        probability: float
        next_steps: List[StepSelector] = Field(
            description="Reference to step which shall be executed if expression evaluates to true",
            examples=[["$steps.on_true"]],
        )
    
        @classmethod
        def describe_outputs(cls) -> List[OutputDefinition]:
            return []
    
        @classmethod
        def get_execution_engine_compatibility(cls) -> Optional[str]:
            return ">=1.0.0,<2.0.0"
    
    
    class RandomContinueBlockV1(WorkflowBlock):
    
        @classmethod
        def get_manifest(cls) -> Type[WorkflowBlockManifest]:
            return BlockManifest
    
        def run(
            self,
            image: WorkflowImageData,
            probability: float,
            next_steps: List[str],
        ) -> BlockResult:
            if not next_steps or random.random() > probability:
                return FlowControl()
            return FlowControl(context=next_steps)
    ```

    * line `10` imports type annotation for step selector which will be used to 
    notify Execution Engine that the block controls the flow

    * line `14` imports `FlowControl` class which is the only viabe response from
    flow-control block

    * line `26` specifies `image` which is batch-oriented input making the block SIMD - 
    which means that for each element of images batch, block will make random choice on 
    flow-control - if not that input block would operate in non-SIMD mode

    * line `28` defines list of step selectors **which effectively turns the block into flow-control one**

    * lines `55` and `56` show how to construct output - `FlowControl` object accept context being `None`, `string` or 
    `list of strings` - `None` represent flow termination for the batch element, strings are expected to be selectors 
    for next steps, passed in input.

??? example "Implementation of flow-control non-SIMD block"
    
    Example provides and comments out implementation of random continue block

    ```{ .py linenums="1" hl_lines="9 11 24-27 50-51"}
    from typing import List, Literal, Optional, Type, Union
    import random
    
    from pydantic import Field
    from inference.core.workflows.execution_engine.entities.base import (
      OutputDefinition,
    )
    from inference.core.workflows.execution_engine.entities.types import (
        StepSelector,
    )
    from inference.core.workflows.execution_engine.v1.entities import FlowControl
    from inference.core.workflows.prototypes.block import (
        BlockResult,
        WorkflowBlock,
        WorkflowBlockManifest,
    )
    
    
    
    class BlockManifest(WorkflowBlockManifest):
        type: Literal["roboflow_core/random_continue@v1"]
        name: str
        probability: float
        next_steps: List[StepSelector] = Field(
            description="Reference to step which shall be executed if expression evaluates to true",
            examples=[["$steps.on_true"]],
        )
    
        @classmethod
        def describe_outputs(cls) -> List[OutputDefinition]:
            return []
    
        @classmethod
        def get_execution_engine_compatibility(cls) -> Optional[str]:
            return ">=1.0.0,<2.0.0"
    
    
    class RandomContinueBlockV1(WorkflowBlock):
    
        @classmethod
        def get_manifest(cls) -> Type[WorkflowBlockManifest]:
            return BlockManifest
    
        def run(
            self,
            probability: float,
            next_steps: List[str],
        ) -> BlockResult:
            if not next_steps or random.random() > probability:
                return FlowControl()
            return FlowControl(context=next_steps)
    ```

    * line `9` imports type annotation for step selector which will be used to 
    notify Execution Engine that the block controls the flow

    * line `11` imports `FlowControl` class which is the only viabe response from
    flow-control block

    * lines `24-27` defines list of step selectors **which effectively turns the block into flow-control one**

    * lines `50` and `51` show how to construct output - `FlowControl` object accept context being `None`, `string` or 
    `list of strings` - `None` represent flow termination for the batch element, strings are expected to be selectors 
    for next steps, passed in input.

## Nested selectors

Some block will require list of selectors or dictionary of selectors to be 
provided in block manifest field. Use cases that may require this are outlined below

### Fusion of predictions from variable number of models

Let's assume that you want to build a block to get majority vote on multiple classifiers predictions - then you would 
like your run method to look like that:

```python
# pseud-code here
def run(self, predictions: List[dict]) -> BlockResult:
    predicted_classes = [p["class"] for p in predictions]
    counts = Counter(predicted_classes)
    return {"top_class": counts.most_common(1)[0]}
```
