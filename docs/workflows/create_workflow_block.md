# Creating Workflow blocks

Workflows blocks development requires an understanding of the
Workflow Ecosystem. Before diving deeper into the details, let's summarize the 
required knowledge:

Understanding of [Workflow execution](/workflows/workflow_execution.md), in particular:
    
* what is the relation of Workflow blocks and steps in Workflow definition

* how Workflow blocks and their manifests are used by [Workflows Compiler](/workflows/workflows_compiler.md)

* what is the `dimensionality level` of batch-oriented data passing through Workflow

* how [Execution Engine](/workflows/workflows_execution_engine.md) interacts with step, regarding 
its inputs and outputs

* what is the nature and role of [Workflow `kinds`](/workflows/kinds/index.md)

* understanding how [`pydantic`](https://docs.pydantic.dev/latest/) works

## Environment setup

As you will soon see, creating a Workflow block is simply a matter of defining a Python class that implements 
a specific interface. This design allows you to run the block using the Python interpreter, just like any 
other Python code. However, you may encounter difficulties when assembling all the required inputs, which would 
normally be provided by other blocks during Workflow execution. Therefore, it's important to set up the development 
environment properly for a smooth workflow. We recommend following these steps as part of the standard development 
process (initial steps can be skipped for subsequent contributions):

1. **Set up the `conda` environment** and install main dependencies of `inference`, as described in
[`inference` contributor guide](https://github.com/roboflow/inference/blob/main/CONTRIBUTING.md).

2. **Familiarize yourself with the organization of the Workflows codebase.**

    ??? "Workflows codebase structure - cheatsheet"
    
        Below are the key packages and directories in the Workflows codebase, along with their descriptions:
    
        * `inference/core/workflows` - the main package for Workflows.
    
        * `inference/core/workflows/core_steps` - contains Workflow blocks that are part of the Roboflow Core plugin. At the top levels, you'll find block categories, and as you go deeper, each block has its own package, with modules hosting different versions, starting from `v1.py`
    
        * `inference/core/workflows/execution_engine` - contains the Execution Engine. You generally won’t need to modify this package unless contributing to Execution Engine functionality.
    
        * `tests/workflows/` - the root directory for Workflow tests
      
        * `tests/workflows/unit_tests/` - suites of unit tests for the Workflows Execution Engine and core blocks. This is where you can test utility functions used in your blocks.
    
        * `tests/workflows/integration_tests/` - suites of integration tests for the Workflows Execution Engine and core blocks. You can run end-to-end (E2E) tests of your workflows in combination with other blocks here.


3. **Create a minimalistic block** – You’ll learn how to do this in the following sections. Start by implementing a simple block manifest and basic logic to ensure the block runs as expected.

4. **Add the block to the plugin** – Once your block is created, add it to the list of blocks exported from the plugin. If you're adding the block to the Roboflow Core plugin, make sure to add an entry for your block in the
[loader.py](https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/loader.py). **If you forget this step, your block won’t be visible!**

5. **Iterate and refine your block** – Continue developing and running your block until you’re satisfied with the results. The sections below explain how to iterate on your block in various scenarios.

### Running your blocks using Workflows UI

We recommend running the inference server with a mounted volume (which is much faster than re-building `inference` 
server on each change):

```bash
inference_repo$ docker run -p 9001:9001 \
   -v ./inference:/app/inference \
   roboflow/roboflow-inference-server-cpu:latest
```
and connecting your local server to Roboflow UI:

<div align="center"><img src="https://media.roboflow.com/inference/workflows_connect_your_local_server.png" width="80%"/></div>

to quickly run previews:

<div align="center"><img src="https://media.roboflow.com/inference/workflow_preview.png"/></div>
  
??? Note "My block requires extra dependencies - I cannot use pre-built `inference` server"

    It's natural that your blocks may sometimes require additional dependencies. To add a dependency, simply include it in one of the 
    [requirements files](https://github.com/roboflow/inference/tree/main/requirements)hat are installed in the relevant Docker image 
    (usually the
    [CPU build](https://github.com/roboflow/inference/blob/main/docker/dockerfiles/Dockerfile.onnx.cpu) 
    of the `inference` server).

    Afterward, run:
      
    ```{ .bash linenums="1" hl_lines="3"}
    inference_repo$ docker build \
       -t roboflow/roboflow-inference-server-cpu:test \ 
       -f docker/dockerfiles/Dockerfile.onnx.cpu .
    ```

    You can then run your local build by specifying the test tag you just created:

    ```{ .bash linenums="1" hl_lines="3"}
    inference_repo$ inference_repo$ docker run -p 9001:9001 \
       -v ./inference:/app/inference \
       roboflow/roboflow-inference-server-cpu:test
    ```

### Running your blocks without Workflows UI

For contributors without access to the Roboflow platform, we recommend running the server as mentioned in the 
section above. However, instead of using the UI editor, you will need to create a simple Workflow definition and 
send a request to the server.
    
??? Note "Running your Workflow without UI"

    The following code snippet demonstrates how to send a request to the `inference` server to run a Workflow. 
    The `inference_sdk` is included with the `inference` package as a lightweight client library for our server.

    ```python
    from inference_sdk import InferenceHTTPClient

    YOUR_WORKFLOW_DEFINITION = ...
    
    client = InferenceHTTPClient(
        api_url=object_detection_service_url,
        api_key="XXX",  # optional, only required if Workflow uses Roboflow Platform
    )
    result = client.run_workflow(
        specification=YOUR_WORKFLOW_DEFINITION,
        images={
            "image": your_image_np,   # this is example input, adjust it
        },
        parameters={
            "my_parameter": 37,   # this is example input, adjust it
        },
    )
    ```

### Recommended way for regular contributors


Creating integration tests in the `tests/workflows/integration_tests/execution` directory is a natural part of the 
development iteration process. This approach allows you to develop and test simultaneously, providing valuable 
feedback as you refine your code. Although it requires some experience, it significantly enhances 
long-term code maintenance.

The process is straightforward:

1. **Create a New Test Module:** For example, name it `test_workflows_with_my_custom_block.py`.

2. **Develop Example Workflows**: Create one or more example Workflows. It would be best if your block cooperates 
with other blocks from the ecosystem. 

3. **Run Tests with Sample Data:** Execute these Workflows in your tests using sample data 
(you can explore our 
[fixtures](https://github.com/roboflow/inference/blob/main/tests/workflows/integration_tests/execution/conftest.py)
to find example data we usually use).

4. **Assert Expected Results:** Validate that the results match your expectations.

By incorporating testing into your development flow, you ensure that your block remains stable over time and 
effectively interacts with existing blocks, enhancing the expressiveness of your work!

You can run your test using the following command:

```bash
pytest tests/workflows/integration_tests/execution/test_workflows_with_my_custom_block
```
Feel free to reference other tests for examples or use the following template:

    
??? Note "Integration test template"

    ```{ .py linenums="1" hl_lines="2 3 4 7-11 19-23"}
    def test_detection_plus_classification_workflow_when_XXX(
        model_manager: ModelManager,
        dogs_image: np.ndarray, 
        roboflow_api_key: str,
    ) -> None:
        # given
        workflow_init_parameters = {
            "workflows_core.model_manager": model_manager,
            "workflows_core.api_key": roboflow_api_key,
            "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
        }
        execution_engine = ExecutionEngine.init(
            workflow_definition=<YOUR-EXAMPLE-WORKLFOW>,
            init_parameters=workflow_init_parameters,
            max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
        )
    
        # when
        result = execution_engine.run(
            runtime_parameters={
                "image": dogs_image,
            }
        )
      
        # then
        assert isinstance(result, list), "Expected list to be delivered"
        assert len(result) == 1, "Expected 1 element in the output for one input image"
        assert set(result[0].keys()) == {
            "predictions",
        }, "Expected all declared outputs to be delivered"
        assert (
            len(result[0]["predictions"]) == 2
        ), "Expected 2 dogs crops on input image, hence 2 nested classification results"
        assert [result[0]["predictions"][0]["top"], result[0]["predictions"][1]["top"]] == [
            "116.Parson_russell_terrier",
            "131.Wirehaired_pointing_griffon",
        ], "Expected predictions to be as measured in reference run"
    ```

    * In line `2`, you’ll find the `model_manager` fixture, which is typically required by model blocks. This fixture provides the `ModelManager` abstraction from `inference`, used for loading and unloading models.

    * Line `3` defines a fixture that includes an image of two dogs (explore other fixtures to find more example images).

    * Line `4` is an optional fixture you may want to use if any of the blocks in your tested workflow require a Roboflow API key. If that’s the case, export the `ROBOFLOW_API_KEY` environment variable with a valid key before running the test.

    * Lines `7-11` provide the setup for the initialization parameters of the blocks that the Execution Engine will create at runtime, based on your Workflow definition.

    * Lines `19-23` demonstrate how to run a Workflow by injecting input parameters. Please ensure that the keys in runtime_parameters match the inputs declared in your Workflow definition.

    * Starting from line `26`, you’ll find example assertions within the test.


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

!!! warning "Understanding internal data representation"

      You may have noticed that we recommend importing the `Batch` and `WorkflowImageData` classes, which are 
      fundamental components used when constructing building blocks in our system. For a deeper understanding of 
      how these classes fit into the overall architecture, we encourage you to refer to the 
      [Data Representations](/workflows/internal_data_types.md) page for more detailed information. 


## Block manifest

A manifest is a crucial component of a Workflow block that defines a prototype 
for step declaration that can be placed in a Workflow definition to use the block. 
In particular, it: 

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
stability. For more details, see [versioning](/workflows/versioning.md).

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

This is the minimal representation of a manifest. It defines two special fields that are important for 
Compiler and Execution engine:

* `type` - required to parse syntax of Workflows definitions based on dynamic pool of blocks - this is the 
[`pydantic` type discriminator](https://docs.pydantic.dev/latest/concepts/unions/#discriminated-unions) that lets the Compiler understand which block manifest is to be verified when 
parsing specific steps in a Workflow definition

* `name` - this property will be used to give the step a unique name and let other steps selects it via selectors

### Adding inputs

We want our step to take two inputs with images to be compared.

??? example "Adding inputs"
    
    Let's see how to add definitions of those inputs to manifest: 

    ```{ .py linenums="1" hl_lines="2 6-9 20-25"}
    from typing import Literal, Union
    from pydantic import Field
    from inference.core.workflows.prototypes.block import (
        WorkflowBlockManifest,
    )
    from inference.core.workflows.execution_engine.entities.types import (
        Selector,
        IMAGE_KIND,
    )
    
    
    class ImagesSimilarityManifest(WorkflowBlockManifest):
        type: Literal["my_plugin/images_similarity@v1"] 
        name: str
        # all properties apart from `type` and `name` are treated as either 
        # hardcoded parameters or data selectors. Data selectors are strings 
        # that start from `$steps.` or `$inputs.` marking references for data 
        # available in runtime - in this case we usually specify kinds of data
        # to let compiler know what we expect the data to look like.
        image_1: Selector(kind=[IMAGE_KIND]) = Field(
            description="First image to calculate similarity",
        )
        image_2: Selector(kind=[IMAGE_KIND]) = Field(
            description="Second image to calculate similarity",
        )
    ```
    
    * in the lines `2-9`, we've added a couple of imports to ensure that we have everything needed
    
    * line `20` defines `image_1` parameter - as manifest is prototype for Workflow Definition, 
    the only way to tell about image to be used by step is to provide selector - we have 
    a specialised type in core library that can be used - `Selector`.
    If you look deeper into codebase, you will discover this is type alias constructor function - telling `pydantic`
    to expect string matching `$inputs.{name}` and `$steps.{name}.*` patterns respectively, additionally providing 
    extra schema field metadata that tells Workflows ecosystem components that the `kind` of data behind selector is 
    [image](/workflows/kinds/image.md). **important note:** we denote *kind* as list - the list of specific kinds 
    is interpreted as *union of kinds* by Execution Engine.
  
    * denoting `pydantic` `Field(...)` attribute in the last parts of line `20` is optional, yet appreciated, 
    especially for blocks intended to cooperate with Workflows UI 
  
    * starting in line `23`, you can find definition of `image_2` parameter which is very similar to `image_1`.


Such definition of manifest can handle the following step declaration in Workflow definition:

```json
{
  "type": "my_plugin/images_similarity@v1",
  "name": "my_step",
  "image_1": "$inputs.my_image",
  "image_2": "$steps.image_transformation.image"
}
```

This definition will make the Compiler and Execution Engine:

* initialize the step from Workflow block declaring type `my_plugin/images_similarity@v1`

* supply two parameters for the steps run method:

  * `input_1` of type `WorkflowImageData` which will be filled with image submitted as Workflow execution input 
  named `my_image`.
  
  * `imput_2` of type `WorkflowImageData` which will be generated at runtime, by another step called 
  `image_transformation`


### Adding parameters to the manifest

Let's now add the parameter that will influence step execution.

??? example "Adding parameter to the manifest"

    ```{ .py linenums="1" hl_lines="9 27-33"}
    from typing import Literal, Union
    from pydantic import Field
    from inference.core.workflows.prototypes.block import (
        WorkflowBlockManifest,
    )
    from inference.core.workflows.execution_engine.entities.types import (
        Selector,
        IMAGE_KIND,
        FLOAT_ZERO_TO_ONE_KIND,
    )
    
    
    class ImagesSimilarityManifest(WorkflowBlockManifest):
        type: Literal["my_plugin/images_similarity@v1"] 
        name: str
        # all properties apart from `type` and `name` are treated as either 
        # hardcoded parameters or data selectors. Data selectors are strings 
        # that start from `$steps.` or `$inputs.` marking references for data 
        # available in runtime - in this case we usually specify kinds of data
        # to let compiler know what we expect the data to look like.
        image_1: Selector(kind=[IMAGE_KIND]) = Field(
            description="First image to calculate similarity",
        )
        image_2: Selector(kind=[IMAGE_KIND]) = Field(
            description="Second image to calculate similarity",
        )
        similarity_threshold: Union[
            float,
            Selector(kind=[FLOAT_ZERO_TO_ONE_KIND]),
        ] = Field(
            default=0.4,
            description="Threshold to assume that images are similar",
        )
    ```
  
    * line `9` imports [`float_zero_to_one`](/workflows/kinds/float_zero_to_one.md) `kind` 
      definition which will be used to define the parameter.
  
    * in line `27` we start defining parameter called `similarity_threshold`. Manifest will accept 
    either float values or selector to workflow input of `kind`
    [`float_zero_to_one`](/workflows/kinds/float_zero_to_one.md), imported in line `9`.

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
  "similarity_threshold": 0.5
}
```

### Declaring block outputs

We have successfully defined inputs for our block, but we are still missing couple of elements required to 
successfully run blocks. Let's define block outputs.

??? example "Declaring block outputs"

    Minimal set of information required is outputs description. Additionally, 
    to increase block stability, we advise to provide information about execution engine 
    compatibility.
    
    ```{ .py linenums="1" hl_lines="5 11 32-39 41-43"}
    from typing import Literal, Union
    from pydantic import Field
    from inference.core.workflows.prototypes.block import (
        WorkflowBlockManifest,
        OutputDefinition,
    )
    from inference.core.workflows.execution_engine.entities.types import (
        Selector,
        IMAGE_KIND,
        FLOAT_ZERO_TO_ONE_KIND,
        BOOLEAN_KIND,
    )
    
    
    class ImagesSimilarityManifest(WorkflowBlockManifest):
        type: Literal["my_plugin/images_similarity@v1"] 
        name: str
        image_1: Selector(kind=[IMAGE_KIND]) = Field(
            description="First image to calculate similarity",
        )
        image_2: Selector(kind=[IMAGE_KIND]) = Field(
            description="Second image to calculate similarity",
        )
        similarity_threshold: Union[
            float,
            Selector(kind=[FLOAT_ZERO_TO_ONE_KIND]),
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
            return ">=1.3.0,<2.0.0"
    ```

    * line `5` imports class that is used to describe step outputs
  
    * line `11` imports [`boolean`](/workflows/kinds/boolean.md) `kind` to be used 
    in outputs definitions
  
    * lines `32-39` declare class method to specify outputs from the block - 
    each entry in list declare one return property for each batch element and its `kind`.
    Our block will return boolean flag `images_match` for each pair of images.
  
    * lines `41-43` declare compatibility of the block with Execution Engine -
    see [versioning page](/workflows/versioning.md) for more details

As a result of those changes:

* Execution Engine would understand that steps created based on this block 
are supposed to deliver specified outputs and other steps can refer to those outputs
in their inputs

* the blocks loading mechanism will not load the block given that Execution Engine is not in version `v1`

??? hint "LEARN MORE: Dynamic outputs"

    Some blocks may not be able to arbitrailry define their outputs using 
    classmethod - regardless of the content of step manifest that is available after 
    parsing. To support this we introduced the following convention:

    * classmethod `describe_outputs(...)` shall return list with one element of 
    name `*` and kind `*` (aka `WILDCARD_KIND`)

    * additionally, block manifest should implement instance method `get_actual_outputs(...)`
    that provides list of actual outputs that can be generated based on filled manifest data 

    ```{ .py linenums="1" hl_lines="13 35-42 44-49"}
    from typing import Literal, Union, List, Optional
    from pydantic import Field
    from inference.core.workflows.prototypes.block import (
        WorkflowBlockManifest,
        OutputDefinition,
    )
    from inference.core.workflows.execution_engine.entities.types import (
        Selector,
        IMAGE_KIND,
        FloatZeroToOne,
        FLOAT_ZERO_TO_ONE_KIND,
        BOOLEAN_KIND,
        WILDCARD_KIND,
    )
    
    
    class ImagesSimilarityManifest(WorkflowBlockManifest):
        type: Literal["my_plugin/images_similarity@v1"] 
        name: str
        image_1: Selector(kind=[IMAGE_KIND]) = Field(
            description="First image to calculate similarity",
        )
        image_2: Selector(kind=[IMAGE_KIND]) = Field(
            description="Second image to calculate similarity",
        )
        similarity_threshold: Union[
            float,
            Selector(kind=[FLOAT_ZERO_TO_ONE_KIND]),
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

At this stage, the manifest of our simple block is ready, we will continue 
with our example. You can check out the [advanced topics](#advanced-topics) section for more details that would just 
be a distractions now.

### Base implementation

Having the manifest ready, we can prepare baseline implementation of the 
block.

??? example "Block scaffolding"

    ```{ .py linenums="1" hl_lines="1 5 6 8-11 53-55 57-63"}
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
        Selector,
        IMAGE_KIND,
        FloatZeroToOne,
        FLOAT_ZERO_TO_ONE_KIND,
        BOOLEAN_KIND,
    )
    
    class ImagesSimilarityManifest(WorkflowBlockManifest):
        type: Literal["my_plugin/images_similarity@v1"] 
        name: str
        image_1: Selector(kind=[IMAGE_KIND]) = Field(
            description="First image to calculate similarity",
        )
        image_2: Selector(kind=[IMAGE_KIND]) = Field(
            description="Second image to calculate similarity",
        )
        similarity_threshold: Union[
            FloatZeroToOne,
            Selector(kind=[FLOAT_ZERO_TO_ONE_KIND]),
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
            return ">=1.3.0,<2.0.0"
    
        
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

    * lines `1`, `5-6` and `8-11` added changes into import surtucture to 
    provide additional symbols required to properly define block class and all
    of its methods signatures

    * lines `53-55` defines class method `get_manifest(...)` to simply return 
    the manifest class we cretaed earlier

    * lines `57-63` define `run(...)` function, which Execution Engine
    will invoke with data to get desired results. Please note that 
    manifest fields defining inputs of [image](/workflows/kinds/image.md) kind
    are marked as `WorkflowImageData` - which is compliant with intenal data 
    representation of `image` kind described in [kind documentation](/workflows/kinds/image.md).

### Providing implementation for block logic

Let's now add an example implementation of  the `run(...)` method to our block, such that
it can produce meaningful results.

!!! Note
    
    The Content of this section is supposed to provide examples on how to interact 
    with the Workflow ecosystem as block creator, rather than providing robust 
    implementation of the block.

??? example "Implementation of `run(...)` method"

    ```{ .py linenums="1" hl_lines="3 55-57 69-80"}
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
        Selector,
        IMAGE_KIND,
        FloatZeroToOne,
        FLOAT_ZERO_TO_ONE_KIND,
        BOOLEAN_KIND,
    )
    
    class ImagesSimilarityManifest(WorkflowBlockManifest):
        type: Literal["my_plugin/images_similarity@v1"] 
        name: str
        image_1: Selector(kind=[IMAGE_KIND]) = Field(
            description="First image to calculate similarity",
        )
        image_2: Selector(kind=[IMAGE_KIND]) = Field(
            description="Second image to calculate similarity",
        )
        similarity_threshold: Union[
            FloatZeroToOne,
            Selector(kind=[FLOAT_ZERO_TO_ONE_KIND]),
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
            return ">=1.3.0,<2.0.0"
    
        
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

    * lines `55-57` defines block constructor, thanks to this - state of block 
    is initialised once and live through consecutive invocation of `run(...)` method - for 
    instance when Execution Engine runs on consecutive frames of video

    * lines `69-80` provide implementation of block functionality - the details are trully not
    important regarding Workflows ecosystem, but there are few details you should focus:
    
        * lines `69` and `70` make use of `WorkflowImageData` abstraction, showcasing how 
        `numpy_image` property can be used to get `np.ndarray` from internal representation of images
        in Workflows. We advise to expole remaining properties of `WorkflowImageData` to discover more.

        * result of workflow block execution, declared in lines `78-80` is in our case just a dictionary 
        **with the keys being the names of outputs declared in manifest**, in line `43`. Be sure to provide all
        declared outputs - otherwise Execution Engine will raise error.


## Exposing block in `plugin`

Now, your block is ready to be used, but Execution Engine is not aware of its existence. This is because no registered 
plugin exports the block you just created. Details of blocks bundling are be covered in [separate page](/workflows/blocks_bundling.md), 
but the remaining thing to do is to add block class into list returned from your plugins' `load_blocks(...)` function:

```python
# __init__.py of your plugin (or roboflow_core plugin if you contribute directly to `inference`)

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

    ```{ .py linenums="1" hl_lines="13 40-42 70-71 74-77 85-86"}
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
        Selector,
        IMAGE_KIND,
        FloatZeroToOne,
        FLOAT_ZERO_TO_ONE_KIND,
        BOOLEAN_KIND,
    )
    
    class ImagesSimilarityManifest(WorkflowBlockManifest):
        type: Literal["my_plugin/images_similarity@v1"] 
        name: str
        image_1: Selector(kind=[IMAGE_KIND]) = Field(
            description="First image to calculate similarity",
        )
        image_2: Selector(kind=[IMAGE_KIND]) = Field(
            description="Second image to calculate similarity",
        )
        similarity_threshold: Union[
            FloatZeroToOne,
            Selector(kind=[FLOAT_ZERO_TO_ONE_KIND]),
        ] = Field(
            default=0.4,
            description="Threshold to assume that images are similar",
        )

        @classmethod
        def get_parameters_accepting_batches(cls) -> bool:
            return ["image_1", "image_2"]
        
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
            return ">=1.3.0,<2.0.0"
    
        
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

    * lines `40-42` define class method that changes default behaviour of the block and make it capable 
    to process batches - we are marking each parameter that the `run(...)` method **recognizes as batch-oriented**. 

    * changes introduced above made the signature of `run(...)` method to change, now `image_1` and `image_2`
    are not instances of `WorkflowImageData`, but rather batches of elements of this type. **Important note:** 
    having multiple batch-oriented parameters we expect that those batches would have the elements related to
    each other at corresponding positions - such that our block comparing `image_1[1]` into `image_2[1]` actually
    performs logically meaningful operation.

    * lines `74-77`, `85-86` present changes that needed to be introduced to run processing across all batch 
    elements - showcasing how to iterate over batch elements if needed

    * it is important to note how outputs are constructed in line `85` - each element of batch will be given
    its entry in the list which is returned from `run(...)` method. Order must be aligned with order of batch 
    elements. Each output dictionary must provide all keys declared in block outputs.


??? Warning "Inputs that accept both batches and scalars"

    It is **relatively unlikely**, but may happen that your block would need to accept both batch-oriented data
    and scalars within a single input parameter. Execution Engine recognises that using 
    `get_parameters_accepting_batches_and_scalars(...)` method of block manifest. Take a look at the 
    example provided below:


    ```{ .py linenums="1" hl_lines="20-22 24-26 45-47 49 50-54 65-70"}
    from typing import Literal, Union, List, Optional, Type, Any, Dict
    from pydantic import Field
    
    from inference.core.workflows.prototypes.block import (
        WorkflowBlockManifest,
        WorkflowBlock,
        BlockResult,
    )
    from inference.core.workflows.execution_engine.entities.base import (
        OutputDefinition,
        Batch,
    )
    from inference.core.workflows.execution_engine.entities.types import (
        Selector,
    )
    
    class ExampleManifest(WorkflowBlockManifest):
        type: Literal["my_plugin/example@v1"] 
        name: str
        param_1: Selector()
        param_2: List[Selector()]
        param_3: Dict[str, Selector()]

        @classmethod
        def get_parameters_accepting_batches_and_scalars(cls) -> bool:
            return ["param_1", "param_2", "param_3"]
        
        @classmethod
        def describe_outputs(cls) -> List[OutputDefinition]:
            return [OutputDefinition(name="dummy")]
    
        @classmethod
        def get_execution_engine_compatibility(cls) -> Optional[str]:
            return ">=1.3.0,<2.0.0"
    
        
    class ExampleBlock(WorkflowBlock):

        @classmethod
        def get_manifest(cls) -> Type[WorkflowBlockManifest]:
            return ExampleManifest
    
        def run(
            self,
            param_1: Any,
            param_2: List[Any],
            param_3: Dict[str, Any],
        ) -> BlockResult:
            batch_size = None
            if isinstance(param_1, Batch):
                param_1_result = ...  # do something with batch-oriented param
                batch_size = len(param_1)
            else:
                param_1_result = ... # do something with scalar param
            for element in param_2:
               if isinstance(element, Batch):
                  ...
               else:
                  ...
            for key, value in param_3.items():
               if isinstance(element, value):
                  ...
               else:
                  ...
            if batch_size is None:
               return {"dummy": "some_result"}
            result = []
            for _ in range(batch_size):
               result.append({"dummy": "some_result"})
            return result
    ```

    * lines `20-22` specify manifest parameters that are expected to accept mixed (both scalar and batch-oriented)
    input data - point out that at this stage there is no difference in definition compared to previous examples.

    * lines `24-26` specify `get_parameters_accepting_batches_and_scalars(...)` method to tell the Execution 
    Engine that block `run(...)` method can handle both scalar and batch-oriented inputs for the specified 
    parameters.

    * lines `45-47` depict the parameters of mixed nature in `run(...)` method signature.

    * line `49` reveals that we must keep track of the expected output size **within the block logic**. That's 
    why it is quite tricky to implement blocks with mixed inputs. Normally, when block `run(...)` method 
    operates on scalars - in majority of cases (exceptions will be described below) - the metod constructs 
    single output dictionary. Similairly, when batch-oriented inputs are accepted - those inputs 
    define expected output size. In this case, however, we must manually detect batches and catch their sizes.

    * lines `50-54` showcase how we usually deal with mixed parameters - applying different logic when 
    batch-oriented data is detected

    * as mentioned earlier, output construction must also be adjusted to the nature of mixed inputs - which 
    is illustrated in lines `65-70`

### Implementation of flow-control block

Flow-control blocks differs quite substantially from other blocks that just process the data. Here we will show 
how to create a flow control block, but first - a little bit of theory:

* flow-control block is the block that declares compatibility with step selectors in their manifest (selector to step
is defined as `$steps.{step_name}` - similar to step output selector, but without specification of output name)

* flow-control blocks cannot register outputs, they are meant to return `FlowControl` objects

* `FlowControl` object specify next steps (from selectors provided in step manifest) that for given 
batch element (SIMD flow-control) or whole workflow execution (non-SIMD flow-control) should pick up next

??? example "Implementation of flow-control"
    
    Example provides and comments out implementation of random continue block

    ```{ .py linenums="1" hl_lines="10 14 28-31 55-56"}
    from typing import List, Literal, Optional, Type, Union
    import random
    
    from pydantic import Field
    from inference.core.workflows.execution_engine.entities.base import (
      OutputDefinition,
      WorkflowImageData,
    )
    from inference.core.workflows.execution_engine.entities.types import (
        StepSelector,
        Selector,
        IMAGE_KIND,
    )
    from inference.core.workflows.execution_engine.v1.entities import FlowControl
    from inference.core.workflows.prototypes.block import (
        BlockResult,
        WorkflowBlock,
        WorkflowBlockManifest,
    )
    
    
    
    class BlockManifest(WorkflowBlockManifest):
        type: Literal["my_plugin/random_continue@v1"]
        name: str
        image: Selector(kind=[IMAGE_KIND]) = ImageInputField
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
            return ">=1.2.0,<2.0.0"
    
    
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

    * line `14` imports `FlowControl` class which is the only viable response from
    flow-control block

    * line `28` defines list of step selectors **which effectively turns the block into flow-control one**

    * lines `55` and `56` show how to construct output - `FlowControl` object accept context being `None`, `string` or 
    `list of strings` - `None` represent flow termination for the batch element, strings are expected to be selectors 
    for next steps, passed in input.

??? example "Implementation of flow-control - batch variant"
    
    Example provides and comments out implementation of random continue block

    ```{ .py linenums="1" hl_lines="8 11 15 29-32 38-40 55 59 60 61-63"}
    from typing import List, Literal, Optional, Type, Union
    import random
    
    from pydantic import Field
    from inference.core.workflows.execution_engine.entities.base import (
      OutputDefinition,
      WorkflowImageData,
      Batch,
    )
    from inference.core.workflows.execution_engine.entities.types import (
        StepSelector,
        Selector,
        IMAGE_KIND,
    )
    from inference.core.workflows.execution_engine.v1.entities import FlowControl
    from inference.core.workflows.prototypes.block import (
        BlockResult,
        WorkflowBlock,
        WorkflowBlockManifest,
    )
    
    
    
    class BlockManifest(WorkflowBlockManifest):
        type: Literal["my_plugin/random_continue@v1"]
        name: str
        image: Selector(kind=[IMAGE_KIND]) = ImageInputField
        probability: float
        next_steps: List[StepSelector] = Field(
            description="Reference to step which shall be executed if expression evaluates to true",
            examples=[["$steps.on_true"]],
        )
    
        @classmethod
        def describe_outputs(cls) -> List[OutputDefinition]:
            return []

        @classmethod
        def get_parameters_accepting_batches(cls) -> List[str]:
            return ["image"]
    
        @classmethod
        def get_execution_engine_compatibility(cls) -> Optional[str]:
            return ">=1.3.0,<2.0.0"
    
    
    class RandomContinueBlockV1(WorkflowBlock):
    
        @classmethod
        def get_manifest(cls) -> Type[WorkflowBlockManifest]:
            return BlockManifest
    
        def run(
            self,
            image: Batch[WorkflowImageData],
            probability: float,
            next_steps: List[str],
        ) -> BlockResult:
            result = []
            for _ in image:
               if not next_steps or random.random() > probability:
                   result.append(FlowControl())
               result.append(FlowControl(context=next_steps))
            return result
    ```

    * line `11` imports type annotation for step selector which will be used to 
    notify Execution Engine that the block controls the flow

    * line `15` imports `FlowControl` class which is the only viable response from
    flow-control block

    * lines `29-32` defines list of step selectors **which effectively turns the block into flow-control one**

    * lines `38-40` contain definition of `get_parameters_accepting_batches(...)` method telling Execution 
    Engine that block `run(...)` method expects batch-oriented `image` parameter.

    * line `59` revels that we need to return flow-control guide for each and every element of `image` batch.

    * to achieve that end, in line `60` we iterate over the contntent of batch.

    * lines `61-63` show how to construct output - `FlowControl` object accept context being `None`, `string` or 
    `list of strings` - `None` represent flow termination for the batch element, strings are expected to be selectors 
    for next steps, passed in input.

### Nested selectors

Some block will require list of selectors or dictionary of selectors to be 
provided in block manifest field. Version `v1` of Execution Engine supports only 
one level of nesting - so list of lists of selectors or dictionary with list of selectors 
will not be recognised properly.

Practical use cases showcasing usage of nested selectors are presented below.

#### Fusion of predictions from variable number of models

Let's assume that you want to build a block to get majority vote on multiple classifiers predictions - then you would 
like your run method to look like that:

```python
# pseud-code here
def run(self, predictions: List[dict]) -> BlockResult:
    predicted_classes = [p["class"] for p in predictions]
    counts = Counter(predicted_classes)
    return {"top_class": counts.most_common(1)[0]}
```

??? example "Nested selectors - models ensemble"

    ```{ .py linenums="1" hl_lines="23-26 50"}
    from typing import List, Literal, Optional, Type
    
    from pydantic import Field
    import supervision as sv
    from inference.core.workflows.execution_engine.entities.base import (
      OutputDefinition,
    )
    from inference.core.workflows.execution_engine.entities.types import (
        Selector,
        OBJECT_DETECTION_PREDICTION_KIND,
    )
    from inference.core.workflows.prototypes.block import (
        BlockResult,
        WorkflowBlock,
        WorkflowBlockManifest,
    )
    
    
    
    class BlockManifest(WorkflowBlockManifest):
        type: Literal["my_plugin/fusion_of_predictions@v1"]
        name: str
        predictions: List[Selector(kind=[OBJECT_DETECTION_PREDICTION_KIND])] = Field(
            description="Selectors to step outputs",
            examples=[["$steps.model_1.predictions", "$steps.model_2.predictions"]],
        )
    
        @classmethod
        def describe_outputs(cls) -> List[OutputDefinition]:
            return [
              OutputDefinition(
                name="predictions", 
                kind=[OBJECT_DETECTION_PREDICTION_KIND],
              )
            ]
    
        @classmethod
        def get_execution_engine_compatibility(cls) -> Optional[str]:
            return ">=1.3.0,<2.0.0"
    
    
    class FusionBlockV1(WorkflowBlock):
    
        @classmethod
        def get_manifest(cls) -> Type[WorkflowBlockManifest]:
            return BlockManifest
    
        def run(
            self,
            predictions: List[sv.Detections],
        ) -> BlockResult:
            merged = sv.Detections.merge(predictions)
            return {"predictions": merged}
    ```

    * lines `23-26` depict how to define manifest field capable of accepting 
    list of selectors

    * line `50` shows what to expect as input to block's `run(...)` method - 
    list of objects which are representation of specific kind. If the block accepted 
    batches, the input type of `predictions` field would be `List[Batch[sv.Detections]`

Such block is compatible with the following step declaration:

```{ .json linenums="1" hl_lines="4-7"}
{
  "type": "my_plugin/fusion_of_predictions@v1",
  "name": "my_step",
  "predictions": [
    "$steps.model_1.predictions",
    "$steps.model_2.predictions"  
  ]
}
```

#### Block with data transformations allowing dynamic parameters

Occasionally, blocks may need to accept group of "named" selectors, 
which names and values are to be defined by creator of Workflow definition. 
In such cases, block manifest shall accept dictionary of selectors, where
keys serve as names for those selectors.

??? example "Nested selectors - named selectors"

    ```{ .py linenums="1" hl_lines="22-25 46"}
    from typing import List, Literal, Optional, Type, Any
    
    from pydantic import Field
    import supervision as sv
    from inference.core.workflows.execution_engine.entities.base import (
      OutputDefinition,
    )
    from inference.core.workflows.execution_engine.entities.types import (
        Selector
    )
    from inference.core.workflows.prototypes.block import (
        BlockResult,
        WorkflowBlock,
        WorkflowBlockManifest,
    )
    
    
    
    class BlockManifest(WorkflowBlockManifest):
        type: Literal["my_plugin/named_selectors_example@v1"]
        name: str
        data: Dict[str, Selector()] = Field(
            description="Selectors to step outputs",
            examples=[{"a": $steps.model_1.predictions", "b": "$Inputs.data"}],
        )
    
        @classmethod
        def describe_outputs(cls) -> List[OutputDefinition]:
            return [
              OutputDefinition(name="my_output", kind=[]),
            ]
    
        @classmethod
        def get_execution_engine_compatibility(cls) -> Optional[str]:
            return ">=1.3.0,<2.0.0"
    
    
    class BlockWithNamedSelectorsV1(WorkflowBlock):
    
        @classmethod
        def get_manifest(cls) -> Type[WorkflowBlockManifest]:
            return BlockManifest
    
        def run(
            self,
            data: Dict[str, Any],
        ) -> BlockResult:
            ...
            return {"my_output": ...}
    ```

    * lines `22-25` depict how to define manifest field capable of accepting 
    dictionary of selectors - providing mapping between selector name and value

    * line `46` shows what to expect as input to block's `run(...)` method - 
    dict of objects which are reffered with selectors. If the block accepted 
    batches, the input type of `data` field would be `Dict[str, Union[Batch[Any], Any]]`.
    In non-batch cases, non-batch-oriented data referenced by selector is automatically 
    broadcasted, whereas for blocks accepting batches - `Batch` container wraps only 
    batch-oriented inputs, with other inputs being passed as singular values.

Such block is compatible with the following step declaration:

```{ .json linenums="1" hl_lines="4-7"}
{
  "type": "my_plugin/named_selectors_example@v1",
  "name": "my_step",
  "data": {
    "a": "$steps.model_1.predictions",
    "b": "$inputs.my_parameter"  
  }
}
```

Practical implications will be the following:

* under `data["a"]` inside `run(...)` you will be able to find model's predictions - 
like `sv.Detections` if `model_1` is object-detection model

* under `data["b"]` inside `run(...)`, you will find value of input parameter named `my_parameter`

### Inputs and output dimensionality vs `run(...)` method

The dimensionality of block inputs plays a crucial role in shaping the `run(...)` method’s signature, and that's 
why the system enforces strict bounds on the differences in dimensionality levels between inputs 
(with the maximum allowed difference being `1`). This restriction is critical for ensuring consistency and 
predictability when writing blocks.

If dimensionality differences weren't controlled, it would be difficult to predict the structure of 
the `run(...)` method, making development harder and less reliable. That’s why validation of this property 
is strictly enforced during the Workflow compilation process.

Similarly, the output dimensionality also affects the method signature and the format of the expected output. 
The ecosystem supports the following scenarios:

* all inputs have **the same dimensionality** and outputs **does not change** dimensionality - baseline case

* all inputs have **the same dimensionality** and output **decreases** dimensionality

* all inputs have **the same dimensionality** and output **increases** dimensionality

* inputs have **different dimensionality** and output is allowed to keep the dimensionality of 
**reference input**

Other combinations of input/output dimensionalities are not allowed to ensure consistency and to prevent ambiguity in 
the method signatures.

??? example "Impact of dimensionality on `run(...)` method - batches disabled"

    === "output dimensionality increase"

        In this example, we perform dynamic crop of image based on predictions.

        ```{ .py linenums="1" hl_lines="28-30 63 64-65"}
        from typing import Dict, List, Literal, Optional, Type, Union
        from uuid import uuid4

        from inference.core.workflows.execution_engine.constants import DETECTION_ID_KEY
        from inference.core.workflows.execution_engine.entities.base import (
            OutputDefinition,
            WorkflowImageData,
            ImageParentMetadata,
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
            type: Literal["my_block/dynamic_crop@v1"]
            image: Selector(kind=[IMAGE_KIND])
            predictions: Selector(
                kind=[OBJECT_DETECTION_PREDICTION_KIND],
            )
        
            @classmethod
            def get_output_dimensionality_offset(cls) -> int:
                return 1
        
            @classmethod
            def describe_outputs(cls) -> List[OutputDefinition]:
                return [
                    OutputDefinition(name="crops", kind=[IMAGE_KIND]),
                ]
        
            @classmethod
            def get_execution_engine_compatibility(cls) -> Optional[str]:
                return ">=1.3.0,<2.0.0"

        class DynamicCropBlockV1(WorkflowBlock):

            @classmethod
            def get_manifest(cls) -> Type[WorkflowBlockManifest]:
                return BlockManifest
            
            def run(
                self,
                image: WorkflowImageData,
                predictions: sv.Detections,
            ) -> BlockResult:
                crops = []
                for (x_min, y_min, x_max, y_max) in predictions.xyxy.round().astype(dtype=int):
                    cropped_image = image.numpy_image[y_min:y_max, x_min:x_max]
                    parent_metadata = ImageParentMetadata(parent_id=f"{uuid4()}")
                    if cropped_image.size:
                        result = WorkflowImageData(
                            parent_metadata=parent_metadata,
                            numpy_image=cropped_image,
                        )
                    else:
                        result = None
                    crops.append({"crops": result})
                return crops
        ```

        * in lines `28-30` manifest class declares output dimensionality 
        offset - value `1` should be understood as adding `1` to dimensionality level
        
        * point out, that in line `63`, block eliminates empty images from further processing but 
        placing `None` instead of dictionatry with outputs. This would utilise the same 
        Execution Engine behaviour that is used for conditional execution - datapoint will
        be eliminated from downstream processing (unless steps requesting empty inputs 
        are present down the line).

        * in lines `64-65` results for single input `image` and `predictions` are collected - 
        it is meant to be list of dictionares containing all registered outputs as keys. Execution
        engine will understand that the step returns batch of elements for each input element and
        create nested sturcures of indices to keep track of during execution of downstream steps.

    === "output dimensionality decrease"
      
        In this example, the block visualises crops predictions and creates tiles
        presenting all crops predictions in single output image.

        ```{ .py linenums="1" hl_lines="30-32 34-36 53-55 65-66"}
        from typing import List, Literal, Type, Union

        import supervision as sv
        
        from inference.core.utils.drawing import create_tiles
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
                annotator = sv.BoxAnnotator()
                visualisations = []
                for image, prediction in zip(crops, crops_predictions):
                    annotated_image = annotator.annotate(
                        image.numpy_image.copy(),
                        prediction,
                    )
                    visualisations.append(annotated_image)
                tile = create_tiles(visualisations)
                return {"visualisations": tile}
        ```

        * in lines `30-32` manifest class declares output dimensionality 
        offset - value `-1` should be understood as decreasing dimensionality level by `1`

        * in lines `34-36` manifest class declares `run(...)` method inputs that will be subject to auto-batch casting
        ensuring that the signature is always stable. Auto-batch casting was introduced in Execution Engine `v0.1.6.0` 
        - refer to [changelog](./execution_engine_changelog.md) for more details.

        * in lines `53-55` you can see the impact of output dimensionality decrease
        on the method signature. First two inputs (declared in line `36`) are artificially wrapped in `Batch[]`
        container, whereas `scalar_parameter` remains primitive type. This is done by Execution Engine automatically 
        on output dimensionality decrease when all inputs have the same dimensionality to enable access to 
        all elements occupying the last dimensionality level. Obviously, only elements related to the same element 
        from top-level batch will be grouped. For instance, if you had two input images that you 
        cropped - crops from those two different images will be grouped separately.

        * lines `65-66` illustrate how output is constructed - single value is returned and that value 
        will be indexed by Execution Engine in output batch with reduced dimensionality

    === "different input dimensionalities"
        
        In this example, block merges detections which were predicted based on 
        crops of original image - result is to provide single detections with 
        all partial ones being merged.

        ```{ .py linenums="1" hl_lines="31-36 38-40 62-63 69"}
        from copy import deepcopy
        from typing import Dict, List, Literal, Optional, Type, Union
        
        import numpy as np
        import supervision as sv
        
        from inference.core.workflows.execution_engine.entities.base import (
            Batch,
            OutputDefinition,
            WorkflowImageData,
        )
        from inference.core.workflows.execution_engine.entities.types import (
            OBJECT_DETECTION_PREDICTION_KIND,
            Selector,
            IMAGE_KIND,
        )
        from inference.core.workflows.prototypes.block import (
            BlockResult,
            WorkflowBlock,
            WorkflowBlockManifest,
        )
        
        
        class BlockManifest(WorkflowBlockManifest):
            type: Literal["my_plugin/stitch@v1"]
            image: Selector(kind=[IMAGE_KIND])
            image_predictions: Selector(
                kind=[OBJECT_DETECTION_PREDICTION_KIND],
            )
        
            @classmethod
            def get_input_dimensionality_offsets(cls) -> Dict[str, int]:
                return {
                    "image": 0,
                    "image_predictions": 1,
                }
        
            @classmethod
            def get_dimensionality_reference_property(cls) -> Optional[str]:
                return "image"
        
            @classmethod
            def describe_outputs(cls) -> List[OutputDefinition]:
                return [
                    OutputDefinition(
                        name="predictions",
                        kind=[
                            OBJECT_DETECTION_PREDICTION_KIND,
                        ],
                    ),
                ]
        
        
        class StitchDetectionsNonBatchBlock(WorkflowBlock):
        
            @classmethod
            def get_manifest(cls) -> Type[WorkflowBlockManifest]:
                return BlockManifest
        
            def run(
                self,
                image: WorkflowImageData,
                image_predictions: Batch[sv.Detections],
            ) -> BlockResult:
                image_predictions = [deepcopy(p) for p in image_predictions if len(p)]
                for p in image_predictions:
                    coords = p["parent_coordinates"][0]
                    p.xyxy += np.concatenate((coords, coords))
                return {"predictions": sv.Detections.merge(image_predictions)}

        ```

        * in lines `31-36` manifest class declares input dimensionalities offset, indicating
        `image` parameter being top-level and `image_predictions` being nested batch of predictions

        * whenever different input dimensionalities are declared, dimensionality reference property
        must be pointed (see lines `38-40`) - this dimensionality level would be used to calculate 
        output dimensionality - in this particular case, we specify `image`. This choice 
        has an implication in the expected format of result - in the chosen scenario we are supposed
        to return single dictionary with all registered outputs keys. If our choice is `image_predictions`,
        we would return list of dictionaries (of size equal to length of `image_predictions` batch). In other worlds,
        `get_dimensionality_reference_property(...)` which dimensionality level should be associated
        to the output.

        * lines `63-64` present impact of dimensionality offsets specified in lines `31-36`. It is clearly
        visible that `image_predictions` is a nested batch regarding `image`. Obviously, only nested predictions
        relevant for the specific `images` are grouped in batch and provided to the method in runtime.

        * as mentioned earlier, line `69` construct output being single dictionary, as we register output 
        at dimensionality level of `image` (which was also shipped as single element)


??? example "Impact of dimensionality on `run(...)` method - batches enabled"

    === "output dimensionality increase"

        In this example, we perform dynamic crop of image based on predictions.

        ```{ .py linenums="1" hl_lines="29-31 33-35 55-56 70 71-73"}
        from typing import Dict, List, Literal, Optional, Type, Union
        from uuid import uuid4

        from inference.core.workflows.execution_engine.constants import DETECTION_ID_KEY
        from inference.core.workflows.execution_engine.entities.base import (
            OutputDefinition,
            WorkflowImageData,
            ImageParentMetadata,
            Batch,
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
            type: Literal["my_block/dynamic_crop@v1"]
            image: Selector(kind=[IMAGE_KIND])
            predictions: Selector(
                kind=[OBJECT_DETECTION_PREDICTION_KIND],
            )

            @classmethod
            def get_parameters_accepting_batches(cls) -> bool:
                return ["image", "predictions"]
        
            @classmethod
            def get_output_dimensionality_offset(cls) -> int:
                return 1
        
            @classmethod
            def describe_outputs(cls) -> List[OutputDefinition]:
                return [
                    OutputDefinition(name="crops", kind=[IMAGE_KIND]),
                ]
        
            @classmethod
            def get_execution_engine_compatibility(cls) -> Optional[str]:
                return ">=1.3.0,<2.0.0"

        class DynamicCropBlockV1(WorkflowBlock):

            @classmethod
            def get_manifest(cls) -> Type[WorkflowBlockManifest]:
                return BlockManifest
            
            def run(
                self,
                image: Batch[WorkflowImageData],
                predictions: Batch[sv.Detections],
            ) -> BlockResult:
                results = []
                for single_image, detections in zip(image, predictions):
                    crops = []
                    for (x_min, y_min, x_max, y_max) in detections.xyxy.round().astype(dtype=int):
                        cropped_image = single_image.numpy_image[y_min:y_max, x_min:x_max]
                        parent_metadata = ImageParentMetadata(parent_id=f"{uuid4()}")
                        if cropped_image.size:
                            result = WorkflowImageData(
                                parent_metadata=parent_metadata,
                                numpy_image=cropped_image,
                            )
                        else:
                            result = None
                        crops.append({"crops": result})
                    results.append(crops)
                return results
        ```
      
        * in lines `29-31` manifest declares that block accepts batches of inputs

        * in lines `33-35` manifest class declares output dimensionality 
        offset - value `1` should be understood as adding `1` to dimensionality level
        
        * in lines `55-66`, signature of input parameters reflects that the `run(...)` method
        runs against inputs of the same dimensionality and those inputs are provided in batches

        * point out, that in line `70`, block eliminates empty images from further processing but 
        placing `None` instead of dictionatry with outputs. This would utilise the same 
        Execution Engine behaviour that is used for conditional execution - datapoint will
        be eliminated from downstream processing (unless steps requesting empty inputs 
        are present down the line).

        * construction of the output, presented in lines `71-73` indicates two levels of nesting.
        First of all, block operates on batches, so it is expected to return list of outputs, one 
        output for each input batch element. Additionally, this output element for each input batch 
        element turns out to be nested batch - hence for each input iage and prediction, block 
        generates list of outputs - elements of that list are dictionaries providing values 
        for each declared output.

    === "output dimensionality decrease"
      
        In this example, the block visualises crops predictions and creates tiles
        presenting all crops predictions in single output image.

        ```{ .py linenums="1" hl_lines="29-31 33-35 52-53 66-67"}
        from typing import List, Literal, Type, Union

        import supervision as sv
        
        from inference.core.utils.drawing import create_tiles
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
            images_crops: Selector(kind=[IMAGE_KIND])
            crops_predictions: Selector(
                kind=[OBJECT_DETECTION_PREDICTION_KIND]
            )

            @classmethod
            def get_parameters_accepting_batches(cls) -> bool:
                return ["images_crops", "crops_predictions"]
        
            @classmethod
            def get_output_dimensionality_offset(cls) -> int:
                return -1
        
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
                images_crops: Batch[Batch[WorkflowImageData]],
                crops_predictions: Batch[Batch[sv.Detections]],
            ) -> BlockResult:
                annotator = sv.BoxAnnotator()
                visualisations = []
                for image_crops, crop_predictions in zip(images_crops, crops_predictions):
                    visualisations_batch_element = []
                    for image, prediction in zip(image_crops, crop_predictions):
                        annotated_image = annotator.annotate(
                            image.numpy_image.copy(),
                            prediction,
                        )
                        visualisations_batch_element.append(annotated_image)
                    tile = create_tiles(visualisations_batch_element)
                    visualisations.append({"visualisations": tile})
                return visualisations
        ```
        
        * lines `29-31` manifest that block is expected to take batches as input

        * in lines `33-35` manifest class declares output dimensionality 
        offset - value `-1` should be understood as decreasing dimensionality level by `1`

        * in lines `52-53` you can see the impact of output dimensionality decrease
        and batch processing on the method signature. First "layer" of `Batch[]` is a side effect of the 
        fact that manifest declared that block accepts batches of inputs. The second "layer" comes 
        from output dimensionality decrease. Execution Engine wrapps up the dimension to be reduced into 
        additional `Batch[]` container porvided in inputs, such that programmer is able to collect all nested
        batches elements that belong to specific top-level batch element.

        * lines `66-67` illustrate how output is constructed - for each top-level batch element, block
        aggregates all crops and predictions and creates a single tile. As block accepts batches of inputs,
        this procedure end up with one tile for each top-level batch element - hence list of dictionaries
        is expected to be returned.

    === "different input dimensionalities"
        
        In this example, block merges detections which were predicted based on 
        crops of original image - result is to provide single detections with 
        all partial ones being merged.

        ```{ .py linenums="1" hl_lines="31-33 35-40 42-44 66-67 76-77"}
        from copy import deepcopy
        from typing import Dict, List, Literal, Optional, Type, Union
        
        import numpy as np
        import supervision as sv
        
        from inference.core.workflows.execution_engine.entities.base import (
            Batch,
            OutputDefinition,
            WorkflowImageData,
        )
        from inference.core.workflows.execution_engine.entities.types import (
            OBJECT_DETECTION_PREDICTION_KIND,
            Selector,
            IMAGE_KIND,
        )
        from inference.core.workflows.prototypes.block import (
            BlockResult,
            WorkflowBlock,
            WorkflowBlockManifest,
        )
        
        
        class BlockManifest(WorkflowBlockManifest):
            type: Literal["my_plugin/stitch@v1"]
            images: Selector(kind=[IMAGE_KIND])
            images_predictions: Selector(
                kind=[OBJECT_DETECTION_PREDICTION_KIND],
            )

            @classmethod
            def get_parameters_accepting_batches(cls) -> bool:
                return ["images", "images_predictions"]
                
            @classmethod
            def get_input_dimensionality_offsets(cls) -> Dict[str, int]:
                return {
                    "image": 0,
                    "image_predictions": 1,
                }
        
            @classmethod
            def get_dimensionality_reference_property(cls) -> Optional[str]:
                return "image"
        
            @classmethod
            def describe_outputs(cls) -> List[OutputDefinition]:
                return [
                    OutputDefinition(
                        name="predictions",
                        kind=[
                            OBJECT_DETECTION_PREDICTION_KIND,
                        ],
                    ),
                ]
        
        
        class StitchDetectionsBatchBlock(WorkflowBlock):
        
            @classmethod
            def get_manifest(cls) -> Type[WorkflowBlockManifest]:
                return BlockManifest
        
            def run(
                self,
                images: Batch[WorkflowImageData],
                images_predictions: Batch[Batch[sv.Detections]],
            ) -> BlockResult:
                result = []
                for image, image_predictions in zip(images, images_predictions):
                    image_predictions = [deepcopy(p) for p in image_predictions if len(p)]
                    for p in image_predictions:
                        coords = p["parent_coordinates"][0]
                        p.xyxy += np.concatenate((coords, coords))
                    merged_prediction = sv.Detections.merge(image_predictions)
                    result.append({"predictions": merged_prediction})
                return result
        ```

        * lines `31-33` manifest that block is expected to take batches as input

        * in lines `35-40` manifest class declares input dimensionalities offset, indicating
        `image` parameter being top-level and `image_predictions` being nested batch of predictions

        * whenever different input dimensionalities are declared, dimensionality reference property
        must be pointed (see lines `42-44`) - this dimensionality level would be used to calculate 
        output dimensionality - in this particular case, we specify `image`. This choice 
        has an implication in the expected format of result - in the chosen scenario we are supposed
        to return single dictionary for each element of `image` batch. If our choice is `image_predictions`,
        we would return list of dictionaries (of size equal to length of nested `image_predictions` batch) for each
        input `image` batch element.

        * lines `66-67` present impact of dimensionality offsets specified in lines `35-40` as well as 
        the declararion of batch processing from lines `32-34`. First "layer" of `Batch[]` container comes 
        from the latter, nested `Batch[Batch[]]` for `images_predictions` comes from the definition of input 
        dimensionality offset. It is clearly visible that `image_predictions` holds batch of predictions relevant
        for specific elements of `image` batch.
        
        * as mentioned earlier, lines `76-77` construct output being single dictionary for each element of `image` 
        batch


### Block accepting empty inputs

As discussed earlier, some batch elements may become "empty" during the execution of a Workflow. 
This can happen due to several factors:

* **Flow-control mechanisms:** Certain branches of execution can mask specific batch elements, preventing them 
from being processed in subsequent steps.

* **In data-processing blocks:** In some cases, a block may not be able to produce a meaningful output for 
a specific data point. For example, a Dynamic Crop block cannot generate a cropped image if the bounding box 
size is zero.

Some blocks are designed to handle these empty inputs, such as block that can replace missing outputs with default 
values. This block can be particularly useful when constructing structured outputs in a Workflow, ensuring 
that even if some elements are empty, the output lacks missing elements making it harder to parse.

??? example "Block accepting empty inputs"

    ```{ .py linenums="1" hl_lines="20-22 41"}
    from typing import Any, List, Literal, Optional, Type

    from inference.core.workflows.execution_engine.entities.base import (
        Batch,
        OutputDefinition,
    )
    from inference.core.workflows.execution_engine.entities.types import Selector
    from inference.core.workflows.prototypes.block import (
        BlockResult,
        WorkflowBlock,
        WorkflowBlockManifest,
    )


    class BlockManifest(WorkflowBlockManifest):
        type: Literal["my_plugin/first_non_empty_or_default@v1"]
        data: List[Selector()]
        default: Any
    
        @classmethod
        def accepts_empty_values(cls) -> bool:
            return True
    
        @classmethod
        def describe_outputs(cls) -> List[OutputDefinition]:
            return [OutputDefinition(name="output")]
    
        @classmethod
        def get_execution_engine_compatibility(cls) -> Optional[str]:
            return ">=1.3.0,<2.0.0"
    
    
    class FirstNonEmptyOrDefaultBlockV1(WorkflowBlock):
    
        @classmethod
        def get_manifest(cls) -> Type[WorkflowBlockManifest]:
            return BlockManifest
    
        def run(
            self,
            data: Batch[Optional[Any]],
            default: Any,
        ) -> BlockResult:
            result = default
            for data_element in data:
                if data_element is not None:
                    return {"output": data_element}
            return {"output": result}
    ```

    * in lines `20-22` you may find declaration stating that block acccepts empt inputs 
    
    * a consequence of lines `20-22` is visible in line `41`, when signature states that 
    input `Batch` may contain empty elements that needs to be handled. In fact - the block 
    generates "artificial" output substituting empty value, which makes it possible for 
    those outputs to be "visible" for blocks not accepting empty inputs that refer to the 
    output of this block. You should assume that each input that is substituted by Execution
    Engine with data generated in runtime may provide optional elements.


### Block with custom constructor parameters

Some blocks may require objects constructed by outside world to work. In such
scenario, Workflows Execution Engine job is to transfer those entities to the block, 
making it possible to be used. The mechanism is described in 
[the page presenting Workflows Compiler](/workflows/workflows_compiler.md), as this is the 
component responsible for dynamic construction of steps from blocks classes.

Constructor parameters must be:

* requested by block - using class method `WorkflowBlock.get_init_parameters(...)`

* provided in the environment running Workflows Execution Engine:

    * directly, as shown in [this](/workflows/modes_of_running.md#workflows-in-python-package) example
    
    * using defaults [registered for Workflow plugin](/workflows/blocks_bundling.md)

Let's see how to request init parameters while defining block.

??? example "Block requesting constructor parameters"

    ```{ .py linenums="1" hl_lines="30-31 33-35"}
    from typing import Any, List, Literal, Optional, Type

    from inference.core.workflows.execution_engine.entities.base import (
        Batch,
        OutputDefinition,
    )
    from inference.core.workflows.execution_engine.entities.types import Selector
    from inference.core.workflows.prototypes.block import (
        BlockResult,
        WorkflowBlock,
        WorkflowBlockManifest,
    )


    class BlockManifest(WorkflowBlockManifest):
        type: Literal["my_plugin/example@v1"]
        data: List[Selector()]
    
        @classmethod
        def describe_outputs(cls) -> List[OutputDefinition]:
            return [OutputDefinition(name="output")]
    
        @classmethod
        def get_execution_engine_compatibility(cls) -> Optional[str]:
            return ">=1.0.0,<2.0.0"
    
    
    class ExampleBlock(WorkflowBlock):
      
        def __init__(my_parameter: int):
            self._my_parameter = my_parameter

        @classmethod
        def get_init_parameters(cls) -> List[str]:
            return ["my_parameter"]
        
        @classmethod
        def get_manifest(cls) -> Type[WorkflowBlockManifest]:
            return BlockManifest
    
        def run(
            self,
            data: Batch[Any],
        ) -> BlockResult:
            pass
    ```

    * lines `30-31` declare class constructor which is not parameter-free

    * to inform Execution Engine that block requires custom initialisation, 
    `get_init_parameters(...)` method in lines `33-35` enlists names of all 
    parameters that must be provided
