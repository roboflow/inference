# How to create custom `workflows` block

!!! note
    We have a plan to simplify creation of new blocks. That means changes into internal design of blocks and their
    interface. The document present current state which is surely suboptimal. We accept feedback and ideas for
    improvements in [GitHub issues](https://github.com/roboflow/inference/issues).

!!! note
    Fundamental context knowledge about the nature of `workflows` is [here](./create_and_run.md). We recommend 
    acknowledging this document before further reading.

## What elements make `workflows` block?
Each `workflows` block has:

* block manifest in form of `pydantic` entity - providing structure for JSON step definition and a logic to validate input
* function to execute step logic with the following signature:

```python
async def run_xxx_step(
    step: YourStepType,
    runtime_parameters: Dict[str, Any],
    outputs_lookup: OutputsLookup,
    model_manager: ModelManager,
    api_key: Optional[str],
    step_execution_mode: StepExecutionMode,
) -> Tuple[NextStepReference, OutputsLookup]:
    ...
```

Both manifest and `run_xxx_step()` function will be used by `workflow` compiler and executor modules. Manifest 
holds the source of truth regarding required inputs and the rules dictating which steps may be connected with each 
other.

## What steps need to be completed to successfully add `workflows` block?

1. Define the block manifest entity and implement all necessary validation logic
2. Implement step execution logic
3. Register the step in [execution engine module](https://github.com/roboflow/inference/blob/main/inference/enterprise/workflows/complier/execution_engine.py) adding entry to `STEP_TYPE2EXECUTOR_MAPPING`
4. Register manifest entity in [workflow specification](https://github.com/roboflow/inference/blob/main/inference/enterprise/workflows/entities/workflows_specification.py) adding entry into `StepType` union and `ALL_BLOCKS_CLASSES`
5. At this step, you should be able to add newly created block into JSON `workflow` definition and run it using one of `workflows` execution entrypoint (Python package function or HTTP endpoint)

Initial design of `workflows` was intended to make compiler and execution engine do heavy-lifting in terms of organising
execution, but there still may be needs to re-design those core modules if we find corner-cases that are not handled.
Please report them in [GitHub issues](https://github.com/roboflow/inference/issues).

## Notion of `kind` in `workflows`
Since `v0.9.21` we introduced a simple type system on top of selectors / references used in `workflows`.
When defining block manifests (you will learn how to do it in next section) you would be in need to provide
Pydantic type annotations for block inputs. You will find that sometimes fields will accept
static values (defined while `workflow` is created), and sometimes you need to refer to element that
will appear dynamically while `workflow` execution (like the output of previous step, or input parameter).
Once that is done, you use selector (example `$steps.<step_name>.<output_name>`). As you see, 
this value in entity definition is string that must match some reg-ex. So the type of value for JSON definition
of `workflow` would be `str`, but we want to understand what is behind the reference.
We can define kind as union of simple "types" - but in this case, we understand kind as high-level
concept, rather than specific type. For instance, among defined kinds we have `object_detection_prediction` that
represents list of bounding boxes details for each of input image.

Kinds definitions can be found in `inference.enterprise.workflows.entities.types` module.

Kind definition is an object that have `name` (used for matching) and `description` to
express high-level meaning. Later on we may add additional characteristics.

We have pre-defined type builders for references and we can use them to annotate pydantic
fields types:
- `StepOutputSelector(kind=[...])` - to point step output of given kind
- `InferenceParameterSelector(kind=[...])` - to point `InferenceParameter` input
- `InferenceImageSelector(kind=[...])` - to point `InferenceImage` input
- `OutputStepImageSelector` - to point image output by step
- `StepSelector` - to point whole step

## How to define block manifest?

!!! note
    Package with blocks manifest is located [here](https://github.com/roboflow/inference/tree/main/inference/enterprise/workflows/entities)


Block manifests are located [here](https://github.com/roboflow/inference/blob/main/inference/enterprise/workflows/entities/steps.py) 
in `inference` repository structure. Creating new one, you should start from:

```python
from typing import Literal, Set, Optional, Any, List
from pydantic import BaseModel
from inference.enterprise.workflows.entities.steps import StepInterface
from inference.enterprise.workflows.entities.base import GraphNone

class MyStep(BaseModel, StepInterface):
    type: Literal["MyStep"]
    name: str
    ... # place here other inputs that block takes
    
    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        # This method was added as part of transition into new workflows design. 
        # It is meant to describe static outputs of block without need for init of the class
        # such that we can generate all blocks descriptions without initialising 
        # any class with data. Each output should be registered with kind.
        return []

    def get_input_names(self) -> Set[str]:
        ...  # Supposed to give the name of all fields expected to be possible for compiler to plug values into

    def get_output_names(self) -> Set[str]:
        ... # Supposed to give the name of all fields expected to represent outputs to be referred by other blocks

    def validate_field_selector(
        self, field_name: str, input_step: GraphNone, index: Optional[int] = None
    ) -> None:
        ...  # Supposed to validate the type of input is referred

    def validate_field_binding(self, field_name: str, value: Any) -> None:
        """
        Supposed to validate the type of value that is to be bounded with field as a result of graph
        execution (values passed by client to invocation, as well as constructed during graph execution)
        """
        ...  

    def get_type(self) -> str:
        return self.type
```

Let's discuss one-by-one the elements of manifest.

### Pydantic model fields
We require `type` and `name` fields to be defined. Rest is up to you. Let's assume that we deal with step that
accept image and additional threshold parameter. Then step definition would look like that:

```python
from typing import Literal, Union, Optional
from pydantic import BaseModel, Field
from inference.enterprise.workflows.entities.steps import StepInterface
from inference.enterprise.workflows.entities.types import (
    InferenceImageSelector, 
    InferenceParameterSelector,
    OutputStepImageSelector, 
    FloatZeroToOne,
    FLOAT_ZERO_TO_ONE_KIND
)


class MyStep(BaseModel, StepInterface):
    type: Literal["MyStep"]
    name: str = Field(description="Unique name of step in workflows")
    image: Union[InferenceImageSelector, OutputStepImageSelector] = Field(
        description="Reference at image to be used as input for step processing",
        examples=["$inputs.image", "$steps.cropping.crops"],
    )
    confidence: Union[
        Optional[FloatZeroToOne],
        InferenceParameterSelector(kind=[FLOAT_ZERO_TO_ONE_KIND]),
    ] = Field(
        default=0.4,
        description="Confidence threshold for predictions",
        examples=[0.3, "$inputs.confidence_threshold"],
    )
```

The idea behind `workflows` is to be able to set the parameters directly in JSON definition of steps, but also make it
possible to defer injection of parameters to `workflows` runtime, when specific values would either been calculated
or provided by users as additional (static) input. 

What happens with `image` here - we say that it reference to input image or to the image output from another step, 
That's why we do not have this field of type `np.ndarray` or any other that usually holds image data. 

With `confidence`, however, we may want to define the value either in JSON definition of `workflow`, or as a reference.

We should aim to validate `workflows` definitions using `pydantic` validation engine. To make it 
possible, we need to create type constraints at the level of type annotations - then
those information will be exportable to outside world (for instance via the endpoint to
describe blocks).


### Why do I need other methods from the step interface?
`Pydantic` validation is very important in making sure that what is sent as JSON definition of `workflow` is actually a 
valid one, but `StepInterface` requires you to implement a couple of additional methods. Let's discover their purpose.

`get_input_names(...)` allows compiler to discover the "placeholders" which can be filled with values in the runtime.
You should return a set of field names defined in entity that are possible to hold selectors - and that will be possible
to substitute with actual values in the runtime.

`get_output_names(...)` is the way to define block outputs - that can be used in selectors of other steps in `workflow`

`get_type(...)` should simply return value of `type` field

Two important, and potentially confusing methods are: `validate_field_selector(...)` and `validate_field_binding(...)`.

#### Validation of field selector
`validate_field_selector(...)` is used during execution graph construction stage of compiler. That method is supposed 
to validate selectors defined in block fields - in particular type of steps / inputs that selector refers to. In our case:

```python
from typing import Literal, Union, Optional
from pydantic import BaseModel, Field

from inference.enterprise.workflows.entities.validators import (
    is_selector,
    validate_selector_holds_image,
    validate_selector_is_inference_parameter,
)
from inference.enterprise.workflows.entities.steps import StepInterface, GraphNone
from inference.enterprise.workflows.entities.types import (
    InferenceImageSelector, 
    InferenceParameterSelector,
    OutputStepImageSelector, 
    FloatZeroToOne,
    FLOAT_ZERO_TO_ONE_KIND
)
from inference.enterprise.workflows.errors import ExecutionGraphError


class MyStep(BaseModel, StepInterface):
    type: Literal["MyStep"]
    name: str = Field(description="Unique name of step in workflows")
    image: Union[InferenceImageSelector, OutputStepImageSelector] = Field(
        description="Reference at image to be used as input for step processing",
        examples=["$inputs.image", "$steps.cropping.crops"],
    )
    confidence: Union[
        Optional[FloatZeroToOne],
        InferenceParameterSelector(kind=[FLOAT_ZERO_TO_ONE_KIND]),
    ] = Field(
        default=0.4,
        description="Confidence threshold for predictions",
        examples=[0.3, "$inputs.confidence_threshold"],
    )
    
    def validate_field_selector(
        self, field_name: str, input_step: GraphNone, index: Optional[int] = None
    ) -> None:
        if not is_selector(selector_or_value=getattr(self, field_name)):
            raise ExecutionGraphError(
                f"Attempted to validate selector value for field {field_name}, but field is not selector."
            )
        validate_selector_holds_image(
            step_type=self.type,
            field_name=field_name,
            input_step=input_step,
        )
        validate_selector_is_inference_parameter(
            step_type=self.type,
            field_name=field_name,
            input_step=input_step,
            applicable_fields={"confidence"},
        )
```

Compiler is going to use `validate_field_selector(...)` only against detected selectors - so initial check should be done to catch corner-cases when it does not work and provide meaningful error message. In the next stage, `validate_selector_holds_image(...)` that triggers if `field_name=image` by default is going to check if `input_step` that was referred by the selector (and delivered to the method automatically by compiler) is an element of the graph that holds image in their output (`InferenceImage` or step with image output). In final stage - `validate_selector_is_inference_parameter(...)` that triggers on field `confidence` will check  if the `input_step` is `InferenceParameter` which is the only `workflow` element supposed to provide static value from user input in runtime.

Additional parameter, called `index` will only be filled by compiler if specific manifest field is a list of selectors, then validation will happen for each element separately.

!!! note 
    
    We have plan to get rid of `validate_field_selector(...)` when we fully apply the notion of kinds

#### Validation of input binding

`validate_field_binding(...)` is used by compiler while substituting selectors with values provided as user input into
`workflow` execution. It plays similar role to `pydantic` validation, just in context of user input available in runtime.

Let's see how we can validate input binding in case of our example block:

```python
from typing import Literal, Union, Optional, Any
from pydantic import BaseModel, Field

from inference.enterprise.workflows.entities.steps import StepInterface, GraphNone
from inference.enterprise.workflows.entities.types import (
    InferenceImageSelector, 
    InferenceParameterSelector,
    OutputStepImageSelector, 
    FloatZeroToOne,
    FLOAT_ZERO_TO_ONE_KIND
)
from inference.enterprise.workflows.entities.validators import (
    validate_image_biding,
    validate_field_has_given_type
)
from inference.enterprise.workflows.errors import VariableTypeError


class MyStep(BaseModel, StepInterface):
    type: Literal["MyStep"]
    name: str = Field(description="Unique name of step in workflows")
    image: Union[InferenceImageSelector, OutputStepImageSelector] = Field(
        description="Reference at image to be used as input for step processing",
        examples=["$inputs.image", "$steps.cropping.crops"],
    )
    confidence: Union[
        Optional[FloatZeroToOne],
        InferenceParameterSelector(kind=[FLOAT_ZERO_TO_ONE_KIND]),
    ] = Field(
        default=0.4,
        description="Confidence threshold for predictions",
        examples=[0.3, "$inputs.confidence_threshold"],
    )
    
    # ... pydantic validation skipped for readability
    # ... validate_field_selector(...) skipped for readability
    
    def validate_field_binding(self, field_name: str, value: Any) -> None:
        if field_name == "image":
            validate_image_biding(value=value)
        elif field_name == "confidence":
            validate_field_has_given_type(
                field_name=field_name,
                allowed_types=[float, type(None)],
                value=value,
                error=VariableTypeError,
            )
```

!!! note 
    
    We have plan to get rid of `validate_field_binding(...)` when we fully apply the notion of kinds


#### Defining static output `kind`
As a consequence of `kind` introduction, we need to implement class method `describe_outputs(...)`. 

```python
from typing import Literal, Union, Optional, List
from pydantic import BaseModel, Field

from inference.enterprise.workflows.entities.steps import StepInterface, OutputDefinition
from inference.enterprise.workflows.entities.types import (
    InferenceImageSelector, 
    InferenceParameterSelector,
    OutputStepImageSelector, 
    FloatZeroToOne,
    FLOAT_ZERO_TO_ONE_KIND,
    CLASSIFICATION_PREDICTION_KIND,
    STRING_KIND,
    PARENT_ID_KIND,
)

class MyStep(BaseModel, StepInterface):
    type: Literal["MyStep"]
    name: str = Field(description="Unique name of step in workflows")
    image: Union[InferenceImageSelector, OutputStepImageSelector] = Field(
        description="Reference at image to be used as input for step processing",
        examples=["$inputs.image", "$steps.cropping.crops"],
    )
    confidence: Union[
        Optional[FloatZeroToOne],
        InferenceParameterSelector(kind=[FLOAT_ZERO_TO_ONE_KIND]),
    ] = Field(
        default=0.4,
        description="Confidence threshold for predictions",
        examples=[0.3, "$inputs.confidence_threshold"],
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        # this is just reference implementation - adjust to your step
        return super(MyStep, cls).describe_outputs() + [
            OutputDefinition(name="predictions", kind=[CLASSIFICATION_PREDICTION_KIND]),
            OutputDefinition(name="parent_id", kind=[PARENT_ID_KIND]),
        ]
```

### Full implementation of manifest
```python
from typing import Literal, Union, Any, Optional, Set, List
from pydantic import BaseModel, Field
from inference.enterprise.workflows.entities.steps import StepInterface, OutputDefinition
from inference.enterprise.workflows.entities.base import GraphNone
from inference.enterprise.workflows.entities.validators import (
    is_selector,
    validate_selector_holds_image,
    validate_selector_is_inference_parameter,
    validate_image_biding,
    validate_field_has_given_type,
)
from inference.enterprise.workflows.errors import ExecutionGraphError, VariableTypeError
from inference.enterprise.workflows.entities.types import (
    InferenceImageSelector, 
    InferenceParameterSelector,
    OutputStepImageSelector, 
    FloatZeroToOne,
    FLOAT_ZERO_TO_ONE_KIND,
    CLASSIFICATION_PREDICTION_KIND,
    PARENT_ID_KIND,
)

class MyStep(BaseModel, StepInterface):
    type: Literal["MyStep"]
    name: str = Field(description="Unique name of step in workflows")
    image: Union[InferenceImageSelector, OutputStepImageSelector] = Field(
        description="Reference at image to be used as input for step processing",
        examples=["$inputs.image", "$steps.cropping.crops"],
    )
    confidence: Union[
        Optional[FloatZeroToOne],
        InferenceParameterSelector(kind=[FLOAT_ZERO_TO_ONE_KIND]),
    ] = Field(
        default=0.4,
        description="Confidence threshold for predictions",
        examples=[0.3, "$inputs.confidence_threshold"],
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        # this is just reference implementation - adjust to your step
        return super(MyStep, cls).describe_outputs() + [
            OutputDefinition(name="predictions", kind=[CLASSIFICATION_PREDICTION_KIND]),
            OutputDefinition(name="parent_id", kind=[PARENT_ID_KIND]),
        ]
    
    def get_input_names(self) -> Set[str]:
        return {"image", "confidence"}

    def get_output_names(self) -> Set[str]:
        # for now must much with describe_outputs(...), but we will get rid of this in the future
        return {"predictions", "parent_id"}  # adjust this to the use-case

    def validate_field_selector(
        self, field_name: str, input_step: GraphNone, index: Optional[int] = None
    ) -> None:
        if not is_selector(selector_or_value=getattr(self, field_name)):
            raise ExecutionGraphError(
                f"Attempted to validate selector value for field {field_name}, but field is not selector."
            )
        validate_selector_holds_image(
            step_type=self.type,
            field_name=field_name,
            input_step=input_step,
        )
        validate_selector_is_inference_parameter(
            step_type=self.type,
            field_name=field_name,
            input_step=input_step,
            applicable_fields={"confidence"},
        )

    def validate_field_binding(self, field_name: str, value: Any) -> None:
        if field_name == "image":
            validate_image_biding(value=value)
        elif field_name == "confidence":
            validate_field_has_given_type(
                field_name=field_name,
                allowed_types=[float, type(None)],
                value=value,
                error=VariableTypeError,
            )

    def get_type(self) -> str:
        return self.type
```

## How to implement block logic?

Let's start from defining the placeholder function:

```python
from typing import Dict, Any, Optional, Tuple
from inference.core.managers.base import ModelManager
from inference.enterprise.workflows.complier.steps_executors.types import (
    NextStepReference,
    OutputsLookup,
)
from inference.enterprise.workflows.complier.entities import StepExecutionMode


async def run_my_step(
    step: MyStep,
    runtime_parameters: Dict[str, Any],
    outputs_lookup: OutputsLookup,
    model_manager: ModelManager,
    api_key: Optional[str],
    step_execution_mode: StepExecutionMode,
) -> Tuple[NextStepReference, OutputsLookup]:
    ...
```

`runtime_parameters` is dict with user parameters provided as an input for execution. 

`OutputsLookup` is a dictionary with each step output.

`model_manager`, `api_key` are `inference` entities to deal with models.

`step_execution_mode` - dictates how step should be executed - locally, within boundary of process running workflow, 
or as request to remote API (if applicable).

What this function returns is optionally the reference of next step (in case of blocks that alter flow of execution)
and `output_lookup` filled with step outputs.

There are two important concepts that need to be discussed:

* how to get actual values of parameters from `runtime_parameters` and `outputs_lookup`

* how to register step outputs in `outputs_lookup`

### Resolution of parameters at block level

!!! note
    That's probably the most tedious and not needed element of block creation, as that could be fully resolved on the
    executor side. We'll try to make that better in next iteration.

!!! note
    Package with blocks logic is located [here](https://github.com/roboflow/inference/tree/main/inference/enterprise/workflows/complier/steps_executors)

In runtime, `runtime_parameters` and `outputs_lookup` holds actual values of parameters needed for execution, whereas
`step` hold instance of block manifest entity, with combination of specific values and references at the fields level.
To resolve all of those sources of data into values you should calculate results - you need to use helper functions:
`resolve_parameter(...)` and `get_image(...)` - for images.

Let's see how that would look like:

```python
from typing import Dict, Any, Optional, Tuple
from inference.core.managers.base import ModelManager
from inference.enterprise.workflows.complier.steps_executors.types import (
    NextStepReference,
    OutputsLookup,
)
from inference.enterprise.workflows.complier.entities import StepExecutionMode
from inference.enterprise.workflows.complier.steps_executors.utils import (
    get_image,
    resolve_parameter,
)

async def run_my_step(
    step: MyStep,
    runtime_parameters: Dict[str, Any],
    outputs_lookup: OutputsLookup,
    model_manager: ModelManager,
    api_key: Optional[str],
    step_execution_mode: StepExecutionMode,
) -> Tuple[NextStepReference, OutputsLookup]:
    images = get_image(   # image always is returned in list - single entry format {"type": "...", "value": "..."} matches image representation in `inference` server
        step=step,
        runtime_parameters=runtime_parameters,
        outputs_lookup=outputs_lookup,
    )
    confidence = resolve_parameter(
        selector_or_value=step.confidence,
        runtime_parameters=runtime_parameters,
        outputs_lookup=outputs_lookup,
    )
    ...
```

Then you need to make the processing (possibly including operations on `ModelManager` to get predictions from model).
Representation of elements of `images` matches standard `inference` format - you can use `load_image(...)` function from 
core of `inference` to get `np.array`.

We shall now discuss the structure of `outputs_lookup`. It is dictionary that maps step name to it's output. Function
should only add values under its step name, not modify existing values (which may lead to unexpected side effects). 
Each block define outputs (via `get_output_names(...)`). As a value saved under step name you should place a dictionary,
with keys being all elements of block manifest `get_output_names(...)` result. Under each of the key representing output 
name you should save list of results - ordered by the order of images in the `image` list.

Let's see how that would look like in practice:

```python
from typing import Dict, Any, Optional, Tuple
from inference.core.managers.base import ModelManager
from inference.enterprise.workflows.complier.steps_executors.types import (
    NextStepReference,
    OutputsLookup,
)
from inference.enterprise.workflows.complier.entities import StepExecutionMode
from inference.enterprise.workflows.complier.steps_executors.utils import (
    get_image,
    resolve_parameter,
)
from inference.enterprise.workflows.execution_engine.compiler.utils import construct_step_selector


async def run_my_step(
        step: MyStep,
        runtime_parameters: Dict[str, Any],
        outputs_lookup: OutputsLookup,
        model_manager: ModelManager,
        api_key: Optional[str],
        step_execution_mode: StepExecutionMode,
) -> Tuple[NextStepReference, OutputsLookup]:
    images = get_image(
        # image always is returned in list - single entry format {"type": "...", "value": "..."} matches image representation in `inference` server
        step=step,
        runtime_parameters=runtime_parameters,
        outputs_lookup=outputs_lookup,
    )
    confidence = resolve_parameter(
        selector_or_value=step.confidence,
        runtime_parameters=runtime_parameters,
        outputs_lookup=outputs_lookup,
    )
    predictions = []
    for single_image in images:
        # ... make predictions, model
        predictions.append({"top": "cat"})
    outputs_lookup[construct_step_selector(step_name=step.name)] = {"prediction": predictions}
    return None, outputs_lookup
```

## Registration of the step
To make step ready to be used, you need to register block in [execution engine module](https://github.com/roboflow/inference/blob/main/inference/enterprise/workflows/complier/execution_engine.py):

```python
STEP_TYPE2EXECUTOR_MAPPING = {
    # ...,
    "MyStep": run_my_step,
}
```

and make changes in [workflow specification](https://github.com/roboflow/inference/blob/main/inference/enterprise/workflows/entities/workflows_specification.py):
```python
StepType = Annotated[
    Union[
        # ...,
        MyStep
    ],
    Field(discriminator="type"),
]
```

At this point - your block should be ready to go!
