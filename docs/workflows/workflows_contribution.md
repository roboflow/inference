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
4. Register manifest entity in [workflow specification](https://github.com/roboflow/inference/blob/main/inference/enterprise/workflows/entities/workflows_specification.py) adding entry into `StepType` union
5. At this step, you should be able to add newly created block into JSON `workflow` definition and run it using one of `workflows` execution entrypoint (Python package function or HTTP endpoint)

Initial design of `workflows` was intended to make compiler and execution engine do heavy-lifting in terms of organising
execution, but there still may be needs to re-design those core modules if we find corner-cases that are not handled.
Please report them in [GitHub issues](https://github.com/roboflow/inference/issues).

## How to define block manifest?
Block manifests are located [here](https://github.com/roboflow/inference/blob/main/inference/enterprise/workflows/entities/steps.py) 
in `inference` repository structure. Creating new one, you should start from:

```python
from typing import Literal, Set, Optional, Any
from pydantic import BaseModel
from inference.enterprise.workflows.entities.steps import StepInterface
from inference.enterprise.workflows.entities.base import GraphNone

class MyStep(BaseModel, StepInterface):
    type: Literal["MyStep"]
    name: str
    ... # place here other inputs that block takes

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
from typing import Literal, Union
from pydantic import BaseModel
from inference.enterprise.workflows.entities.steps import StepInterface

class MyStep(BaseModel, StepInterface):
    type: Literal["MyStep"]
    name: str
    image: str
    confidence: Union[float, str]
```

The idea behind `workflows` is to be able to set the parameters directly in JSON definition of steps, but also make it
possible to defer injection of parameters to `workflows` runtime, when specific values would either been calculated
or provided by users as additional (static) input. 

What happens with `image` here - we say that it is of type `str`, with intention of that string to hold reference
to either user input or other step output. That's why we do not have this field of type `np.ndarray` or any other that
usually holds image data. 

With `confidence`, however, we may want to define the value either in JSON definition of `workflow`, or as a reference.
That's why we allow either `float` value to be defined or `str`.

We would also want to be able to validate `workflows` definitions using `pydantic` validation engine. To make that happen,
you need to create custom validator method for specific fields:

```python
from typing import Literal, Union, Any, List, Optional
from pydantic import BaseModel, field_validator
from inference.enterprise.workflows.entities.steps import StepInterface
from inference.enterprise.workflows.entities.validators import (
    validate_image_is_valid_selector,
    validate_field_is_in_range_zero_one_or_empty_or_selector,
)

class MyStep(BaseModel, StepInterface):
    type: Literal["MyStep"]
    name: str
    image: str
    confidence: Union[Optional[float], str]

    @field_validator("image")
    @classmethod
    def validate_image(cls, value: Any) -> Union[str, List[str]]:
        validate_image_is_valid_selector(value=value)
        return value
    
    @field_validator("confidence")
    @classmethod
    def confidence_must_be_selector_or_number(
        cls, value: Any
    ) -> Union[Optional[float], str]:
        validate_field_is_in_range_zero_one_or_empty_or_selector(value=value)
        return value
```

In this example, you can see that our `image` field that hold `str` is only allowed to hold special kind of string - 
namely selector that refers to specific element of `workflow`.

It would be tedious to create custom validators for each and every field of each and every block. That's why we 
have module with utils useful for validation that can be chained together to get desired effect. 
See [`inference.enterprise.workflows.entities.validators` module](https://github.com/roboflow/inference/blob/main/inference/enterprise/workflows/entities/validators.py)


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
from pydantic import BaseModel
from inference.enterprise.workflows.entities.steps import StepInterface
from inference.enterprise.workflows.entities.validators import (
    is_selector,
    validate_selector_holds_image,
    validate_selector_is_inference_parameter,
)
from inference.enterprise.workflows.entities.base import GraphNone
from inference.enterprise.workflows.errors import ExecutionGraphError

class MyStep(BaseModel, StepInterface):
    type: Literal["MyStep"]
    name: str
    image: str
    confidence: Union[Optional[float], str]
    
    # ... pydantic validation skipped for readability
    
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

Compiler is going to use `validate_field_selector(...)` only against detected selectors - so initial check should
be done to catch corner-cases when it does not work and provide meaningful error message. In the next stage, 
`validate_selector_holds_image(...)` that triggers if `field_name=image` by default is going to check if 
`input_step` that was referred by the selector (and delivered to the method automatically by compiler) is 
an element of the graph that holds image in their output (`InferenceImage` or step with image output).
In final stage - `validate_selector_is_inference_parameter(...)` that triggers on field `confidence` will check 
if the `input_step` is `InferenceParameter` which is the only `workflow` element supposed to provide static value 
from user input in runtime.

Additional parameter, called `index` will only be filled by compiler if specific manifest field is a list of selectors,
then validation will happen for each element separately.

#### Validation of input binding

`validate_field_binding(...)` is used by compiler while substituting selectors with values provided as user input into
`workflow` execution. It plays similar role to `pydantic` validation, just in context of user input available in runtime.

Let's see how we can validate input binding in case of our example block:

```python
from typing import Literal, Union, Optional, Any
from pydantic import BaseModel
from inference.enterprise.workflows.entities.steps import StepInterface
from inference.enterprise.workflows.entities.validators import (
    validate_image_biding,
    validate_field_has_given_type
)
from inference.enterprise.workflows.errors import VariableTypeError

class MyStep(BaseModel, StepInterface):
    type: Literal["MyStep"]
    name: str
    image: str
    confidence: Union[Optional[float], str]
    
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

### Full implementation of manifest
```python
from typing import Literal, Union, Any, List, Optional
from pydantic import BaseModel, field_validator
from inference.enterprise.workflows.entities.steps import StepInterface
from inference.enterprise.workflows.entities.base import GraphNone
from inference.enterprise.workflows.entities.validators import (
    validate_image_is_valid_selector,
    validate_field_is_in_range_zero_one_or_empty_or_selector,
    is_selector,
    validate_selector_holds_image,
    validate_selector_is_inference_parameter,
    validate_image_biding,
    validate_field_has_given_type,
)
from inference.enterprise.workflows.errors import ExecutionGraphError, VariableTypeError


class MyStep(BaseModel, StepInterface):
    type: Literal["MyStep"]
    name: str
    image: str
    confidence: Union[Optional[float], str]

    @field_validator("image")
    @classmethod
    def validate_image(cls, value: Any) -> Union[str, List[str]]:
        validate_image_is_valid_selector(value=value)
        return value
    
    @field_validator("confidence")
    @classmethod
    def confidence_must_be_selector_or_number(
        cls, value: Any
    ) -> Union[Optional[float], str]:
        validate_field_is_in_range_zero_one_or_empty_or_selector(value=value)
        return value

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

