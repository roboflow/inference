# Roboflow `workflows`

Roboflow `workflows` let you integrate your Machine Learning model into your environment without 
writing a single line of code. You only need to select which steps you need to complete desired
actions and connect them together to create a `workflow`.


## What makes a `workflow`?

As mentioned above, `workflows` are made of `steps`. `Steps` perform specific actions (like 
making prediction from a model or registering results somewhere). They are instances of `blocks`, 
which ship the `code` (implementing what's happening behind the scene) and `manifest` (describing 
the way on how `block` can be used). Great news is you do not even need to care about the `code`.
(If you do, that's great - `workflows` only may benefit from your skills). But the clue is - 
there are two (non mutually exclusive) archetypes of people using `workflows` - the ones 
employing existing `blocks` to work by combining them together and running as `workflow` using 
`execution engine`. The other use their programming skills to create own `block` and 
seamlessly plug them into `workflows` ecosystem such that other people can use them to create 
their `workflows`. We will cover `blocks` creation topics later on, so far let's focus on 
how you can use `workflows` in your project.

While creating a `workflow` you can use graphical UI or JSON `workflow definition`. UI will ultimately generate
JSON definition, but it is much easier to use. JSON definition can be created manually, but it is
rather something that is needed to ship `workflow` to `execution engine` rather than a tool that 
people uses on the daily basis. 

The first step is picking set of blocks needed to achieve your goal. Then you define `inputs` describing 
either `images` or `parameters` that will become base for processing when `workflow` is run by `execution engine`. 
Those `inputs` are placeholders in `workflow` definition and will be substituted in runtime with actual values. 

Once you defined your `inputs` you need to connect them into appropriate `steps`. You can do it by drawing 
connection between `workflow input` and compatible `step` input. That action creates a `selector` in 
`workflow` definition that references the specific `input` and will be resolved by `execution engine` in runtime.

On the similar note, `steps` are possible to be connected. Output of one step can be referenced as an input to
another `step` given the pair is compatible (we will describe that later or).

At the end of the day, you define `workflow outputs` by selecting and naming outputs of `steps` that you 
want to expose in result of `workflow run` that will be performed by `execution engine`.

And - that's everything to know about `workflows` to hit the ground running.


## What do I need to know to create blocks?

In this section we provide the knowledge required to understand on how to create a custom `workflow block`.
This is not the ultimate guide providing all technical details - instead, this section provides description
for terminology used in more in-depth sections of documentation and lays the foundations required to
understand workflows in-depth.

What users define as JSON `workflow definition` is actually a high-level programming language to define
`workflow`. All the symbols provided in `definition` must be understood and resolved by `execution engine`.
Itself, `execution engine` consist of two components - `workflow compiler` and `workflows executor`.
Compiler is responsible for taking the `workflow definition` and turning it into `execution graph` that 
is understood by `workflows executor`, which is responsible for running the `workflow` against specific
input. Compilation may happen once and after that different input data may be fed to `workflows executor`
to provide results. 

It is important to understand the relation between code that is run by `workflows executor` (defined within
`workflow blocks`) and JSON `workflow definitions` which is parsed and transformed by `workflows compiler`.
First of all - valid block ships the code via implementation of abstract class called `WorkflowBlock`. 
Implementation must define methods to run the computations, but also the methods to provide description
for inputs and outputs of the block. The latter part is probably the difference that programmers would
experience comparing writing a custom code vs creating `workflows block`. 

Block inputs are described by `block manifest` - pydantic entity that defines two obligatory fields (`name` and
`type`) and as many additional properties as needed describing parametrisation of the `block`. pydantic 
manifest serves multiple roles in `workflows`. First of all - it is the source of syntax definition. Whatever
is valid as part of JSON `workflow definition` comes from the structure of `manifests` declared for `blocks`.
Pydantic also validates the syntax of `workflow definition` automatically and provides OpenAPI 3.x compatible
schemas for `blocks` (making it possible to integrate with UI). Additionally, manifest tells `execution engine`
what are the parameters that must be injected to invocation of function to run the `block` logic.

Looking at specific example. The following manifest:
```python
class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "description": "This block produces dynamic crops based on detections from detections-based model.",
            "docs": "https://inference.roboflow.com/workflows/crop",
            "block_type": "transformation",
        }
    )
    type: Literal["Crop"]
    image: Union[InferenceImageSelector, OutputStepImageSelector] = Field(
        description="Reference at image to be used as input for step processing",
        examples=["$inputs.image", "$steps.cropping.crops"],
    )
    predictions: StepOutputSelector(
        kind=[
            OBJECT_DETECTION_PREDICTION_KIND,
            INSTANCE_SEGMENTATION_PREDICTION_KIND,
            KEYPOINT_DETECTION_PREDICTION_KIND,
        ]
    ) = Field(
        description="Reference to predictions of detection-like model, that can be based of cropping "
        "(detection must define RoI - eg: bounding box)",
        examples=["$steps.my_object_detection_model.predictions"],
        validation_alias=AliasChoices("predictions", "detections"),
    )
```
defines manifest for `block` performing image cropping based on detections. One may point few elements:
* native pydantic validation of fields is heavily in use - we want that, as thanks to this mechanism, 
we can automatically validate the syntax and provide all of necessary details about entities validation
to the outside world (for instance to the UI)
* custom types (`InferenceImageSelector`, `OutputStepImageSelector`) are in use and somewhat strange 
concept of `kind` is introduced - we will cover that later
* `model_config` defining extra metadata added for auto-generated `block manifest` schema

Having that manifest, at the level of `workflow definition` you can instantiate the cropping step using:
```json
{
  "type": "crop",
  "name": "my_step",
  "image": "$inputs.image",
  "predictions": "$steps.detection.predictions"
}
```
and that definition will be automatically validated. Looking at the JSON document provided above, 
you can probably see two unusual entries - `"$inputs.image"` and `"$steps.detection.predictions"` - 
those are selectors which we use to reference something that is not possible to be defined statically
while creating `workflow definition`, but will be accessible in runtime - like image and output 
of some previous step.

On the similar note, `block` must declare its outputs - which is list of properties which will be
returned after `block` is run. Those must be declared with name (unique within single `block`) and `kind`
(which will be covered later).

### How data flows through `workflow`?

Knowing the details about blocks it's also good to understand how data is managed by `executor engine`. 
Running `workflow`, user provides actual inputs that are declared in `workflow definition`. Those are 
registered in special cache created for the run. Once steps are run - `execution engine` provides 
all required parameters to steps and register outputs in the cache. Those outputs may be referenced
by other steps by `selectors` in their manifests. At the end of the run - all declared outputs are 
grabbed from the cache.

There are two categories of data that we recognise in `workflows`. Parameters that are "singular" - 
like confidence thresholds or other hyperparameters. Those are declared for all the images that 
are used in processing, not at the level of single batch elements. There are also `workflows`
data that are organised in batches - like input images or outputs from steps. The assumptions are:
* static elements of `manifests` (fields that do not provide selectors) and input parameters of type
`InferenceParameter` are "singular"
* input of type `InferenceImage`, `manifests` fields declared as selectors (apart from step selectors)
and **all step outputs** are assumed batch-major (expect list of things to be provided)


### Selectors and `kind`

Selectors are needed to make references between steps. You can compare them to pointers in such 
programming languages as C or C++. As you probably see - selectors are quite abstract - we do not
even know if something that is selected (input - `$inputs.name`, step - `$steps.my_step` or step 
output `$steps.my_step.output`) even exist. That aspect can be automatically verified by `compiler`.
But even if referred element exists - what is that thing? That would barely impossible to guess in
general case. To solve this problem we introduced simple type system on top of `selectors` - `kinds`.

Rules are quite simple. If you do not define `selector` kind - it is assumed to be `wildcard` - 
equivalent to `Any`. It would work, but `compiler` would have zero knowledge about data that should
be shipped - hence error preventions mechanisms may not fully work. It is possible however to declare
one or many `kind` that is expected for the `selector` (or step output - as those also declare `kinds`).
Many alternatives provided are treated as union of types.

How to define `kind`? Well - `kinds` represent some abstract concept with a name - that's it. There can be
as general `kind` as "number" and as specific as "detection_with_tracking". `blocks` creators should use
`kinds` defined in core of the `workflows` library or create own ones - that will fit their custom `blocks`.

### Stateful nature of `blocks`
Blocks are stateful to make it possible to compile a `workflow` and run it against series of images to 
support for use cases like tracking. It is ensured by the fact that each instance of `block` (`step`) 
is a class that have constructor and maintain it's state. Some classes would require init parameters -
those must be declared in `block` definition and will be resolved and provided by `execution engine`.

## What are the sources of `blocks`?

`Blocks` can be defined withing `workflows` core library, but may also be shipped as custom plugin in form
of python library. This library must define `blocks` and provide two additional elements to fulfill
the contract:
* `load_blocks()` function accessible in `__init__.py` of library that provides `List[BlockSpecification]`
with details of blocks.
* optional dict of `REGISTERED_INITIALIZERS` - with mapping from init parameter name to specific value or
parameter-free method to assembly the value.

To load `blocks` from `plugin` - install the library in your environment, and then add the name of library 
into environmental variable `WORKFLOWS_PLUGINS` - this should be comma separated list of libraries with `plugins`.
And that's it - `plugin` should be automatically loaded.