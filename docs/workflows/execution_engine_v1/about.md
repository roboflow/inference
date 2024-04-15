# Roboflow `workflows`

Roboflow `workflows` let you integrate your Machine Learning model into your environment without 
writing a single line of code. You only need to select which steps you need to complete desired
actions and connect them together to create a `workflow`.


## What makes a `workflow`?

As mentioned above, `workflows` are made of `steps`. `Steps` perform specific actions (like 
making prediction from a model or registering results somewhere). They are created from `blocks`, 
which ship the `code` (implementing what's happening behind the scene) and `manifest` (describing 
the way on how `block` can be used). Great news is you do not even need to care about the `code`.
(If you do, that's great - `workflows` only may benefit from your skills). But the clue is - 
there are two (non mutually exclusive) archetypes of people using `workflows` - the ones 
employing existing `blocks` to work by combining them together and running as `workflow` using 
`execution engine`. The other use their programming skills to create own `block` and 
seamlessly plug them into `workflows` ecosystem such that other people can use them to create 
their `workflows`. We will cover `blocks` creation topics later on, so far let's focus on 
how you can use `workflows` in your project.

While creating a `workflow` you can use graphical UI or JSON definition. UI will ultimately generate
JSON definition, but it is much easier to use. JSON definition can be created manually, but it is
rather something that is needed to ship `workflow` to `execution engine` rather than a tool that 
people uses on the daily basis. 

The first step is picking set of blocks needed to achieve your goal. Then you define `inputs` describing 
either `images` or `parameters` that will be basis of  processing when `workflow` is run by `execution engine`. 
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

