# `Condition`

You can use the `Condition` block to control the flow of a workflow based on the result of a step.

!!! important

    `Condition` step is only capable to operate, when single image is provided to the input of the `workflow` (or more precisely, both `left` and `right` if provided with reference, then the reference can only hold value for a result of operation made against single input). This is to prevent situation when evaluation of condition for multiple images yield different execution paths.  

## Step parameters
* `type`: must be `Condition` (required)
* `name`: must be unique within all steps - used as identifier (required)
* `left`: left operand of `operator`, can be actual value, reference to input or step output (required)
* `right`: left operand of `operator`, can be actual value, reference to input or step output (required)
* `operator`: one of `equal`, `not_equal`, `lower_than`, `greater_than`, `lower_or_equal_than`, `greater_or_equal_than`, 
`in`, `str_starts_with` (meaning `left` ends with `right`), `str_ends_with` (meaning `left` starts with `right`), 
`str_contains` (meaning `left` contains `right`) (required)
* `step_if_true`: reference to the step that will be executed if condition is true (required)
* `step_if_false`: reference to the step that will be executed if condition is false (required)