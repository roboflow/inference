# Workflows endpoints

Regarding Workflows, there are fundamental issues to discuss which are far more important than providing full specification of all endpoints - namely:
* how to efficiently represent and transfer data inputs
* how to provide responses to clients (hence big output sizes)
* which way to extend introspection API to give people tools to better understand how the blocks could be used and when certain blocks are available in the environment


## POST `/v2/workflows/run`

### Input data representation (in context of `POST /v2/workflows/run`)

Running workflows always supported two modes:
* pre-defined Workflow which we can support with query params: `workflow_id` and `workflow_version` (shipped via control query params)
* in-lined Workflow - which was carried in the body (JSON variant - alongside `inputs`, add `workflow_definition` - multipart variant - `workflow_definition` part)

Request body was then the career of runtime inputs (images and parameters), as well as potentially JSON definition of workflow itself. The only way of delivering input payload in previous version of API was JSON body - but that itself was not efficient:
* all input images always must have been provided as `base64` bytes inside of JSON payload, which **was slow to create, slow to send (encoding overhead) and slow to deserialize**
*  it was impossible to verify workflow w/o decoding whole input (waste of resources)
*  ..., but - it was relatively convenient

We want to follow the same ideas as for the models run (agent comment - please elaborate)


### Output data representation (in context of `POST /v2/workflows/interface`)

The same as for models (agent comment - please elaborate), just adjust schema to reflect workflows specifics + 
more details regarding batching, empty values etc.



## Other endpoints
Explain that those are analogue for existing implmentation, not that much of changes

Optional addition - /validate could optionally verify workflow runtime (check if we can execute regarding the environment vs block requirements and point errors)