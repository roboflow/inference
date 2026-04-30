# Workflows endpoints

Regarding Workflows, there are fundamental issues to discuss which are far more important than providing full specification of all endpoints - namely:
* how to efficiently represent and transfer data inputs
* how to provide responses to clients (hence big output sizes)
* which way to extend introspection API to give people tools to better understand how the blocks could be used and when certain blocks are available in the environment


## Input data representation (in context of `POST /v2/workflows/run`)

Running workflows always supported two modes:
* pre-defined Workflow which we can support with query params: `workflow_id` and `workflow_version`
* in-lined Workflow - which was carried in the body

Request body was then the career of runtime inputs (images and parameters), as well as potentially JSON definition of workflow itself. The only way of delivering input payload in previous version of API was JSON body - but that itself was not efficient:
* all input images always must have been provided as `base64` bytes inside of JSON payload, which **was slow to create, slow to send (encoding overhead) and slow to deserialize**
*  it was impossible to verify workflow w/o decoding whole input (waste of resources)
*  ..., but - it was relatively convenient

So, it's clear that we can also apply our `compact` or `rich` alternatives also here, for example:
* by keeping simple JSON body format as is at the moment (inefficient but really easy to explain) [obviously, with auth externalised to header]
* use new, more efficient, multi-part request format


### Multi-part requests

> [!NOTE]  
> Let's make an observation. Currently - Workflow definition has references to inputs (`$inputs.image`, `$inputs.confidence` etc). Only due to convention, in runtime we look for those inputs in JSON payload - we can do whatever we want, w/o change to Workflow definition - and even altering behaviour based on request type of `query` parameter.

We can imagine that we support multipart requests, with designated `definition` part which should convey Workflow definition and we bind each `$inputs.<input-name>` to a part name (maybe adding ability to use `query` params as alternatives to ship data.

#### Example requests

```bash
curl -X POST https://serverless.roboflow.com/v2/workflows/run \
    --data-urlencode 'workflow_id=workspace/workflow-url' \
    --data-urlencode 'workflow_version=some-workflows-hash' \
    -F "image=@photo-1.jpg;type=image/jpeg" \
    -F "image=@photo-2.jpg;type=image/jpeg" \
    --data-urlencode 'confidence=0.3'
```

> [!WARNING]  
> What we call `workflow_id`? Looks like it's composition of `workspace-url` and `workflow-url` - given the lack of path-params, we are compacting into query


```bash
curl -X POST https://serverless.roboflow.com/v2/workflows/run \
    -F "image=@photo-1.jpg;type=image/jpeg" \
    -F "image=@photo-2.jpg;type=image/jpeg" \
    -F 'definition={"your": "inlined-workflow-definition"};type=application/json'
    --data-urlencode 'confidence=0.3'
```


## Output data representation (in context of `POST /v2/workflows/run`)

Currently, Workflows EE delivers results in JSON payload, where both images and dense / sparse arrays are JSONified.
We could use method proposed for _models endpoints_, to deliver response in multipart format and stitch references to separate parts in JSON document that we usually were placing raw outputs to JSONify.
