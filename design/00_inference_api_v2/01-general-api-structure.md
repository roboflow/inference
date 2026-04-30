# API structure

## Models endpoints

* `POST /v2/models/infer` - predict from model
* `GET /v2/models/interface` - discover model interface
* `GET /v2/models/compatibility` - discover models compatible with current server configuration
* `GET /v2/models` - discover loaded models
* `DELETE /v2/models` - unload all models
* `POST /v2/models/load` - load given model
* `POST /v2/models/unload` - unload given model

## Workflows endpoints
* `POST /v2/workflows/run` - run workflow
* `POST /v2/workflows/interface` - disover workflow interface
* `POST /v2/workflows/validate` - validate workflow
* `GET /v2/workflows/system/blocks` - descrbe available blocks
* `GET /v2/workflows/system/definition-schema` - get workflow definition schema
* `GET /v2/workflows/system/engine-versions` - get available engine versions

## Video stream processing 

:TODO

## Server status
* `GET /v2/server/health`
* `GET /v2/server/ready`
* `GET /v2/server/info`
* `GET /v2/server/metrics`
