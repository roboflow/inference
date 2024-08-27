# Bundling Workflows blocks

To efficiently manage the Workflows ecosystem, a standardized method for building and distributing blocks is 
essential. This allows users to create their own blocks and bundle them into Workflow plugins. A Workflow plugin 
is essentially a Python library that implements a defined interface and can be structured in various ways. 

This page outlines the mandatory interface requirements and suggests a structure for blocks that aligns with 
the [Workflows versioning](/workflows/versioning) guidelines.

## Proposed structure of plugin

We propose the following structure of plugin:

```
.
├── requirements.txt   # file with requirements
├── setup.py           # use different package creation method if you like
├── {plugin_name}
│   ├── __init__.py    # main module that contains loaders
│   ├── kinds.py       # optionally - definitions of custom kinds
│   ├── {block_name}   # package for your block
│   │   ├── v1.py      # version 1 of your block
│   │   ├── ...        # ... next versions
│   │   └── v5.py      # version 5 of your block
│   └── {block_name}   # package for another block
└── tests              # tests for blocks
```

## Required interface

Plugin must only provide few extensions to `__init__.py` in main package
compared to standard Python library:

* `load_blocks()` function to provide list of blocks' classes (required)

* `load_kinds()` function to return all custom kinds the plugin defines (optional)

* `REGISTERED_INITIALIZERS` module property which is a dict mapping name of block 
init parameter into default value or parameter-free function constructing that value - optional 


### `load_blocks()` function

Function is supposed to enlist all blocks in the plugin - it is allowed to define 
a block once.

Example:

```python
from typing import List, Type
from inference.core.workflows.prototypes.block import WorkflowBlock

# example assumes that your plugin name is `my_plugin` and
# you defined the blocks that are imported here
from my_plugin.block_1.v1 import Block1
from my_plugin.block_2.v1 import Block2

def load_blocks() -> List[Type[WorkflowBlock]]:
    return [
        Block1,
        Block2,
]
```

### `load_kinds()` function

`load_kinds()` function to return all custom kinds the plugin defines. It is optional as your blocks
may not need custom kinds.

Example:

```python
from typing import List
from inference.core.workflows.execution_engine.entities.types import Kind

# example assumes that your plugin name is `my_plugin` and
# you defined the imported kind
from my_plugin.kinds import MY_KIND


def load_kinds() -> List[Kind]:
    return [MY_KIND]
```


## `REGISTERED_INITIALIZERS` dictionary

As you know from [page describing Workflows Compiler](/workflows/workflows_compiler/) 
and [blocks development guide](/workflows/create_workflow_block/), Workflow
blocs are dynamically initialised during compilation and may require constructor 
parameters. Those parameters can default into values registered in `REGISTERED_INITIALIZERS`
dictionary. To expose default value for init parameter of your block - 
simply register name of init param and value (of function generating value) in the dictionary.
It is optional part of plugin interface, as not every blocks requires constructor.

Example:

```python
import os


REGISTERED_INITIALIZERS = {
    "param_1": 37,
    "param_2": lambda: os.getenv("MY_ENV_VARIABLE")
}
```

## Enabling plugin in your Workflows ecosystem

To load plugin you must:

* install Python package with the plugin in the environment you run Workflows

* export environment variable `WORKFLOWS_PLUGINS` with comma-separated list of names
of plugins you want to load. 
  
  * Example: to load two plugins `plugin_a` and `plugin_b`, you need to run 
  `export WORKFLOWS_PLUGINS="plugin_a,plugin_b"`
