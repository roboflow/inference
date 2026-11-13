# Action Recognition Contract

Read what a client must know before an action recognition model's first call,
without downloading its weights. A client that windows video itself and sends
each window to a server needs this before any server answers.

```python
from inference_models import get_action_recognition_contract

contract = get_action_recognition_contract("my-project/3", api_key="...")
print(contract.class_names, contract.video_sampling)
```

The function covers models registered under the action recognition task. It
refuses a model registered under another task, such as the Cosmos3 Edge base
model that a server adapts for zero-shot use. It needs the weights provider,
so it does not work in offline mode.

::: inference_models.get_action_recognition_contract
    options:
      show_root_heading: true
      show_source: false

::: inference_models.ActionRecognitionContract
    options:
      show_root_heading: true
      show_source: false
      members: true
