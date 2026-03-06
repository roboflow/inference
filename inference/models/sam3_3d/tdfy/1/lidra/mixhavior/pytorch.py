import copy
import torch

import lidra.mixhavior as mixhavior


# equiping a module with that behavior allows the serialization
# of mixhavior's behaviors attached to that module
# example :
# mixh = mixhavior.get_mixhavior(model)
# assert "__mixhavior__" not in model.state_dict()
# mixh.equip(PyTorchMixhaviorStateDictHandling())
# assert "__mixhavior__" in model.state_dict()
class PyTorchMixhaviorStateDictHandling(mixhavior.SingleUse):
    # Fix class.
    # "register_state_dict_post_hook" is trying to add an attribute "_from_public_api" to the hook function
    # however, it cannot be done for a method, this class wraps the method to allow attribute setting and fix the issue
    class _HookCallWrapFix:
        def __init__(self, fn):
            self._fn = fn

        def __call__(self, *args, **kwargs):
            return self._fn(*args, **kwargs)

    def __init__(self):
        super().__init__()
        self._cached_behaviors = None
        self._hook_handles = None

    def _re_equip(self, mixh):
        for name, behavior in self._cached_behaviors.items():
            mixh.equip(behavior, name)
        self._cached_behaviors = None

    def state_dict_pre_hook(self, module, prefix, keep_vars):
        # unequip all behaviors
        mixh = self.mixh
        assert mixh is mixhavior.get_mixhavior(module)

        # cache everything except "self"
        # 1. because hook handles have reference to model (i.e. will be serialized)
        # 2. no need to serialize self since the loading will only occur if `PyTorchMixhaviorStateDictHandling` is already equipped
        self._cached_behaviors = {
            name: mixh.unequip(name)
            for name in mixh.behaviors
            if mixh[name] is not self
        }

    def state_dict_post_hook(self, module, state_dict, prefix, local_metadata):
        # self.mixh isn't available because we are in temporary detached state
        mixh = mixhavior.get_mixhavior(module)

        # copy behaviors to state dict
        state_dict["__mixhavior__"] = copy.deepcopy(self._cached_behaviors)
        # re-equip behaviors
        self._re_equip(mixh)

    def load_state_dict_pre_hook(
        self,
        module,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        if "__mixhavior__" not in state_dict:
            return
        self._cached_behaviors = state_dict["__mixhavior__"]

    def load_state_dict_post_hook(self, module, incompatible_keys):
        if "__mixhavior__" in incompatible_keys.unexpected_keys:
            incompatible_keys.unexpected_keys.remove("__mixhavior__")

        # re-equip behaviors
        if self._cached_behaviors is not None:
            mixh = mixhavior.get_mixhavior(module)
            self._re_equip(mixh)

    def _attach(self, mixh):
        model: torch.nn.Module = mixh.obj
        self._hook_handles = [
            model.register_state_dict_pre_hook(self.state_dict_pre_hook),
            model.register_state_dict_post_hook(
                PyTorchMixhaviorStateDictHandling._HookCallWrapFix(
                    self.state_dict_post_hook
                )
            ),
            model.register_load_state_dict_pre_hook(self.load_state_dict_pre_hook),
            model.register_load_state_dict_post_hook(self.load_state_dict_post_hook),
        ]

    def _detach(self):
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles = None
