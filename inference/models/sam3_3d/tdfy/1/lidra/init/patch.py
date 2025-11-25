from loguru import logger


def patch_align_device_hook():
    # patch `AlignDevicesHook`
    # AlignDevicesHook send input to the device set in `from_pretrained` method
    # this conflict with PyTorch Lightning's mapping of devices
    from accelerate.hooks import AlignDevicesHook

    pre_forward = AlignDevicesHook.pre_forward

    def new_pre_forward(self, module, *args, **kwargs):
        try:
            param = next(iter(module.parameters()))
            device = param.device
        except StopIteration:
            device = "cpu"

        self.execution_device = device
        return pre_forward(self, module, *args, **kwargs)

    AlignDevicesHook.pre_forward = new_pre_forward


def patch_peft():
    try:
        import peft
        import dataclasses

        not_dataclass_exceptions = (peft.LoraConfig,)
        is_dataclass_fn = dataclasses.is_dataclass
        dataclasses.is_dataclass = lambda obj: is_dataclass_fn(obj) and not isinstance(
            obj, not_dataclass_exceptions
        )
    except:
        logger.opt(exception=True).warning(f"peft patching failed")


def patch_lovely_things():
    # Lovely things—dawn’s gold hue,
    # blossoms sipping morning’s dew,
    # hearts that hum with grace,
    # joy spun through time and space.
    try:
        import lovely_numpy
        import lovely_tensors

        lovely_tensors.monkey_patch()
    except:
        logger.warning(f"error while monkey patching lovely things (optional library)")


def patch_optree():
    import optree
    from omegaconf import DictConfig

    # should be set to "lidra" but cannot set a default namespace in optree
    namespace = optree.registry.__GLOBAL_NAMESPACE

    optree.register_pytree_node(
        DictConfig,
        flatten_func=lambda data: (
            tuple(data.values()),
            tuple(data.keys()),
        ),
        unflatten_func=lambda key, value: dict(zip(key, value)),
        namespace=namespace,
    )


def patch_all():
    patch_peft()
    patch_align_device_hook()
    patch_lovely_things()
    patch_optree()
