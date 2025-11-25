import torch
from typing import Sequence, Union, Optional, Dict
from lightning.pytorch import Callback, LightningModule
from loguru import logger
from contextlib import contextmanager

import lidra.mixhavior as mixhavior
from lidra.utils.model.hash import hash_module, diff_hashed_model


class CannotEMAUpdate(RuntimeError):
    pass


class EMACache(mixhavior.Behavior):
    def __init__(self, decay, device=None, force_fp32=False):
        super().__init__()
        self.decay = decay
        self.original_decay = decay
        self.device = device
        self._parameters = None
        self._swap_state = "ema"  # in {"ema", "model"}
        self._model = None
        self.force_fp32 = force_fp32

    def set_decay(self, decay):
        """Temporarily override the decay value."""
        self.decay = decay

    def restore_decay(self):
        """Restore the original decay value."""
        self.decay = self.original_decay

    @property
    def swap_state(self):
        return self._swap_state

    def _tensors_from_model(self):
        # use cached keys rather than model key when available
        if self._parameters is not None:
            all_named_parameters = dict(self._model.named_parameters())
            named_parameters = {
                key: all_named_parameters[key] for key in self._parameters
            }
        else:
            named_parameters = {
                key: param
                # "requires_grad" filtering should be parametrized ?
                for key, param in self._model.named_parameters()
                if param.requires_grad
            }
        for key, param in named_parameters.items():
            yield key, param.data.detach()

    @staticmethod
    def _clone_tensor(src: torch.Tensor, device=None, dtype=None):
        device = device if device else src.device
        new_data = torch.empty_like(src, device=device, dtype=dtype)
        return new_data

    def _clone_from_model(self):
        if self.force_fp32:
            dtype = torch.float32
        else:
            dtype = None
        self._parameters = {
            key: EMACache._clone_tensor(data, self.device, dtype)
            for key, data in self._tensors_from_model()
        }
        if not len(self._parameters) > 0:
            logger.warning(f"no parameter has been cached")
        self._copy_from_model()

    def _param_operations(self, op_fn):
        assert (
            self._parameters is not None
        ), "ema parameters should have been initialized"
        for key, data in self._tensors_from_model():
            op_fn(self._parameters[key], data)
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # because of potential non-blocking copies

    def _copy_from_model(self):
        self._param_operations(lambda dst, src: dst.copy_(src, non_blocking=True))

    def load_to_model(self):
        self.ensure_in_state("model")

    def unload_from_model(self):
        self.ensure_in_state("ema")

    def swap(self, ensure_in_state=None):
        # swap operator
        def _swap(buff0, buff1):
            tmp = buff1.clone()
            buff1.copy_(buff0)
            buff0.copy_(tmp)

        # check this is the intended swap
        if (ensure_in_state is not None) and ensure_in_state != self._swap_state:
            raise RuntimeError(
                f'EMA swapping expecting to be in state "{ensure_in_state}", found to be in state "{self._swap_state}"'
            )

        # if we switch to cache the model, we need to make sure the model isn't loaded with some other ema
        if self._swap_state == "ema":
            self._ensure_all_emas_are_in_ema_state(self.mixh)

        # do the swap
        self._param_operations(_swap)
        self._swap_state = {"ema": "model", "model": "ema"}[self._swap_state]

    def ema_update(self):
        if self._swap_state != "ema":
            raise CannotEMAUpdate('cannot make an ema update in "model" state')

        # EMA update : ema -> decay * ema + (1 - decay) * x
        def ema_update(ema, x):
            x = x * (1.0 - self.decay)
            x = x.to(device=ema.device, dtype=ema.dtype)
            ema.mul_(self.decay)
            ema.add_(x)

        self._param_operations(ema_update)

    def ema_load(self):
        self.ensure_in_state("model")

    def ema_unload(self):
        self.ensure_in_state("ema")

    @contextmanager
    def ema_loaded(self):
        self.ema_load()
        try:
            yield self
        finally:
            self.ema_unload()

    def parameters(self):
        if self._parameters:
            yield from self._parameters.items()

    def _ensure_all_emas_are_in_ema_state(self, mixh: mixhavior.Mixhavior):
        model_state_emas = mixh.find_all_of(
            lambda _, b: isinstance(b, EMACache) and b.swap_state == "model"
        )
        assert (
            len(model_state_emas) < 2
        ), f'EMA Cache behavior in weird state (found {len(model_state_emas)}) ema caches in "model" state'
        if len(model_state_emas):
            logger.warning("one ema cache was found ")
            model_state_ema = next(iter(model_state_emas.values()))
            model_state_ema.ensure_in_state("ema")

    def ensure_in_state(self, state):
        if self.swap_state != state:
            self.swap()

    def _move_params_to_cpu(self):
        if self._parameters is None:
            return
        self._parameters = {
            key: data.to("cpu") for key, data in self._parameters.items()
        }

    def _move_params_to_device(self):
        if self._parameters is None:
            return
        if self.device is not None:
            # mode to EMA device
            self._parameters = {
                key: data.to(self.device) for key, data in self._parameters.items()
            }
        else:
            # move to model device
            model_parameters = dict(self._model.named_parameters())
            self._parameters = {
                key: data.to(model_parameters[key].device)
                for key, data in self._parameters.items()
            }

    def _attach(self, mixh):
        self._model = mixh.obj
        self._move_params_to_device()

    def _detach(self):
        self.ensure_in_state("ema")
        self._move_params_to_cpu()
        self._model = None

    def _setup(self, mixh):
        assert isinstance(mixh.obj, torch.nn.Module)
        self._model = mixh.obj
        self._ensure_all_emas_are_in_ema_state(mixh)
        self._clone_from_model()

    def _cleanup(self):
        self._parameters = None

    # serialization methods (ensure we don't store the model and we store EMA weights)
    def __getstate__(self):
        self.ensure_in_state("ema")
        state = self.__dict__.copy()
        del state["_model"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._model = None


class EMA(Callback):
    MIXHAVIOR_PREFIX = "ema_callback/"
    CHECK_WEIGHTS_SYNCHRONIZATION_DEFAULT = False

    def __init__(
        self,
        decays: Union[Sequence[float], Dict[str, float]],
        default_decay_for_eval: Optional[str] = None,
        device=None,
        recover_existing=True,
        check_weights_synchronization=CHECK_WEIGHTS_SYNCHRONIZATION_DEFAULT,
        force_fp32=False,
        start_step=None,
        start_epoch=None,
    ):
        super().__init__()

        self._decays = decays
        self._device = device
        self._recover_existing = recover_existing
        self._check_weights_synchronization = check_weights_synchronization

        self._ema_caches = None

        self.default_decay_for_eval = default_decay_for_eval

        self._force_fp32 = force_fp32
        self.start_step = start_step
        self.start_epoch = start_epoch
        self._is_ema_active = False
        assert not (
            self.start_step is not None and self.start_epoch is not None
        ), "start_step and start_epoch cannot be set at the same time"

    def _update_is_ema_active(self, trainer) -> float:
        _was_ema_active = self._is_ema_active
        _is_ema_active = False
        if self.start_step is not None:
            _is_ema_active = trainer.global_step >= self.start_step
        elif self.start_epoch is not None:
            _is_ema_active = trainer.current_epoch >= self.start_epoch
        else:
            _is_ema_active = True
        if _was_ema_active != _is_ema_active:
            logger.debug(f"EMA is now {'active' if _is_ema_active else 'inactive'}")
        self._is_ema_active = _is_ema_active
        return _is_ema_active

    @property
    def default_decay_for_eval(self):
        return self._default_decay_for_eval

    @default_decay_for_eval.setter
    def default_decay_for_eval(self, name):
        assert (
            name in self._decays
        ), f'"{name}" should be one of the decays {set(self._decays)}'

        self._default_decay_for_eval = name

    @property
    def default_ema_cache(self) -> EMACache:
        name = self._default_decay_for_eval
        if (self._ema_caches is not None) and (name in self._ema_caches):
            return self._ema_caches[name]
        return None

    def check_weights_are_synced(self, model: LightningModule):
        if not self._check_weights_synchronization:
            return
        hashed_model = hash_module(model)
        hashed_model_rank_0 = model.trainer.strategy.broadcast(hashed_model, src=0)

        if hashed_model_rank_0 != hashed_model:
            raise RuntimeError(
                f"model were found to be out-of-sync",
                diff_hashed_model(hashed_model, hashed_model_rank_0),
            )

    def _recover_existing_caches(self, pl_module):
        mixh = mixhavior.get_mixhavior(pl_module)
        ema_caches = mixh.find_all_of_prefix(EMA.MIXHAVIOR_PREFIX)
        for mixh_name in ema_caches:
            name = mixh_name.removeprefix(EMA.MIXHAVIOR_PREFIX)
            if name in self._decays:
                if mixh[mixh_name].decay != self._decays[name]:
                    raise RuntimeError(
                        f'ema cache "{name}" has been found in model, but decay is different ({mixh[mixh_name].decay} != {self._decays[name]})'
                    )
                else:
                    logger.debug(
                        f"recovering EMA cache (name={name}, decay={mixh[mixh_name].decay})"
                    )
                    self._ema_caches[name] = mixh[mixh_name]

    def _create_missing_caches(self, pl_module):
        mixh = mixhavior.get_mixhavior(pl_module)
        for name in self._decays:
            # if already created
            if name in self._ema_caches:
                continue
            behavior = EMACache(
                decay=self._decays[name],
                device=self._device,
                force_fp32=self._force_fp32,
            )
            mixh_name = mixh.equip(behavior, name, prefix=EMA.MIXHAVIOR_PREFIX)
            logger.debug(
                f"creating new EMA cache (name={name}, decay={mixh[mixh_name].decay})"
            )
            self._ema_caches[name] = mixh[mixh_name]

    def _ensure_initialized_caches(self, model, load_caches_from_model=False) -> None:
        # set caches on first setup
        if self._ema_caches is None:
            self._ema_caches = {}
            if self._recover_existing:
                self._recover_existing_caches(model)

        if load_caches_from_model:
            self._create_missing_caches(model)

    # initialization callbacks
    def on_train_start(self, trainer, pl_module):
        # ensure ema caches are ready
        self.check_weights_are_synced(pl_module)
        self._ensure_initialized_caches(pl_module, load_caches_from_model=True)

    def _on_eval_start(self, trainer, pl_module):
        # ensure ema caches are ready
        self.check_weights_are_synced(pl_module)
        self._ensure_initialized_caches(pl_module, load_caches_from_model=False)

    on_validation_start = _on_eval_start
    on_test_start = _on_eval_start
    on_predict_start = _on_eval_start

    # training updates callbacks
    def on_train_epoch_start(self, trainer, pl_module):
        self.check_weights_are_synced(pl_module)
        if self.default_ema_cache is not None:
            assert self.default_ema_cache.obj == pl_module, "pl_module was changed"
        # make sure it's unloaded for the updates
        logger.debug(f"unloading all EMA weights")
        mixhavior.Mixcaller(self._ema_caches).ema_unload()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.check_weights_are_synced(pl_module)
        if self.default_ema_cache is not None:
            assert self.default_ema_cache.obj == pl_module, "pl_module was changed"

        if not self._update_is_ema_active(trainer):
            mixhavior.Mixcaller(self._ema_caches).set_decay(0.0)
        mixhavior.Mixcaller(self._ema_caches).ema_update()
        mixhavior.Mixcaller(self._ema_caches).restore_decay()

    def _on_eval_epoch_start(self, trainer, pl_module):
        self.check_weights_are_synced(pl_module)
        self._ensure_initialized_caches(pl_module, load_caches_from_model=True)
        if self.default_ema_cache is not None:
            assert self.default_ema_cache.obj == pl_module, "pl_module was changed"
            logger.debug(
                f"loading EMA weights (name={self.default_decay_for_eval}, decay={self.default_ema_cache.decay})"
            )
            self.default_ema_cache.ema_load()  # use ema weight for eval

    def _on_eval_epoch_end(self, trainer, pl_module):
        self.check_weights_are_synced(pl_module)
        if self.default_ema_cache is not None:
            assert self.default_ema_cache.obj == pl_module, "pl_module was changed"
            logger.debug(
                f"unloading EMA weights (name={self.default_decay_for_eval}, decay={self.default_ema_cache.decay})"
            )
            self.default_ema_cache.ema_unload()

    on_validation_epoch_start = _on_eval_epoch_start
    on_validation_epoch_end = _on_eval_epoch_end
    on_test_epoch_start = _on_eval_epoch_start
    on_test_epoch_end = _on_eval_epoch_end
    on_predict_epoch_start = _on_eval_epoch_start
    on_predict_epoch_end = _on_eval_epoch_end
