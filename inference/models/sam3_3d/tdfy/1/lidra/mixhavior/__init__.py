from typing import Any, Optional
import uuid
import copy
import optree


class DoNotSetup(BaseException):
    def __init__(self):
        super().__init__()


class DoNotAttach(BaseException):
    def __init__(self):
        super().__init__()


class DetachedState(RuntimeError):
    def __init__(self, *args):
        super().__init__(*args)


# a behavior is an horizontal augmentation of existing object
class Behavior:
    def __init__(self):
        self._is_attached = False
        self._is_setup = False
        self._mixh = None

    @property
    def obj(self):
        return self.mixh.obj

    @property
    def mixh(self):
        if not self._is_attached:
            raise DetachedState("cannot run behavior in detached state")
        return self._mixh

    @property
    def is_attached(self):
        return self._is_attached

    @property
    def is_setup(self):
        return self._is_setup

    def __mixh_setup__(self, mixh):
        if not self._is_setup:
            self._setup(mixh)
            self._is_setup = True
        return self

    def __mixh_cleanup__(self):
        self.__mixh_detach__()  # make sure it's detached
        if self._is_setup:
            self._cleanup()
            self._is_setup = False
        return self

    def __mixh_attach__(self, mixh):
        if not self._is_attached:
            self.__mixh_setup__(mixh)  # make sure it's setup
            self._attach(mixh)
            self._mixh = mixh
            self._is_attached = True
        return self

    def __mixh_detach__(self):
        if self._is_attached:
            self._detach()
            self._is_attached = False
        self._mixh = None

    def _setup(self, mixh):
        pass

    def _cleanup(self):
        pass

    def _attach(self, mixh):
        pass

    def _detach(self):
        pass

    def remove(self):
        key = self._mixh.key_of(self)
        if key is not None:
            self._mixh.unequip(key)

    def __del__(self):
        self.__mixh_cleanup__()


class SingleUse(Behavior):
    def __mixh_setup__(self, mixh):
        # do not setup behavior is already installed
        if len(mixh.find_all_of_instance(type(self))) > 0:
            raise DoNotSetup()
        return Behavior.__mixh_setup__(self, mixh)

    def __mixh_attach__(self, mixh):
        # do not setup behavior is already installed
        if len(mixh.find_all_of_instance(type(self))) > 0:
            raise DoNotAttach()
        return Behavior.__mixh_attach__(self, mixh)


class Mixhavior:
    def __init__(self, obj: Any):
        self._obj = obj
        self._behaviors = {}

    def __getitem__(self, name):
        return self._behaviors[name]

    @property
    def obj(self):
        return self._obj

    @property
    def behaviors(self):
        return copy.copy(self._behaviors)

    def equip(
        self,
        behavior: Behavior,
        name: Optional[str] = None,
        prefix: Optional[str] = None,
        override_ok=False,
    ):
        name = Mixhavior._behavior_name(behavior, name, prefix)

        # check if exist
        if (not override_ok) and (name in self._behaviors):
            raise RuntimeError(
                f'behavior "{name}" already exist, set override_ok=True if needs be'
            )
        assert isinstance(
            behavior, Behavior
        ), "cannot equip anything other than instances of `Behavior`"
        try:
            behavior.__mixh_attach__(self)
        except (DoNotSetup, DoNotAttach):
            behavior.__mixh_detach__()
            return None  # skip installing behavior

        self._behaviors[name] = behavior
        return name

    def unequip(self, key):
        behavior = None
        if key in self._behaviors:
            behavior = self._behaviors[key]
            behavior.__mixh_detach__()
            del self._behaviors[key]
        return behavior

    def find_all_of(self, filter_fn):
        return {
            key: behavior
            for key, behavior in self._behaviors.items()
            if filter_fn(key, behavior)
        }

    def key_of(self, behavior):
        me = self.find_all_of(lambda key, x: x is behavior)
        assert len(me) in {0, 1}
        if len(me) > 0:
            return next(iter(me.keys()))
        return None

    def find_all_of_instance(self, klass):
        return self.find_all_of(
            filter_fn=lambda key, behavior: isinstance(behavior, klass)
        )

    def find_all_of_prefix(self, prefix):
        return self.find_all_of(filter_fn=lambda key, behavior: key.startswith(prefix))

    def find_all_of_suffix(self, suffix):
        return self.find_all_of(filter_fn=lambda key, behavior: key.endswith(suffix))

    @staticmethod
    def _behavior_name(
        behavior: Behavior,
        name: Optional[str] = None,
        prefix: Optional[str] = None,
    ):
        if name is None:
            prefix = f"{type(behavior).__name__}/" if prefix is None else prefix
        else:
            prefix = "" if prefix is None else prefix
        name = str(uuid.uuid4()) if name is None else name
        return f"{prefix}{name}"

    def __repr__(self):
        if len(self._behaviors) > 0:
            max_name_size = max(len(name) for name in self._behaviors)
            body = "\n    ".join(
                f"{name.ljust(max_name_size)} : {repr(behavior)}"
                for name, behavior in self._behaviors.items()
            )
        else:
            body = "<empty>"
        return f"""Mixavior
  object : {repr(self._obj)}
  behaviors :
    {body}
"""


# allow calling the same function of multiple behaviors
class Mixcaller:
    class _Caller:
        def __init__(self, behaviors, name):
            self._behaviors = behaviors
            self._name = name

        def __call__(self, *args, **kwargs):
            exec_fn = lambda b: getattr(b, self._name)(*args, **kwargs)
            return optree.tree_map(
                exec_fn,
                self._behaviors,
                is_leaf=lambda b: isinstance(b, Behavior),
                none_is_leaf=False,
            )

    def __init__(self, behaviors):
        self._behaviors = behaviors

    def __getattr__(self, name):
        return Mixcaller._Caller(self._behaviors, name)


def _add_mixhavior(obj):
    obj.__mixhavior__ = Mixhavior(obj)


# check if an object is compatible with mixhavior
def has_mixhavior(obj) -> bool:
    return hasattr(obj, "__mixhavior__")


def get_mixhavior(obj) -> Mixhavior:
    if not has_mixhavior(obj):
        _add_mixhavior(obj)
    return obj.__mixhavior__


def get_behavior(obj, key) -> Behavior:
    return get_mixhavior(obj)[key]
