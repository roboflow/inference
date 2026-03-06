INIT_FUNCTIONS = {}
SINGLETONS = {}


def register(name: str, init_fn):
    if name in SINGLETONS:
        raise RuntimeError('singleton "{name}" already registered')
    INIT_FUNCTIONS[name] = init_fn


def get(name: str):
    if name not in SINGLETONS:
        # initialize singleton
        if name not in INIT_FUNCTIONS:
            raise RuntimeError('singleton "{name}" has been registered')
        SINGLETONS[name] = INIT_FUNCTIONS[name]()
    return SINGLETONS[name]
