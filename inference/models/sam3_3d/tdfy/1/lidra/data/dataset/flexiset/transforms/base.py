from lidra.data.dataset.flexiset.args import Args


class Base:
    def __init__(self):
        self._args = Args(self._transform)

    def _transform(self, *args):
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement the _transform method."
        )

    def transform(self, *args):
        transformed = self._transform(*args)
        if len(args) == 1:
            transformed = (transformed,)
        if len(transformed) != len(args):
            raise ValueError(
                f"The number of outputs from the transform does not match the number "
                f"of inputs. Expected {len(args)} but got {len(transformed)}."
            )
        return transformed
