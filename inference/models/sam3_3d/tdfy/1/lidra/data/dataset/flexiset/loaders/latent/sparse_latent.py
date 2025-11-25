import torch

from lidra.data.dataset.flexiset.loaders.base import Base
from lidra.data.dataset.flexiset.loaders.numpy.simple_db import (
    SimpleDB as SimpleNumpyDB,
)


class SparseLatent(Base):
    MEAN = torch.tensor(
        [
            -2.1687545776367188,
            -0.004347046371549368,
            -0.13352349400520325,
            -0.08418072760105133,
            -0.5271206498146057,
            0.7238689064979553,
            -1.1414450407028198,
            1.2039363384246826,
        ]
    )
    STD = torch.tensor(
        [
            2.377650737762451,
            2.386378288269043,
            2.124418020248413,
            2.1748552322387695,
            2.663944721221924,
            2.371192216873169,
            2.6217446327209473,
            2.684523105621338,
        ]
    )

    # TODO(Pierre) should extend the auto-collator to handle custom objects (e.g. if obj.__collate__ is defined)
    # create pseudo sparse tensor to avoid auto-collator
    class PseudoSparseTensor:
        def __init__(self, mean, logvar, coords):
            self.mean = mean
            self.logvar = logvar
            self.coords = coords

    def __init__(self):
        super().__init__()
        self.register_default_loader("data", SimpleNumpyDB())

    def _load(self, data):
        assert "feats" in data, "Missing 'feats' in loaded data"
        assert "logvar" in data, "Missing 'logvar' in loaded data"
        assert "coords" in data, "Missing 'coords' in loaded data"
        sparse_t = SparseLatent.PseudoSparseTensor(
            mean=(torch.from_numpy(data["feats"]) - SparseLatent.MEAN)
            / SparseLatent.STD,
            logvar=(torch.from_numpy(data["logvar"]) - SparseLatent.MEAN)
            / SparseLatent.STD,
            coords=torch.from_numpy(data["coords"]),
        )
        return sparse_t
