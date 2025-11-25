from lidra.data.dataset.flexiset.loaders.base import Base


class Identity(Base):
    def _load(self, data):
        return data
