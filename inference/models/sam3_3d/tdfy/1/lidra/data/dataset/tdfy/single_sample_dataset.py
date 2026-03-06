from torch.utils.data import Dataset


class SingleSampleDataset(Dataset):
    def __init__(self, sample):
        self.sample = sample

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.sample
