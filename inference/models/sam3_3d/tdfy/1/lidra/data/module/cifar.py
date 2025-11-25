import lightning.pytorch as pl
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

from torchvision.datasets import CIFAR10 as CIFAR10Dataset


class CIFAR10(pl.LightningDataModule):
    def __init__(
        self,
        batch_size,
        data_dir: str = "./",
        download=False,
        random_split=(45000, 5000),
    ):
        super().__init__()

        self.batch_size = batch_size
        self.data_dir = data_dir
        self.download = download
        self.random_split = random_split

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        self.dims = (3, 32, 32)
        self.num_classes = 10

    def prepare_data(self):
        if self.download:
            CIFAR10Dataset(self.data_dir, train=True, download=self.download)
            CIFAR10Dataset(self.data_dir, train=False, download=self.download)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            cifar_full = CIFAR10Dataset(
                self.data_dir, train=True, transform=self.transform
            )
            self.cifar_train, self.cifar_val = random_split(
                cifar_full, self.random_split
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.cifar_test = CIFAR10Dataset(
                self.data_dir, train=False, transform=self.transform
            )

    def train_dataloader(self):
        return DataLoader(self.cifar_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.cifar_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.cifar_test, batch_size=self.batch_size)
