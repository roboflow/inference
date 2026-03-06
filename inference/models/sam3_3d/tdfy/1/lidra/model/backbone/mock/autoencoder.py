import torch


class Encoder(torch.nn.Sequential):
    def __init__(self):
        super().__init__(
            torch.nn.Conv2d(3, 6, kernel_size=5),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(6, 16, kernel_size=5),
            torch.nn.ReLU(True),
        )


class Decoder(torch.nn.Sequential):
    def __init__(self):
        super().__init__(
            torch.nn.ConvTranspose2d(16, 6, kernel_size=5),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(6, 3, kernel_size=5),
            torch.nn.ReLU(True),
        )
