import torch


# simple model for testing (do not use for any production model)
class MockMLP(torch.nn.Module):
    def __init__(self, input_size=8, hidden_sizes=(16, 32, 16), output_size=8):
        super().__init__()
        self.layers = torch.nn.ModuleList()

        # input
        self.layers.append(torch.nn.Linear(input_size, hidden_sizes[0]))

        # hidden
        for i in range(len(hidden_sizes) - 1):
            self.layers.append(torch.nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))

        # output
        self.layers.append(torch.nn.Linear(hidden_sizes[-1], output_size))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = torch.nn.functional.relu(x)
        return x
