from typing import Callable
import torch
from copy import deepcopy


class MLP(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        hidden_depth: int,
        output_size: int,
        non_linearity_fn: Callable = torch.nn.functional.relu,
        norm_layer: Callable = None,
        residual: bool = True,
    ):
        super(MLP, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hidden_depth = hidden_depth
        self.output_size = output_size
        self.residual = residual
        self.non_linearity_fn = non_linearity_fn
        norm_layer = (
            torch.nn.LayerNorm(hidden_size) if norm_layer is None else norm_layer
        )

        self.init_layer = torch.nn.Linear(input_size, hidden_size, bias=True)
        self.hidden_layers = torch.nn.ModuleList(
            [
                torch.nn.Linear(hidden_size, hidden_size, bias=True)
                for _ in range(hidden_depth)
            ]
        )
        self.norm_layers = torch.nn.ModuleList(
            [deepcopy(norm_layer) for _ in range(hidden_depth)]
        )
        self.final_layer = torch.nn.Linear(hidden_size, output_size, bias=True)

    def forward(self, x):
        x = self.init_layer(x)
        for layer, norm in zip(self.hidden_layers, self.norm_layers):
            dx = layer(x)
            dx = self.non_linearity_fn(dx)

            if self.residual:
                x = x + dx
            else:
                x = dx

            x = norm(x)
        x = self.final_layer(x)
        return x
