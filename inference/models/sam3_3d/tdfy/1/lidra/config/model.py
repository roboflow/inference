import torch


def freeze(model: torch.nn.Module):
    for param in model.parameters(recurse=True):
        param.requires_grad = False
    model.eval()
    return model
