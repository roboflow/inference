import torch
import hashlib


def hash_tensor(tensor: torch.Tensor):
    bytes = tensor.cpu().numpy().tobytes()
    sha256_hash = hashlib.sha256(bytes).hexdigest()
    return sha256_hash


def hash_parameters(module: torch.nn.Module):
    return {key: hash_tensor(param.data) for key, param in module.named_parameters()}


def hash_buffers(module: torch.nn.Module):
    return {key: hash_tensor(buff) for key, buff in module.named_buffers()}


def hash_module(module: torch.nn.Module):
    return {
        "parameters": hash_parameters(module),
        "buffers": hash_buffers(module),
    }


def diff_hashed_model(module_0, module_1):
    return {
        "parameters": diff_hashed_dict(module_0["parameters"], module_1["parameters"]),
        "buffers": diff_hashed_dict(module_0["buffers"], module_1["buffers"]),
    }


def diff_hashed_dict(hashes_0, hashes_1):
    result = {}
    keys = set(hashes_0.keys()) | set(hashes_1.keys())
    for key in keys:
        v0 = hashes_0.get(key, None)
        v1 = hashes_1.get(key, None)
        if v0 != v1:
            result[key] = (v0, v1)
    return result
