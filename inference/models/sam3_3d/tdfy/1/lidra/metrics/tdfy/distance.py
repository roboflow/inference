import torch


def masked_and_summed_error(error_fn, dims=None):
    def _masked_and_summed_error_fn(prediction, target, mask):
        return torch.sum(error_fn(prediction, target) * mask, dim=dims) / torch.sum(
            mask, dims=dims
        )

    return _masked_and_summed_error_fn


def squared_error(prediction, target):
    return torch.pow(prediction - target, 2)


def abs_relative_error(prediction, target):
    return torch.abs(prediction - target) / torch.abs(target)


def abs_log_rel(prediction, target):
    return torch.abs(torch.log(prediction) - torch.log(target))


def is_within_threshold(prediction, target, threshold_val) -> torch.Tensor:
    abr = abs_log_rel(prediction, target)
    return torch.exp(abr) < threshold_val


def delta1_acc(prediction, target):
    return is_within_threshold(prediction, target, 1.25)


def delta2_acc(prediction, target):
    return is_within_threshold(prediction, target, 1.25**2)


def delta3_acc(prediction, target):
    return is_within_threshold(prediction, target, 1.25**3)
