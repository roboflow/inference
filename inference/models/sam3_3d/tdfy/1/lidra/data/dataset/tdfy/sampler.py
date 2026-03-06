import torch


def concatenate_and_weight_datasets(dataset_dict, dataset_weights):
    keys = list(dataset_dict.keys())
    datasets = [dataset_dict[key] for key in keys]
    datasets_weights = [dataset_weights[key] for key in keys]

    samples_weight = []
    for dataset, dataset_weight in zip(datasets, datasets_weights):
        _datasets_samples_weights = torch.ones(len(dataset)).float() * 1.0
        _datasets_samples_weights = (
            _datasets_samples_weights * dataset_weight / len(dataset)
        )
        samples_weight.append(_datasets_samples_weights)

    concat_dataset = torch.utils.data.dataset.ConcatDataset(datasets)
    samples_weight = torch.cat(samples_weight, dim=0)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(
        samples_weight, len(samples_weight)
    )
    return {"dataset": concat_dataset, "sampler": sampler}


def replicate_dataset(dataset, num_replicas=1):
    return torch.utils.data.dataset.ConcatDataset([dataset] * num_replicas)


def basic_dataset(length=100):
    # Create random tensor data
    data = torch.randn(length, 10)  # 100 samples, 10 features each
    targets = torch.randint(0, 2, (length,))  # Binary targets

    # Create TensorDataset
    tensor_dataset = torch.utils.data.TensorDataset(data, targets)
    return tensor_dataset
