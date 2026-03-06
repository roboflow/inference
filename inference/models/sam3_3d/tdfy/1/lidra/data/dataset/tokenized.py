from lidra.data.dataset.mapped import mapped


def tokenized(
    dataset,
    tokenizer,
    subkey=None,
    **kwargs,
):
    if subkey is None:
        fn = tokenizer
    else:

        def fn(item, **kwargs):
            return tokenizer(item[subkey], **kwargs)

    return mapped(dataset, fn, **kwargs)
