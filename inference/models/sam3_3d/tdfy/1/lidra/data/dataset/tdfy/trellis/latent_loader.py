import torch


# the stats for trellis latents
# source: https://huggingface.co/JeffreyXiang/TRELLIS-image-large/blob/25e0d31ffbebe4b5a97464dd851910efc3002d96/pipeline.json#L36
STATS = {
    "trellis": {
        "mean": [
            -2.1687545776367188,
            -0.004347046371549368,
            -0.13352349400520325,
            -0.08418072760105133,
            -0.5271206498146057,
            0.7238689064979553,
            -1.1414450407028198,
            1.2039363384246826,
        ],
        "std": [
            2.377650737762451,
            2.386378288269043,
            2.124418020248413,
            2.1748552322387695,
            2.663944721221924,
            2.371192216873169,
            2.6217446327209473,
            2.684523105621338,
        ],
    },
    "depth-vae_500k": {
        "mean": [
            1.09678917,
            0.90180622,
            -1.28466444,
            -0.91574466,
            -1.64116028,
            0.81207581,
            -1.73082393,
            -0.51753748,
        ],
        "std": [
            2.26849692,
            2.03889322,
            2.13405562,
            2.20259538,
            2.05722601,
            1.89593702,
            2.13656548,
            2.48455034,
        ],
    },
    "depth-vae_2.4m": {
        "mean": [
            1.1351601,
            0.81150615,
            -0.85855174,
            -0.8067008,
            -1.3295876,
            0.7872059,
            -1.2165385,
            -0.38523743,
        ],
        "std": [
            2.0564985,
            1.7355415,
            1.9573436,
            1.8759315,
            1.7732683,
            1.5849748,
            1.85003,
            2.2084258,
        ],
    },
    "depth-vae_500k_ambient": {
        "mean": [
            0.68767279,
            0.83831454,
            -1.09745193,
            -1.26545926,
            -2.34965322,
            0.25253675,
            -1.46167297,
            -0.07045046,
        ],
        "std": [
            2.38880902,
            2.07723739,
            2.31934423,
            2.42143944,
            2.14585211,
            1.92875535,
            2.38230733,
            2.13752401,
        ],
    },
    "depth-vae_500k_ambient4k": {
        "mean": [
            0.12211431,
            0.37204156,
            -1.26521907,
            -2.05276058,
            -3.10432536,
            -0.11294304,
            -0.85146744,
            0.45506954,
        ],
        "std": [
            2.37326008,
            2.13174402,
            2.2413953,
            2.30589401,
            2.1191894,
            1.8969511,
            2.41684989,
            2.08374642,
        ],
    },
}


class PseudoSparseTensor:
    def __init__(self, mean, logvar, coords):
        self.mean = mean
        self.logvar = logvar
        self.coords = coords


def load_structure_latents(item):
    out_item = {}
    out_item["mean"] = torch.from_numpy(item["mean"]).reshape(8, -1).transpose(0, 1)
    out_item["logvar"] = torch.from_numpy(item["logvar"]).reshape(8, -1).transpose(0, 1)
    return out_item


def load_sparse_feature_latents(item, version="trellis"):
    # create pseudo sparse tensor to avoid auto-collator
    out_item = {}
    latent_mean = torch.tensor(STATS[version]["mean"])
    latent_std = torch.tensor(STATS[version]["std"])
    sparse_t = PseudoSparseTensor(
        mean=(torch.from_numpy(item["feats"]) - latent_mean) / latent_std,
        logvar=(torch.from_numpy(item["logvar"]) - latent_mean) / latent_std,
        coords=torch.from_numpy(item["coords"]),
    )
    out_item["sparse_t"] = sparse_t
    out_item["feats_mean"] = latent_mean
    out_item["feats_std"] = latent_std
    return out_item
