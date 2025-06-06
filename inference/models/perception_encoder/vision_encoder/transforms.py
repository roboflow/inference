import torchvision.transforms as T

from core.vision_encoder.tokenizer import SimpleTokenizer


def get_image_transform(
    image_size: int,
    center_crop: bool = False,
    interpolation: T.InterpolationMode = T.InterpolationMode.BILINEAR  # We used bilinear during training
):
    if center_crop:
        crop = [
            T.Resize(image_size, interpolation=interpolation),
            T.CenterCrop(image_size)
        ]
    else:
        # "Squash": most versatile
        crop = [
            T.Resize((image_size, image_size), interpolation=interpolation)
        ]
    
    return T.Compose(crop + [
        T.Lambda(lambda x: x.convert("RGB")),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5], inplace=True),
    ])



def get_text_tokenizer(context_length: int):
    return SimpleTokenizer(context_length=context_length)