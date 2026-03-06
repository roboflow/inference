import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Union
import torch
from loguru import logger
import seaborn as sns
import cv2
import numpy as np


def show_image_mask_keys_from_batch(
    samples,
    sample_idx=0,
    title=None,
    show=True,
    image_key="image",
    mask_key="mask",
):
    if not isinstance(samples, dict):
        logger.warning(f"Samples is not a dict, assuming it's a single sample")
        assert isinstance(samples, (tuple, list))
        sample_uuids, samples = samples

    sample = {
        image_key: samples[image_key][sample_idx],
        mask_key: samples[mask_key][sample_idx],
    }

    return show_image_mask_keys_from_sample(
        sample, title, show=show, image_key=image_key, mask_key=mask_key
    )


def show_image_mask_keys_from_sample(
    sample: Dict[str, torch.Tensor],
    title=None,
    show: bool = False,
    image_key: str = "image",
    mask_key: str = "mask",
):
    if image_key not in sample:
        raise ValueError(f"Image key {image_key} not in sample")
    if mask_key not in sample:
        raise ValueError(f"Mask key {mask_key} not in sample")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(sample[image_key].permute(1, 2, 0).cpu().numpy())
    ax1.set_title("Image")
    ax1.axis("off")

    if mask_key in sample:
        ax2.imshow(sample[mask_key].squeeze().cpu().numpy(), cmap="gray")
        ax2.set_title("Mask")
        ax2.axis("off")

    if title:
        plt.suptitle(title)
    plt.tight_layout()
    if show:
        print("Showing figure")
        # plt.show()
    return fig


def show_images_row(
    images: Union[List[torch.Tensor], torch.Tensor],
    titles: Optional[List[str]] = None,
    figsize: Optional[tuple] = None,
    cmap: Optional[str] = None,
    show: bool = False,
    main_title: Optional[str] = None,
) -> plt.Figure:
    """
    Display multiple images in a row with titles.

    Args:
        images: List of image tensors or a single tensor of shape [N, C, H, W]
        titles: Optional list of titles for each image
        figsize: Optional figure size tuple (width, height)
        cmap: Optional colormap for grayscale images
        show: Whether to call plt.show()
        main_title: Optional title for the entire figure

    Returns:
        matplotlib Figure object
    """
    # Handle both list of tensors and batch tensor
    if isinstance(images, torch.Tensor) and len(images.shape) == 4:
        num_images = images.shape[0]
        is_batch = True
    elif isinstance(images, list):
        num_images = len(images)
        is_batch = False
    else:
        raise ValueError(
            f"Expected list of tensors or tensor of shape [N, C, H, W], got {type(images)}"
        )

    # Calculate figsize if not provided
    if figsize is None:
        figsize = (5 * num_images, 5)

    # Create figure and axes
    fig, axes = plt.subplots(1, num_images, figsize=figsize)
    if num_images == 1:
        axes = [axes]

    # Process each image
    for i in range(num_images):
        if is_batch:
            img = images[i]
        else:
            img = images[i]

        # Convert to numpy and handle channel dimension
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu()
            if len(img.shape) == 3 and img.shape[0] in [1, 3]:
                img = img.permute(1, 2, 0).numpy()
                if img.shape[2] == 1:
                    img = img.squeeze(2)
            else:
                img = img.numpy()

        # Display image
        axes[i].imshow(img, cmap=cmap)
        axes[i].axis("off")

        # Set title if provided
        if titles and i < len(titles):
            axes[i].set_title(titles[i])

    # Set main title if provided
    if main_title:
        plt.suptitle(main_title)

    plt.tight_layout()
    # if show:
    #     plt.show()

    return fig


def colored_mask_tensor(
    masks: torch.Tensor,
    palette: str = "husl",
) -> torch.Tensor:
    """
    Create a tensor of colored masks.

    Args:
        masks: Tensor of shape [N, 1, H, W] or [N, H, W] where N is the number of masks
        background_image: Optional tensor of shape [3, H, W] or [N, 3, H, W] to use as background
        alpha: Transparency of the colored masks
        palette: Seaborn color palette to use

    Returns:
        Tensor of shape [N, H, W, 4] containing colored masks with RGBA channels
    """
    # Ensure masks are the right shape
    if len(masks.shape) == 4 and masks.shape[1] == 1:
        masks = masks.squeeze(1)  # [N, 1, H, W] -> [N, H, W]
    elif len(masks.shape) != 3:
        raise ValueError(
            f"Expected masks of shape [N, 1, H, W] or [N, H, W], got {masks.shape}"
        )

    n_masks = masks.shape[0]
    h, w = masks.shape[1], masks.shape[2]

    # Generate colors using seaborn
    colors = torch.tensor(sns.color_palette(palette, n_masks), device=masks.device)
    alpha = torch.ones((n_masks, 1), dtype=torch.float32, device=masks.device)
    colors = torch.cat([colors, alpha], dim=1)

    # Create output tensor [N, H, W, 4] for RGBA
    mask_rgba = torch.zeros((h, w, 4), dtype=torch.float32, device=masks.device)

    # Process each mask
    for i in range(n_masks):
        mask_rgba[masks[i] > 0.5] = colors[i]

    return mask_rgba


def get_masked_sample_cv2(
    original, mask, add_bbox=True, box_color=(0, 0, 255), box_thickness=2
):  # cv2 version is faster for batch use cases
    # Load images
    original = (original.permute(1, 2, 0).cpu().numpy() * 255)[..., ::-1].astype(
        np.uint8
    )
    mask = mask.squeeze().cpu().numpy()

    # Ensure they are the same size
    assert original.shape[:2] == mask.shape, "Images must be the same shape"
    # Create purple overlay where the mask is
    purple_overlay = np.zeros_like(original)
    purple_overlay[:] = (130, 0, 130)  # Purple in BGR

    # Apply mask
    highlighted = original.copy()
    highlighted[mask > 0] = cv2.addWeighted(
        original[mask > 0], 0.3, purple_overlay[mask > 0], 0.7, 0
    )

    if add_bbox and mask.sum() > 0:
        ys, xs = np.where(mask.astype(bool))
        y_min, y_max = ys.min(), ys.max()
        x_min, x_max = xs.min(), xs.max()
        cv2.rectangle(
            highlighted,
            (x_min, y_min),
            (x_max, y_max),
            color=box_color,
            thickness=box_thickness,
        )

    highlighted_tensor = torch.from_numpy(highlighted).permute(2, 0, 1).float() / 255.0
    return highlighted_tensor
