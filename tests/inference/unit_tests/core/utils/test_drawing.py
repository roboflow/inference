from typing import List

import numpy as np
import pytest

from inference.core.utils.drawing import _generate_color_image, create_tiles


def test_generate_color_image() -> None:
    # when
    result = _generate_color_image(shape=(192, 168), color=(127, 127, 127))

    # then
    assert np.allclose(result, np.ones((168, 192, 3)) * 127)


def test_create_tiles_with_one_image(
    one_image: np.ndarray, single_image_tile: np.ndarray
) -> None:
    # when
    result = create_tiles(images=[one_image])

    # then
    assert np.allclose(result, single_image_tile, atol=5.0)


def test_create_tiles_with_one_image_and_enforced_grid(
    one_image: np.ndarray, single_image_tile_enforced_grid: np.ndarray
) -> None:
    # when
    result = create_tiles(images=[one_image], grid_size=(None, 3))

    # then
    assert np.allclose(result, single_image_tile_enforced_grid, atol=5.0)


def test_create_tiles_with_two_images(
    two_images: List[np.ndarray], two_images_tile: np.ndarray
) -> None:
    # when
    result = create_tiles(images=two_images)

    # then
    assert np.allclose(result, two_images_tile, atol=5.0)


def test_create_tiles_with_three_images(
    three_images: List[np.ndarray], three_images_tile: np.ndarray
) -> None:
    # when
    result = create_tiles(images=three_images)

    # then
    assert np.allclose(result, three_images_tile, atol=5.0)


def test_create_tiles_with_four_images(
    four_images: List[np.ndarray], four_images_tile: np.ndarray
) -> None:
    # when
    result = create_tiles(images=four_images)

    # then
    assert np.allclose(result, four_images_tile, atol=5.0)


def test_create_tiles_with_all_images(
    all_images: List[np.ndarray], all_images_tile: np.ndarray
) -> None:
    # when
    result = create_tiles(images=all_images)

    # then
    assert np.allclose(result, all_images_tile, atol=5.0)


def test_create_tiles_with_all_images_and_custom_grid(
    all_images: List[np.ndarray], all_images_tile_and_custom_grid: np.ndarray
) -> None:
    # when
    result = create_tiles(images=all_images, grid_size=(3, 3))

    # then
    assert np.allclose(result, all_images_tile_and_custom_grid, atol=5.0)


def test_create_tiles_with_all_images_and_custom_colors(
    all_images: List[np.ndarray], all_images_tile_and_custom_colors: np.ndarray
) -> None:
    # when
    result = create_tiles(
        images=all_images,
        tile_margin_color=(127, 127, 127),
        tile_padding_color=(224, 224, 224),
    )

    # then
    assert np.allclose(result, all_images_tile_and_custom_colors, atol=5.0)


def test_create_tiles_with_all_images_and_custom_grid_to_small_to_fit_images(
    all_images: List[np.ndarray],
) -> None:
    with pytest.raises(ValueError):
        _ = create_tiles(images=all_images, grid_size=(2, 2))
