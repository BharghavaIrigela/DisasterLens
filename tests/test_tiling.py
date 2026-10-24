import numpy as np
import pytest

from disasterlens.preprocessing.tiling import crop, tile_grid


def test_grid_covers_whole_image():
    tiles = tile_grid(1024, 1024, 512, 64)
    cover = np.zeros((1024, 1024), dtype=bool)

    for tile in tiles:
        cover[
            tile.y:tile.y + tile.size,
            tile.x:tile.x + tile.size
        ] = True

    assert cover.all()


def test_coordinates_round_trip():
    image = np.arange(1024 * 1024).reshape(1024, 1024)

    for tile in tile_grid(1024, 1024, 512, 64):
        cropped = crop(image, tile)
        assert cropped[5, 7] == image[tile.y + 5, tile.x + 7]
        assert tile.to_image_xy(7, 5) == (tile.x + 7, tile.y + 5)


def test_invalid_arguments():
    with pytest.raises(ValueError):
        tile_grid(256, 256, 512, 0)

    with pytest.raises(ValueError):
        tile_grid(1024, 1024, 512, 512)