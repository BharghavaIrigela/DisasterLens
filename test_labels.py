import json

import numpy as np
import pytest

from disasterlens.data.labels import (IGNORE, load_buildings, parse_wkt_polygon,
                                      rasterize_mask)


def test_parse_wkt_polygon_with_hole():
    rings = parse_wkt_polygon("POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0), (2 2, 4 2, 4 4, 2 4, 2 2))")
    assert len(rings) == 2 and rings[0].shape == (5, 2)


def test_parse_rejects_non_polygon():
    with pytest.raises(ValueError):
        parse_wkt_polygon("MULTIPOLYGON (((0 0, 1 0, 1 1, 0 0)))")


def test_load_and_rasterize(synthetic_label):
    b = load_buildings(synthetic_label)
    assert [x.damage for x in b] == [1, 2, 3, 4, IGNORE]
    assert b[0].rings_lnglat is not None
    mask = rasterize_mask(b, 1024, 1024)
    assert mask[30, 30] == 1 and mask[30, 120] == 2 and mask[30, 220] == 3
    assert mask[30, 320] == 4 and mask[30, 420] == IGNORE and mask[500, 500] == 0


def test_unknown_subtype_raises(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"features": {"xy": [{"properties": {"feature_type": "building",
        "subtype": "weird"}, "wkt": "POLYGON ((0 0, 1 0, 1 1, 0 0))"}]}}))
    with pytest.raises(ValueError):
        load_buildings(p)


def test_missing_file():
    with pytest.raises(FileNotFoundError):
        load_buildings("nope.json")
