import json

import numpy as np
import pytest


def _poly(x0, y0, x1, y1):
    return f"POLYGON (({x0} {y0}, {x1} {y0}, {x1} {y1}, {x0} {y1}, {x0} {y0}))"


@pytest.fixture
def synthetic_label(tmp_path):
    """Tiny 1024x1024 post-disaster label with one building per class + one un-classified."""
    specs = [("no-damage", 10, 10, 60, 60), ("minor-damage", 100, 10, 150, 60),
             ("major-damage", 200, 10, 250, 60), ("destroyed", 300, 10, 350, 60),
             ("un-classified", 400, 10, 450, 60)]
    xy = [{"properties": {"feature_type": "building", "subtype": s, "uid": f"u{i}"},
           "wkt": _poly(*c)} for i, (s, *c) in enumerate(specs)]
    ll = [{"properties": {"feature_type": "building", "uid": f"u{i}"},
           "wkt": _poly(77.0 + i * 1e-4, 12.0, 77.0 + i * 1e-4 + 5e-5, 12.0 + 5e-5)} for i in range(len(specs))]
    path = tmp_path / "synthetic_post_disaster.json"
    path.write_text(json.dumps({"features": {"xy": xy, "lng_lat": ll}, "metadata": {}}))
    return path
