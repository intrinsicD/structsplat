import numpy as np
import pytest

from structsplat.cli import load_image, save_image


def test_save_image_rounds_not_truncates(tmp_path):
    # astype(uint8) floored every pixel down by up to 1/255; np.rint round-trips exact 8-bit
    # values and halves the quantization bias (COMP-002).
    pytest.importorskip("PIL")
    vals = (np.arange(256) / 255.0).astype(np.float32)
    img = np.repeat(vals[None, :, None], 3, axis=2).reshape(1, 256, 3)
    p = tmp_path / "grad.png"
    save_image(str(p), img)
    back = load_image(str(p))
    # every exact k/255 level must survive the round-trip; truncation shifted them down
    assert np.abs(back - img).max() <= 1e-6
