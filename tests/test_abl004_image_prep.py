import importlib.util
import json
from pathlib import Path


def _load_prepare_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "prepare_abl004_images.py"
    spec = importlib.util.spec_from_file_location("prepare_abl004_images", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_kodak_items_are_the_expected_24_urls():
    mod = _load_prepare_module()
    items = mod.kodak_items()

    assert len(items) == 24
    assert items[0] == ("kodim01.png", "https://r0k.us/graphics/kodak/kodak/kodim01.png")
    assert items[-1] == ("kodim24.png", "https://r0k.us/graphics/kodak/kodak/kodim24.png")


def test_manifest_and_image_list_are_written(tmp_path):
    mod = _load_prepare_module()
    kodak = tmp_path / "kodak24" / "kodim01.png"
    coco = tmp_path / "coco" / "COCO_train2014_000000000009.jpg"
    kodak.parent.mkdir()
    coco.parent.mkdir()
    kodak.write_bytes(b"kodak")
    coco.write_bytes(b"coco")

    manifest = mod.build_manifest([kodak], [coco])
    mod.write_outputs(tmp_path / "out", manifest)

    loaded = json.loads((tmp_path / "out" / "manifest.json").read_text())
    assert [r["split"] for r in loaded["images"]] == ["kodak24", "coco_pinned"]
    assert loaded["images"][0]["url"].endswith("/kodim01.png")
    assert all(len(r["sha256"]) == 64 for r in loaded["images"])
    image_list = (tmp_path / "out" / "images.txt").read_text().splitlines()
    assert image_list == [str(kodak), str(coco)]
