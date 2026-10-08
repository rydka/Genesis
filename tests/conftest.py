import pytest
from pathlib import Path
from PIL import Image


@pytest.fixture
def project_root():
    return Path(__file__).parent.parent


@pytest.fixture
def sample_image(tmp_path):
    img = Image.new("RGB", (224, 224), color="red")
    path = tmp_path / "test_render.png"
    img.save(path)
    return path


@pytest.fixture
def sample_view_paths(tmp_path):
    colors = {
        "front": "red",
        "back": "blue",
        "left": "green",
        "right": "yellow",
        "top": "purple",
        "iso_3d": "orange",
    }
    views = {}
    for view_name, color in colors.items():
        img = Image.new("RGB", (224, 224), color=color)
        path = tmp_path / f"test_{view_name}.png"
        img.save(path)
        views[view_name] = path
    return views


@pytest.fixture
def chroma_tmp_dir(tmp_path):
    return tmp_path / "test_chroma_db"
