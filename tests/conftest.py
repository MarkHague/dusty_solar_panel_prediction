import pytest
from pathlib import Path
import shutil


@pytest.fixture
def test_images(tmp_path):
    """Fetch the sample images dir"""
    source = Path("tests/fixtures/sample_images")
    test_dir = tmp_path / "images"
    shutil.copytree(source, test_dir)
    return test_dir