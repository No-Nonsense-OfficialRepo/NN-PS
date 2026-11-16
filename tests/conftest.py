import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def ph():
    """The engine module (photo_helper.py). Import errors show up as test errors."""
    import photo_helper
    return photo_helper