import packaging.version
import pytest

import PartSeg


def test_analysis_import():
    try:
        import PartSeg._roi_analysis.main_window  # noqa: F401
    except ImportError:  # pragma: no cover
        pytest.fail("Error in importing segmentation ui")


def test_launcher_import():
    try:
        import PartSeg._launcher.main_window  # noqa: F401
    except ImportError:  # pragma: no cover
        pytest.fail("Error in importing launcher ui")


def test_segmentation_import():
    try:
        import PartSeg._roi_mask.main_window  # noqa: F401
    except ImportError:  # pragma: no cover
        pytest.fail("Error in importing mask segmentation ui")


def test_version_string():
    assert isinstance(packaging.version.parse(PartSeg.__version__), packaging.version.Version)
