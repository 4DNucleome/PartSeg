import pytest
import PartSeg
import packaging.version


def test_analysis_import():
    try:
        import PartSeg.segmentation_analysis.main_window
    except ImportError:
        pytest.fail("Error in importing segmentation ui")


def test_launcher_import():
    try:
        import PartSeg.launcher.main_window
    except ImportError:
        pytest.fail("Error in importing launcher ui")


def test_segmentation_import():
    try:
        import PartSeg.segmentation_mask.stack_gui_main
    except ImportError:
        pytest.fail("Error in importing mask segmentation ui")

def test_version_string():
    assert isinstance(packaging.version.parse(PartSeg.__version__), packaging.version.Version)
