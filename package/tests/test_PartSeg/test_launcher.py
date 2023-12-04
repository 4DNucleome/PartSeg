import pytest
from pydantic import BaseModel
from qtpy import API

from PartSeg._launcher.main_window import MainWindow as LauncherMain
from PartSeg._roi_analysis.main_window import MainWindow as AnalysisMain
from PartSeg._roi_mask.main_window import MainWindow as MaskMain
from PartSeg.common_backend.base_argparser import CustomParser
from PartSeg.launcher_main import create_parser, select_window


class ArgsMock(BaseModel):
    gui: str
    mf: bool = False
    image: str = ""
    batch: bool = False


def test_create_parser():
    assert isinstance(create_parser(), CustomParser)


@pytest.mark.skipif(API.startswith("pyside"), reason="problem with make_napari_viewer fixture")
@pytest.mark.parametrize(
    ("gui", "klass"), [("roi_analysis", AnalysisMain), ("roi_mask", MaskMain), ("launcher", LauncherMain)]
)
def test_select_window(qtbot, gui, klass):
    args = ArgsMock(gui=gui)
    widget = select_window(args)
    assert isinstance(widget, klass)
    widget.close()
    widget.deleteLater()
    qtbot.wait(50)
