# pylint: disable=no-self-use
import numpy as np
import pytest

from PartSeg._roi_mask.main_window import ImageInformation, MainWindow
from PartSegCore.mask.io_functions import MaskProjectTuple
from PartSegCore.roi_info import ROIInfo


class TestMaskMainWindow:
    # @pytest.mark.skipif((platform.system() == "Linux") and CI_BUILD, reason="vispy problem")
    @pytest.mark.pyside_skip
    def test_opening(self, qtbot, tmpdir):
        main_window = MainWindow(tmpdir, initial_image=False)
        qtbot.addWidget(main_window)
        qtbot.wait(50)

    @pytest.mark.pyside_skip
    def test_change_theme(self, qtbot, tmpdir):
        main_window = MainWindow(tmpdir, initial_image=False)
        qtbot.addWidget(main_window)
        assert main_window.image_view.viewer.theme == "light"
        main_window.settings.theme_name = "dark"
        assert main_window.image_view.viewer.theme == "dark"

    @pytest.mark.pyside_skip
    def test_scale_bar(self, qtbot, tmpdir):
        main_window = MainWindow(tmpdir, initial_image=False)
        qtbot.addWidget(main_window)
        main_window._scale_bar_warning = False
        assert not main_window.image_view.viewer.scale_bar.visible
        main_window._toggle_scale_bar()
        assert main_window.image_view.viewer.scale_bar.visible

    def test_get_project_info(self, image, tmp_path):
        res = MainWindow.get_project_info(str(tmp_path / "test.tiff"), image)
        assert isinstance(res.roi_info, ROIInfo)
        assert res.roi_extraction_parameters == {}
        assert isinstance(res, MaskProjectTuple)

        roi = np.zeros(image.shape, dtype=np.uint8)
        roi[:, 2:-2] = 1
        res = MainWindow.get_project_info(str(tmp_path / "test.tiff"), image, ROIInfo(roi))
        assert res.roi_extraction_parameters == {1: None}

    @pytest.mark.pyside_skip
    def test_window_title(self, tmpdir, image, image2, qtbot):
        main_window = MainWindow(tmpdir, initial_image=False)
        qtbot.addWidget(main_window)
        assert main_window.windowTitle() == "PartSeg"
        main_window.settings.image = image
        assert main_window.windowTitle().replace("\\", "/") == "PartSeg: test_window_title0/test.tiff"

        image2.file_path = None
        main_window.settings.image = image2
        assert main_window.windowTitle() == "PartSeg"

        image.file_path = ""
        main_window.settings.image = image
        assert main_window.windowTitle() == "PartSeg: "


class TestImageInformation:
    def test_create(self, stack_settings, qtbot):
        info = ImageInformation(stack_settings)
        qtbot.addWidget(info)

    def test_multiple_files(self, stack_settings, qtbot):
        info = ImageInformation(stack_settings)
        qtbot.addWidget(info)

        assert info.multiple_files.isChecked()
        assert stack_settings.get("multiple_files_widget")

        stack_settings.set("multiple_files_widget", False)
        assert not info.multiple_files.isChecked()

        info.multiple_files.setChecked(True)
        assert stack_settings.get("multiple_files_widget")

    def test_sync_dirs(self, stack_settings, qtbot):
        info = ImageInformation(stack_settings)
        qtbot.addWidget(info)

        assert not info.sync_dirs.isChecked()
        assert not stack_settings.get("sync_dirs")

        stack_settings.set("sync_dirs", True)
        assert info.sync_dirs.isChecked()

        info.sync_dirs.setChecked(False)
        assert not stack_settings.get("sync_dirs")
