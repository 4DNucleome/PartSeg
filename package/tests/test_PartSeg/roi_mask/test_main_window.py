# pylint: disable=R0201
import pytest
import qtpy

from PartSeg._roi_mask import main_window as mask_main_window

pyside_skip = pytest.mark.skipif(qtpy.API_NAME == "PySide2", reason="PySide2 problem")


class TestMaskMainWindow:
    # @pytest.mark.skipif((platform.system() == "Linux") and CI_BUILD, reason="vispy problem")
    @pyside_skip
    def test_opening(self, qtbot, tmpdir):
        main_window = mask_main_window.MainWindow(tmpdir, initial_image=False)
        qtbot.addWidget(main_window)
        qtbot.wait(50)

    @pyside_skip
    def test_change_theme(self, qtbot, tmpdir):
        main_window = mask_main_window.MainWindow(tmpdir, initial_image=False)
        qtbot.addWidget(main_window)
        assert main_window.image_view.viewer.theme == "light"
        main_window.settings.theme_name = "dark"
        assert main_window.image_view.viewer.theme == "dark"

    @pyside_skip
    def test_scale_bar(self, qtbot, tmpdir):
        main_window = mask_main_window.MainWindow(tmpdir, initial_image=False)
        qtbot.addWidget(main_window)
        main_window._scale_bar_warning = False
        assert not main_window.image_view.viewer.scale_bar.visible
        main_window._toggle_scale_bar()
        assert main_window.image_view.viewer.scale_bar.visible


class TestImageInformation:
    def test_create(self, stack_settings, qtbot):
        info = mask_main_window.ImageInformation(stack_settings)
        qtbot.addWidget(info)

    def test_multiple_files(self, stack_settings, qtbot):
        info = mask_main_window.ImageInformation(stack_settings)
        qtbot.addWidget(info)

        assert info.multiple_files.isChecked()
        assert stack_settings.get("multiple_files_widget")

        stack_settings.set("multiple_files_widget", False)
        assert not info.multiple_files.isChecked()

        info.multiple_files.setChecked(True)
        assert stack_settings.get("multiple_files_widget")

    def test_sync_dirs(self, stack_settings, qtbot):
        info = mask_main_window.ImageInformation(stack_settings)
        qtbot.addWidget(info)

        assert not info.sync_dirs.isChecked()
        assert not stack_settings.get("sync_dirs")

        stack_settings.set("sync_dirs", True)
        assert info.sync_dirs.isChecked()

        info.sync_dirs.setChecked(False)
        assert not stack_settings.get("sync_dirs")
