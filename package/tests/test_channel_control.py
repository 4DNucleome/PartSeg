from qtpy.QtGui import QImage
from qtpy.QtCore import Qt, QPoint
import numpy as np
from PartSeg.common_gui.channel_control import ColorComboBox, ColorComboBoxGroup
from PartSeg.project_utils_qt.settings import ViewSettings

from PartSeg.utils.color_image import color_image


def array_from_image(image: QImage):
    size = image.size().width() * image.size().height()
    return np.frombuffer(image.bits().asstring(size*3), dtype=np.uint8)


def test_color_combo_box(qtbot):
    box = ColorComboBox(0, ["BlackBlue", "BlackGreen", "BlackMagenta", "BlackRed", "gray"])
    box.show()
    qtbot.add_widget(box)
    with qtbot.waitSignal(box.channel_visible_changed):
        with qtbot.assertNotEmitted(box.clicked):
            qtbot.mouseClick(box.check_box, Qt.LeftButton)
    with qtbot.waitSignal(box.clicked, timeout=1000):
        qtbot.mouseClick(box, Qt.LeftButton, pos=QPoint(5, 5))
    with qtbot.assertNotEmitted(box.clicked):
        qtbot.mouseClick(box, Qt.LeftButton, pos=QPoint(box.width() - 5, 5))

    box.set_color("BlackMagenta")
    img = color_image(np.arange(0, 256).reshape((1, 256, 1)), ["BlackMagenta"], [(0, 256)])
    assert np.all(array_from_image(box.image) == img.flatten())


class TestColorComboBoxGroup:
    def test_change_channels_num(self, qtbot):
        settings = ViewSettings()
        box = ColorComboBoxGroup(settings, "test", 30)
        qtbot.add_widget(box)
        box.set_channels(1)
        box.set_channels(4)
        box.set_channels(10)
        box.set_channels(4)
        box.set_channels(10)
        box.set_channels(2)

    def test_color_combo_box_group(self, qtbot):
        settings = ViewSettings()
        box = ColorComboBoxGroup(settings, "test", 30)
        qtbot.add_widget(box)
        box.set_channels(3)
