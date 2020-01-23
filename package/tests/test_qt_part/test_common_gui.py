from enum import Enum

from PartSeg.common_gui.universal_gui_part import EnumComboBox


class Enum1(Enum):
    test1 = 1
    test2 = 2
    test3 = 3


class Enum2(Enum):
    test1 = 1
    test2 = 2
    test3 = 3
    test4 = 4

    def __str__(self):
        return self.name


class TestEnumComboBox:
    def test_enum1(self, qtbot):
        widget = EnumComboBox(Enum1)
        qtbot.addWidget(widget)
        assert widget.count() == 3
        assert widget.currentText() == "Enum1.test1"
        with qtbot.waitSignal(widget.current_choose):
            widget.set_value(Enum1.test2)

    def test_enum2(self, qtbot):
        widget = EnumComboBox(Enum2)
        qtbot.addWidget(widget)
        assert widget.count() == 4
        assert widget.currentText() == "test1"
        with qtbot.waitSignal(widget.current_choose):
            widget.set_value(Enum2.test2)
