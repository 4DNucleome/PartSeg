from qtpy.QtWidgets import QComboBox

from PartSegCore.image_operations import RadiusType


class DimComboBox(QComboBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.addItems(["No", "2d", "3d"])

    def value(self):
        return RadiusType(self.currentIndex())

    def setValue(self, val: RadiusType):
        if not isinstance(val, RadiusType):
            val = RadiusType.NO
        self.setCurrentIndex(val.value)
