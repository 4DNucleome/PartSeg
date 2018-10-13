from PyQt5.QtWidgets import QComboBox

from project_utils.image_operations import RadiusType


class DimComboBox(QComboBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.addItems(["No", "2d", "3d"])

    def value(self):
        return(RadiusType(self.currentIndex()))

    def setValue(self, val:RadiusType):
        self.setCurrentIndex(val.value)