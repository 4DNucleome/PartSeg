from qtpy.QtWidgets import QComboBox, QStackedWidget, QWidget


class StackedWidgetWithSelector(QStackedWidget):
    """Stacked widget with selector which can be putted somewhere. Check show and hide event before usage"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.selector = QComboBox()
        self.selector.currentIndexChanged.connect(super().setCurrentIndex)

    # noinspection PyMethodOverriding
    def addWidget(self, widget: QWidget, name: str):
        """register widget"""
        super().addWidget(widget)
        self.selector.addItem(name)

    def setCurrentIndex(self, p: int):
        self.selector.setCurrentIndex(p)

    # noinspection PyMethodOverriding
    def insertWidget(self, p: int, widget: QWidget, name: str):
        super().insertWidget(p, widget)
        self.selector.insertItem(p, name)

    def removeWidget(self, widget: QWidget):
        index = self.indexOf(widget)
        self.selector.removeItem(index)
        super().removeWidget(widget)

    def showEvent(self, _):
        self.selector.show()

    def hideEvent(self, _):
        self.selector.hide()
