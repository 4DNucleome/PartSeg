from qtpy.QtWidgets import QStackedWidget, QComboBox, QWidget, QGridLayout


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
        if isinstance(self.parent().layout(), QGridLayout):
            index = self.parent().layout().indexOf(self)
            self.parent().layout().setColumnStretch(self.parent().layout().getItemPosition(index)[1], 1)
        self.selector.show()

    def hideEvent(self, _):
        if isinstance(self.parent().layout(), QGridLayout):
            self.selector.hide()
            index = self.parent().layout().indexOf(self)
        self.parent().layout().setColumnStretch(self.parent().layout().getItemPosition(index)[1], 0)
