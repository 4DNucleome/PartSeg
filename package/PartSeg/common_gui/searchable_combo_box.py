from qtpy.QtCore import Qt
from qtpy.QtWidgets import QComboBox, QCompleter


class SearchCombBox(QComboBox):
    """
    ComboCox with completer for fast search in multiple options
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setEditable(True)
        self.completer_object = QCompleter()
        self.completer_object.setCaseSensitivity(Qt.CaseInsensitive)
        self.setCompleter(self.completer_object)

    def addItem(self, *args):
        super().addItem(*args)
        self.completer_object.setModel(self.model())

    def addItems(self, *args):
        super().addItems(*args)
        self.completer_object.setModel(self.model())
