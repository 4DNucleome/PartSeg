from packaging.version import parse
from qtpy import QT_VERSION
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import QComboBox, QCompleter


class SearchCombBox(QComboBox):
    """
    ComboCox with completer for fast search in multiple options
    """

    if parse(QT_VERSION) < parse("5.14.0"):
        textActivated = Signal(str)  # pragma: no cover

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setEditable(True)
        self.completer_object = QCompleter()
        self.completer_object.setCaseSensitivity(Qt.CaseInsensitive)
        self.setCompleter(self.completer_object)
        # FIXME
        if parse(QT_VERSION) < parse("5.14.0"):  # pragma: no cover
            self.currentIndexChanged.connect(self._text_activated)

    def _text_activated(self):  # pragma: no cover
        self.textActivated.emit(self.currentText())

    def addItem(self, *args):
        super().addItem(*args)
        self.completer_object.setModel(self.model())

    def addItems(self, *args):
        super().addItems(*args)
        self.completer_object.setModel(self.model())
