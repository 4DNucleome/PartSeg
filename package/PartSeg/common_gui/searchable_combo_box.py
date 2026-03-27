from packaging.version import parse
from qtpy import QT_VERSION
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import QComboBox, QCompleter


class SearchComboBox(QComboBox):
    """
    ComboCox with completer for fast search in multiple options
    """

    if parse(QT_VERSION) < parse("5.14.0"):
        textActivated = Signal(str)  # pragma: no cover

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setEditable(True)
        self.completer_object = QCompleter()
        self.completer_object.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self.completer_object.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
        self.completer_object.setFilterMode(Qt.MatchFlag.MatchContains)
        self.setCompleter(self.completer_object)
        self.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
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
