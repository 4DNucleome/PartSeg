"""
THis module contains widgets used for error reporting. The report backed is sentry_.

.. _sentry: https://sentry.io
"""
import getpass
import io
import pprint
import re
import traceback
import typing

import numpy as np
import requests
import sentry_sdk
from napari.settings import get_settings
from napari.utils.theme import get_theme
from qtpy.QtGui import QIcon
from qtpy.QtWidgets import (
    QApplication,
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)
from sentry_sdk.utils import event_from_exception, exc_info_from_error
from traceback_with_variables import Format, print_exc

from PartSeg import __version__
from PartSeg.common_backend.python_syntax_highlight import Pylighter
from PartSegCore import state_store
from PartSegCore.io_utils import find_problematic_leafs
from PartSegCore.segmentation.algorithm_base import SegmentationLimitException
from PartSegCore.utils import numpy_repr

_email_regexp = re.compile(r"[\w+]+@\w+\.\w+")
_feedback_url = "https://sentry.io/api/0/projects/{organization_slug}/{project_slug}/user-feedback/".format(
    organization_slug="cent", project_slug="partseg"
)


class ErrorDialog(QDialog):
    """
    Dialog to present user the exception information. User can send error report (possible to add custom information)
    """

    def __init__(self, exception: Exception, description: str, additional_notes: str = "", additional_info=None):
        super().__init__()
        self.exception = exception
        self.additional_notes = additional_notes
        self.send_report_btn = QPushButton("Send information")
        self.send_report_btn.setDisabled(not state_store.report_errors)
        self.cancel_btn = QPushButton("Cancel")
        self.error_description = QTextEdit()
        theme = get_theme(get_settings().appearance.theme, as_dict=False)
        self._highlight = Pylighter(self.error_description.document(), "python", theme.syntax_style)
        self.traceback_summary = additional_info
        if additional_info is None:
            stream = io.StringIO()
            print_exc(exception, file_=stream, fmt=Format(custom_var_printers=[(np.ndarray, numpy_repr)]))
            self.error_description.setText(stream.getvalue())
        elif isinstance(additional_info, traceback.StackSummary):
            self.error_description.setText("".join(additional_info.format()))
        elif isinstance(additional_info[1], traceback.StackSummary):
            self.error_description.setText("".join(additional_info[1].format()))
        self.error_description.append(str(exception))
        self.error_description.setReadOnly(True)
        self.additional_info = QTextEdit()
        self.contact_info = QLineEdit()
        self.user_name = QLineEdit()
        self.cancel_btn.clicked.connect(self.reject)
        self.send_report_btn.clicked.connect(self.send_information)

        layout = QVBoxLayout()
        self.desc = QLabel(description)
        self.desc.setWordWrap(True)
        info_text = QLabel(
            "If you see these dialog it not means that you do something wrong. "
            "In such case you should see some message box not error report dialog."
        )
        info_text.setWordWrap(True)
        layout.addWidget(info_text)
        layout.addWidget(self.desc)
        layout.addWidget(self.error_description, 1)
        layout.addWidget(QLabel("Contact information"))
        layout.addWidget(self.contact_info)
        layout.addWidget(QLabel("User name"))
        layout.addWidget(self.user_name)
        layout.addWidget(QLabel("Additional information from user:"))
        layout.addWidget(self.additional_info)
        if not state_store.report_errors:
            layout.addWidget(
                QLabel(
                    "Sending reports was disabled by runtime flag. "
                    "You can report it manually by creating report on "
                    "https://github.com/4DNucleome/PartSeg/issues"
                )
            )
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.cancel_btn)
        btn_layout.addWidget(self.send_report_btn)
        layout.addLayout(btn_layout)
        self.setLayout(layout)
        if isinstance(additional_info, tuple):
            self.exception_tuple = additional_info[0], None
        else:
            exec_info = exc_info_from_error(exception)
            self.exception_tuple = event_from_exception(exec_info)

    def exec(self):
        self.exec_()

    def exec_(self):
        """
        Check if dialog should be shown  base on :py:data:`state_store.show_error_dialog`.
        If yes then show dialog. Otherwise print exception traceback on stderr.
        """
        # TODO check if this check is needed
        if not state_store.show_error_dialog:
            print_exc(self.exception)
            return False
        super().exec_()

    def send_information(self):
        """
        Function with construct final error message and send it using sentry.
        """
        with sentry_sdk.push_scope() as scope:
            text = self.desc.text() + "\n\nVersion: " + __version__ + "\n"
            if len(self.additional_notes) > 0:
                scope.set_extra("additional_notes", self.additional_notes)
            if len(self.additional_info.toPlainText()) > 0:
                scope.set_extra("user_information", self.additional_info.toPlainText())
            if len(self.contact_info.text()) > 0:
                scope.set_extra("contact", self.contact_info.text())
            event, hint = self.exception_tuple

            event["message"] = text
            if self.traceback_summary is not None:
                scope.set_extra("traceback", self.error_description.toPlainText())

            event_id = sentry_sdk.capture_event(event, hint=hint)
        if event_id is None:
            event_id = sentry_sdk.hub.Hub.current.last_event_id()

        if len(self.additional_info.toPlainText()) > 0:
            contact_text = self.contact_info.text()
            user_name = self.user_name.text()
            data = {
                "comments": self.additional_info.toPlainText(),
                "event_id": event_id,
                "email": contact_text if _email_regexp.match(contact_text) else "unknown@unknown.com",
                "name": user_name or getpass.getuser(),
            }

            r = requests.post(
                url=_feedback_url,
                data=data,
                headers={"Authorization": "DSN https://d4118280b73d4ee3a0222d0b17637687@sentry.io/1309302"},
            )
            if r.status_code != 200:
                data["email"] = "unknown@unknown.com"
                data["name"] = getpass.getuser()
                requests.post(
                    url=_feedback_url,
                    data=data,
                    headers={"Authorization": "DSN https://d4118280b73d4ee3a0222d0b17637687@sentry.io/1309302"},
                )

        self.accept()


class ExceptionListItem(QListWidgetItem):
    """
    Element storing exception and showing basic information about it

    :param exception: exception or union of exception and traceback
    """

    # TODO Prevent from reporting disc error
    def __init__(
        self, exception: typing.Union[Exception, typing.Tuple[Exception, typing.List]], parent: QListWidget = None
    ):
        if isinstance(exception, Exception):
            traceback_summary = None
        else:
            exception, traceback_summary = exception
        if isinstance(exception, SegmentationLimitException):
            super().__init__(f"{exception}", parent, QListWidgetItem.UserType)
        elif isinstance(exception, Exception):
            super().__init__(f"{type(exception)}: {exception}", parent, QListWidgetItem.UserType)
            self.setToolTip("Double click for report")
        self.exception = exception
        self.additional_info = traceback_summary


class ExceptionList(QListWidget):
    """
    List to store exceptions
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.itemDoubleClicked.connect(self.item_double_clicked)

    @staticmethod
    def item_double_clicked(el: QListWidgetItem):
        """
        if element clicked is :py:class:`ExceptionListItem` then open
        :py:class:`ErrorDialog` for reporting this error.

        This function is connected to :py:meth:`QListWidget.itemDoubleClicked`
        """
        if isinstance(el, ExceptionListItem) and not isinstance(el.exception, SegmentationLimitException):
            dial = ErrorDialog(el.exception, "Error during batch processing", additional_info=el.additional_info)
            dial.exec_()


class DataImportErrorDialog(QDialog):
    def __init__(
        self,
        errors: typing.Dict[str, typing.Union[Exception, typing.List[typing.Tuple[str, dict]]]],
        parent: QWidget = None,
        text: str = "During import data part of the entries was filtered out",
    ):
        super().__init__(parent)
        self.setWindowTitle("Data import error")
        self.setWindowIcon(QIcon(":/icons/error.png"))
        self.setMinimumWidth(500)
        self.setMinimumHeight(500)
        self.setLayout(QVBoxLayout())
        self.setModal(True)
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.layout().setSpacing(0)
        self.errors = errors
        self.error_view = QTreeWidget()
        self.layout().addWidget(QLabel(text))
        self.layout().addWidget(self.error_view)
        self.error_view.setHeaderLabels(["Keys", "Details"])
        for file_path, values in self.errors.items():
            file_item = QTreeWidgetItem(self.error_view, [file_path])
            file_item.setExpanded(True)
            file_item.setFirstColumnSpanned(True)
            if isinstance(values, Exception):
                QTreeWidgetItem(file_item, [str(values)]).setFirstColumnSpanned(True)
                continue
            for key, desc in values:
                problematic_entries = find_problematic_leafs(desc)
                item = QTreeWidgetItem(file_item, [key, str(problematic_entries[0]["__error__"]) + "..."])
                if len(problematic_entries) == 1:
                    QTreeWidgetItem(item, ["", pprint.pformat(problematic_entries[0])])
                if len(problematic_entries) > 1:
                    for entry in problematic_entries:
                        item2 = QTreeWidgetItem(item, ["", str(entry["__error__"])])
                        QTreeWidgetItem(item2, ["", pprint.pformat(entry)])

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        copy_btn = QPushButton("Copy to clipboard")
        copy_btn.clicked.connect(self._copy_to_clipboard)
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(close_btn)
        btn_layout.addWidget(copy_btn)
        self.layout().addLayout(btn_layout)

    def _copy_to_clipboard(self):
        res = ""
        for file_path, values in self.errors.items():
            res += f"{file_path}\n"
            for key, desc in values:
                problematic_entries = find_problematic_leafs(desc)
                for entry in problematic_entries:
                    res += f"{key}: {entry['__error__']}\n{pprint.pformat(entry)}\n\n"

        QApplication.clipboard().setText(res)
