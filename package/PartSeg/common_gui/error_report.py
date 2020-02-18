"""
THis module contains widgets used for error reporting. The report backed is sentry_.

.. _sentry: https://sentry.io
"""
import sys
import typing

from qtpy.QtWidgets import (
    QDialog,
    QPushButton,
    QTextEdit,
    QHBoxLayout,
    QVBoxLayout,
    QLabel,
    QLineEdit,
    QListWidgetItem,
    QListWidget,
)
import traceback

from sentry_sdk.utils import exc_info_from_error, event_from_exception

from PartSegCore import state_store
import sentry_sdk
from packaging.version import parse as pares_version

from PartSeg import __version__


class ErrorDialog(QDialog):
    """
    Dialog to present user the exception information. User can send error report (possible to add custom information)
    """

    def __init__(self, exception: Exception, description: str, additional_notes: str = "", traceback_summary=None):
        super().__init__()
        self.exception = exception
        self.additional_notes = additional_notes
        self.send_report_btn = QPushButton("Send information")
        self.send_report_btn.setDisabled(not state_store.report_errors)
        self.cancel_btn = QPushButton("Cancel")
        self.error_description = QTextEdit()
        if traceback_summary is None:
            self.error_description.setText(
                "".join(traceback.format_exception(type(exception), exception, exception.__traceback__))
            )
        elif isinstance(traceback_summary, traceback.StackSummary):
            self.error_description.setText("".join(traceback_summary.format()))
        self.error_description.append(str(exception))
        self.error_description.setReadOnly(True)
        self.additional_info = QTextEdit()
        self.contact_info = QLineEdit()

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
        layout.addWidget(self.error_description)
        layout.addWidget(QLabel("Contact information"))
        layout.addWidget(self.contact_info)
        layout.addWidget(QLabel("Additional information from user:"))
        layout.addWidget(self.additional_info)
        if not state_store.report_errors:
            layout.addWidget(
                QLabel(
                    "Sending reports was disabled by runtime flag. "
                    "You can report it manually by creating report on"
                    "https://github.com/4DNucleome/PartSeg/issues"
                )
            )
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.cancel_btn)
        btn_layout.addWidget(self.send_report_btn)
        layout.addLayout(btn_layout)
        self.setLayout(layout)
        exec_info = exc_info_from_error(exception)
        self.exception_tuple = event_from_exception(exec_info)

    def exec(self):
        """
        Check if dialog should be shown  base on :py:data:`state_store.show_error_dialog`.
        If yes then show dialog. Otherwise print exception traceback on stderr.
        """
        # TODO check if this check is needed
        if not state_store.show_error_dialog:
            sys.__excepthook__(type(self.exception), self.exception, self.exception.__traceback__)
            return False
        if pares_version(__version__).is_prerelease or pares_version(__version__).is_devrelease:
            sentry_sdk.capture_exception(self.exception)
        super().exec_()

    def send_information(self):
        """
        Function with construct final error message and send it using sentry.
        """
        with sentry_sdk.configure_scope() as scope:
            text = self.desc.text() + "\n\nVersion: " + __version__ + "\n"
            if len(self.additional_notes) > 0:
                scope.set_extra("additional_notes", self.additional_notes)
            if len(self.additional_info.toPlainText()) > 0:
                scope.set_extra("user_information", self.additional_info.toPlainText())
            if len(self.contact_info.text()) > 0:
                scope.set_extra("contact", self.contact_info.text())
            event, hint = self.exception_tuple

            event["message"] = text
            sentry_sdk.capture_event(event, hint=hint)
        # sentry_sdk.capture_event({"message": text, "level": "error", "exception": self.exception})
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
            super().__init__(f"{type(exception)}: {exception}", parent, QListWidgetItem.UserType)
            self.exception = exception
            self.traceback_summary = None
        else:
            super().__init__(f"{type(exception[0])}: {exception[0]}", parent, QListWidgetItem.UserType)
            self.exception = exception[0]
            self.traceback_summary = exception[1]

        self.setToolTip("Double click for report")


class ExceptionList(QListWidget):
    """
    List to store exceptions
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.itemDoubleClicked.connect(self.item_double_clicked)

    def add_exception(self, exc: Exception):
        """
        Add exception to list
        """
        ExceptionListItem(exc, self)
        if pares_version(__version__).is_prerelease or pares_version(__version__).is_devrelease:
            sentry_sdk.capture_exception(exc)

    @staticmethod
    def item_double_clicked(el: QListWidgetItem):
        """
        if element clicked is :py:class:`ExceptionListItem` then open
        :py:class:`ErrorDialog` for reporting this error.

        This function is connected to :py:meth:`QListWidget.itemDoubleClicked`
        """
        if isinstance(el, ExceptionListItem):
            dial = ErrorDialog(el.exception, "Error during batch processing", traceback_summary=el.traceback_summary)
            dial.exec()
