"""
THis module contains widgets used for error reporting. The report backed is sentry_.

.. _sentry: https://sentry.io
"""
import getpass
import re
import sys
import traceback
import typing

import requests
import sentry_sdk
from qtpy.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
)
from sentry_sdk.utils import event_from_exception, exc_info_from_error

from PartSeg import __version__
from PartSegCore import state_store
from PartSegCore.segmentation.algorithm_base import SegmentationLimitException

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
        self.traceback_summary = additional_info
        if additional_info is None:
            self.error_description.setText(
                "".join(traceback.format_exception(type(exception), exception, exception.__traceback__))
            )
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
        layout.addWidget(self.error_description)
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
        """
        Check if dialog should be shown  base on :py:data:`state_store.show_error_dialog`.
        If yes then show dialog. Otherwise print exception traceback on stderr.
        """
        # TODO check if this check is needed
        if not state_store.show_error_dialog:
            sys.__excepthook__(type(self.exception), self.exception, self.exception.__traceback__)
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
            dial.exec()
