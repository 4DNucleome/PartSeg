import argparse
import sys
import sentry_sdk
from typing import Optional, Sequence, Text
from . import report_utils
from ..project_utils_qt.except_hook import my_excepthook

class CustomParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_argument("--no_report", action="store_false", help="disable error reporting")
        self.add_argument("--no_dialog", action="store_false", help="disable error reporting and showing error dialog")

    def parse_args(self, args: Optional[Sequence[Text]] = None,
                   namespace: Optional[argparse.Namespace] = None):
        args = super().parse_args(args, namespace)
        print(args)
        report_utils.report_errors = args.no_report
        report_utils.show_error_dialog = args.no_dialog
        if args.no_report and args.no_dialog:
            sentry_sdk.init("https://d4118280b73d4ee3a0222d0b17637687@sentry.io/1309302")
        sys.excepthook = my_excepthook
        return args
