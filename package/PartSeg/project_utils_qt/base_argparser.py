import argparse
import sys
import sentry_sdk
from typing import Optional, Sequence, Text
from PartSeg.utils import state_store
from PartSeg.project_utils_qt.except_hook import my_excepthook

class CustomParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_argument("--no_report", action="store_false", help="disable error reporting")
        self.add_argument("--no_dialog", action="store_false", help="disable error reporting and showing error dialog")
        self.add_argument("--no_update", action="store_false", help="disable check for updates")
        self.add_argument("--inner_plugins", action="store_true", help=argparse.SUPPRESS)

    def parse_args(self, args: Optional[Sequence[Text]] = None,
                   namespace: Optional[argparse.Namespace] = None):
        args = super().parse_args(args, namespace)
        state_store.report_errors = args.no_report
        state_store.show_error_dialog = args.no_dialog
        state_store.custom_plugin_load = args.inner_plugins
        state_store.check_for_updates = args.no_update
        if args.no_report and args.no_dialog:
            sentry_sdk.init("https://d4118280b73d4ee3a0222d0b17637687@sentry.io/1309302")
        sys.excepthook = my_excepthook
        return args
