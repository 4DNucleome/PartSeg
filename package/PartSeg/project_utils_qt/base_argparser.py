import argparse
import sys
import sentry_sdk
import os
from typing import Optional, Sequence, Text
from PartSeg.utils import state_store
from PartSeg.project_utils_qt.except_hook import my_excepthook


def proper_suffix(val: str):
    if len(val) > 0 and not val.isalnum():
        raise argparse.ArgumentTypeError(f"suffix '{val}' need to contains only alpha numeric characters")
    return val


def proper_path(val: str):
    if os.path.exists(val) and os.path.isdir(val):
        return val
    try:
        os.makedirs(val)
    except OSError:
        raise argparse.ArgumentTypeError(f" Path {val} is not a valid path in this system")
    return val


class CustomParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_argument("--no_report", action="store_false", help="disable error reporting")
        self.add_argument("--no_dialog", action="store_false", help="disable error reporting and showing error dialog")
        self.add_argument("--no_update", action="store_false", help="disable check for updates")
        self.add_argument("--save_suffix", "--ssuf", type=proper_suffix, default=[""],
                          help="suffix for configuration_directory", nargs=1, metavar="suffix")
        self.add_argument("--save_directory", "--sdir", type=proper_path, default=[state_store.save_folder],
                          help="path to custom configuration folder", nargs=1, metavar="path")
        self.add_argument("--inner_plugins", action="store_true", help=argparse.SUPPRESS)
        self.add_argument("--develop", action="store_true")

    def parse_args(self, args: Optional[Sequence[Text]] = None,
                   namespace: Optional[argparse.Namespace] = None):
        args = super().parse_args(args, namespace)
        state_store.report_errors = args.no_report
        state_store.show_error_dialog = args.no_dialog
        state_store.custom_plugin_load = args.inner_plugins
        state_store.check_for_updates = args.no_update
        state_store.develop = args.develop
        state_store.save_suffix = args.save_suffix[0]
        state_store.save_folder = args.save_directory[0]
        if args.no_report and args.no_dialog:
            sentry_sdk.init("https://d4118280b73d4ee3a0222d0b17637687@sentry.io/1309302")
        sys.excepthook = my_excepthook
        return args
