import argparse
import getpass
import locale
import os
import platform
import sys
import zlib
from typing import Optional, Sequence

import numpy as np
import sentry_sdk
import sentry_sdk.serializer
import sentry_sdk.utils
from sentry_sdk.utils import safe_repr as _safe_repr

import PartSeg
from PartSeg.common_backend.except_hook import my_excepthook
from PartSegCore import state_store
from PartSegCore.utils import numpy_repr


def proper_suffix(val: str):
    """
    check if val contains only alphanumeric characters

    :raise argparse.ArgumentTypeError: on validation error
    """
    if len(val) > 0 and not val.isalnum():
        raise argparse.ArgumentTypeError(f"suffix '{val}' need to contains only alpha numeric characters")
    return val


def proper_path(val: str):
    """
    Check if `val` is proper path in current system

    :raise argparse.ArgumentTypeError: on validation error
    """
    if os.path.exists(val) and os.path.isdir(val):
        return val
    try:
        os.makedirs(val)
    except OSError:
        raise argparse.ArgumentTypeError(f" Path {val} is not a valid path in this system")
    return val


class CustomParser(argparse.ArgumentParser):
    """
    Argument parser with set of predefined flags:

    #. ``--no_report`` - disable error reporting, still showing dialog with information.
       Set :py:data:`.state_store.report_errors`.
    #. ``--no_dialog`` - disable error reporting and showing dialog. Exceptions will be printed on stderr.
       Set :py:data:`.state_store.show_error_dialog`.
    #. ``--no_update`` - disable check for updates on application startup
        Set :py:data:`.state_store.check_for_updates`.
    #. ``--save_suffix`` - add special suffix to directory contains saved state. designed to allow separate projects.
        Set :py:data:`.state_store.save_suffix`.
    #. ``--save_directory`` - set custom save directory. default can be previewed in `Help > State directory`
         from PartSeg main menu, Set :py:data:`.state_store.save_folder`.
    #. ``--inner_plugins`` - designed to hide part plugin from default start.
         Set :py:data:`.state_store.custom_plugin_load`.
    #. ``--develop`` -- for developer purpose. Allow to reload part of Program without restarting. May be unstable.
         Set :py:data:`.state_store.develop`. Base on this :py:class:`PartSeg.common_gui.advanced_tabs.AdvancedWindow`
         constructor add developer tab..

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_argument("--no_report", action="store_false", help="disable error reporting")
        self.add_argument("--no_dialog", action="store_false", help="disable error reporting and showing error dialog")
        self.add_argument("--no_update", action="store_false", help="disable check for updates")
        self.add_argument(
            "--save_suffix",
            "--ssuf",
            type=proper_suffix,
            default=[""],
            help="suffix for configuration_directory",
            nargs=1,
            metavar="suffix",
        )
        self.add_argument(
            "--save_directory",
            "--sdir",
            type=proper_path,
            default=[state_store.save_folder],
            help="path to custom configuration folder",
            nargs=1,
            metavar="path",
        )
        self.add_argument("--inner_plugins", action="store_true", help=argparse.SUPPRESS)
        self.add_argument("--develop", action="store_true", help=argparse.SUPPRESS)

    def parse_args(self, args: Optional[Sequence[str]] = None, namespace: Optional[argparse.Namespace] = None):
        """
        overload of :py:meth:`argparse.ArgumentParser.parse_args`. Set flags like described in class documentation.
        """
        args = super().parse_args(args, namespace)
        state_store.report_errors = args.no_report
        state_store.show_error_dialog = args.no_dialog
        state_store.custom_plugin_load = args.inner_plugins
        state_store.check_for_updates = args.no_update
        state_store.develop = args.develop
        state_store.save_suffix = args.save_suffix[0]
        state_store.save_folder = os.path.abspath(
            args.save_directory[0] + ("_" + state_store.save_suffix if state_store.save_suffix else "")
        )
        if args.no_report and args.no_dialog:
            _setup_sentry()
        sys.excepthook = my_excepthook
        locale.setlocale(locale.LC_NUMERIC, "")
        return args


def _setup_sentry():
    sentry_sdk.utils.MAX_STRING_LENGTH = 10 ** 4
    sentry_sdk.serializer.safe_repr = safe_repr
    sentry_sdk.init(
        "https://d4118280b73d4ee3a0222d0b17637687@sentry.io/1309302",
        release=f"PartSeg@{PartSeg.__version__}",
    )
    with sentry_sdk.configure_scope() as scope:
        scope.set_user(
            {"name": getpass.getuser(), "id": zlib.adler32((getpass.getuser() + "#" + platform.node()).encode())}
        )


def safe_repr(val):
    if isinstance(val, np.ndarray):
        return numpy_repr(val)
    return _safe_repr(val)
