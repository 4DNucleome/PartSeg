import argparse
import getpass
import locale
import os
import platform
import sys
import zlib
from contextlib import suppress
from importlib.metadata import version as package_version
from typing import Optional, Sequence

import sentry_sdk
import sentry_sdk.serializer
import sentry_sdk.utils
from packaging.version import parse as parse_version

from PartSeg import __version__, state_store
from PartSeg.common_backend.except_hook import my_excepthook
from PartSegCore.utils import safe_repr

SENTRY_GE_1_29 = parse_version(package_version("sentry_sdk")) >= parse_version("1.29.0")


def proper_suffix(val: str):
    """
    check if val contains only alphanumeric characters

    :raise argparse.ArgumentTypeError: on validation error
    """
    if not val or val.isalnum():
        return val
    raise argparse.ArgumentTypeError(f"suffix '{val}' need to contains only alpha numeric characters")


def proper_path(val: str):
    """
    Check if `val` is proper path in current system

    :raise argparse.ArgumentTypeError: on validation error
    """
    if os.path.exists(val) and os.path.isdir(val):
        return val
    try:
        os.makedirs(val)
    except OSError as e:
        raise argparse.ArgumentTypeError(f" Path {val} is not a valid path in this system") from e

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
         constructor add developer tab.

    """

    def __init__(self, *args, **kwargs):
        if "epilog" not in kwargs and len(args) < 4:
            kwargs["epilog"] = "To control server for sentry reporting use PARTSEG_SENTRY_URL environment variable"
        super().__init__(*args, **kwargs)

        self.add_argument("--no_report", action="store_false", help="disable error reporting")
        self.add_argument("--always_report", action="store_true", help="always report errors without asking user")
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
            help=f"path to custom configuration folder, if not set then {state_store.save_folder} will be used. "
            "Could be customized using 'PARTSEG_SETTINGS_DIR' environment variable.",
            nargs=1,
            metavar="path",
        )
        self.add_argument("--inner_plugins", action="store_true", help=argparse.SUPPRESS)
        self.add_argument("--develop", action="store_true", help=argparse.SUPPRESS)
        self.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    def parse_args(self, args: Optional[Sequence[str]] = None, namespace: Optional[argparse.Namespace] = None):
        """
        overload of :py:meth:`argparse.ArgumentParser.parse_args`. Set flags like described in class documentation.
        """
        args = super().parse_args(args, namespace)
        if args.always_report and not args.no_report:
            self.error("Cannot set both --always_report and --no_report")
        state_store.always_report = args.always_report
        state_store.report_errors = args.no_report
        state_store.show_error_dialog = args.no_dialog
        state_store.custom_plugin_load = args.inner_plugins
        state_store.check_for_updates = args.no_update
        state_store.develop = args.develop
        state_store.save_suffix = args.save_suffix[0]
        state_store.save_folder = os.path.abspath(
            args.save_directory[0] + (f"_{state_store.save_suffix}" if state_store.save_suffix else "")
        )

        if args.no_report and args.no_dialog and state_store.sentry_url:
            _setup_sentry()
        sys.excepthook = my_excepthook
        with suppress(locale.Error):
            # some bug in reset locale
            # https://stackoverflow.com/questions/68962248/python-setlocale-with-empty-string-default-locale-gives-unsupported-locale-se
            locale.setlocale(locale.LC_NUMERIC, "")
        return args


def _setup_sentry():  # pragma: no cover
    if not state_store.sentry_url:
        state_store.report_errors = False
        return
    sentry_sdk.utils.MAX_STRING_LENGTH = 10**4
    sentry_sdk.utils.DEFAULT_MAX_VALUE_LENGTH = 10**4
    sentry_sdk.serializer.safe_repr = safe_repr
    sentry_sdk.serializer.MAX_DATABAG_BREADTH = 100
    init_kwargs = {"release": f"PartSeg@{__version__}"}
    if SENTRY_GE_1_29:
        init_kwargs["max_value_length"] = 10**4
    sentry_sdk.init(
        state_store.sentry_url,
        **init_kwargs,
    )
    with sentry_sdk.configure_scope() as scope:
        scope.set_user(
            {
                "name": getpass.getuser(),
                "id": zlib.adler32(f"{getpass.getuser()}#{platform.node()}".encode()),
            }
        )
