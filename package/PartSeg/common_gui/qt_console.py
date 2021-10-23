"""
Code from https://github.com/napari/napari-console
"""
import re
import sys
from typing import TYPE_CHECKING

from ipykernel.connect import get_connection_file
from ipykernel.inprocess.ipkernel import InProcessInteractiveShell
from ipykernel.zmqshell import ZMQInteractiveShell
from IPython import get_ipython
from IPython.terminal.interactiveshell import TerminalInteractiveShell
from qtconsole.client import QtKernelClient
from qtconsole.inprocess import QtInProcessKernelManager
from qtconsole.rich_jupyter_widget import RichJupyterWidget
from qtpy.QtGui import QColor

if TYPE_CHECKING:  # pragma: no cover
    from PartSeg.common_gui.main_window import BaseMainWindow


def str_to_rgb(arg):
    """Convert an rgb string 'rgb(x,y,z)' to a list of ints [x,y,z]."""
    return list(map(int, re.match(r"rgb\((\d+),\s*(\d+),\s*(\d+)\)", arg).groups()))


# set default asyncio policy to be compatible with tornado
# Tornado 6 (at least) is not compatible with the default
# asyncio implementation on Windows
# Pick the older SelectorEventLoopPolicy on Windows
# if the known-incompatible default policy is in use.
# FIXME: if/when tornado supports the defaults in asyncio,
# remove and bump tornado requirement for py38
# borrowed from ipykernel:  https://github.com/ipython/ipykernel/pull/456

if sys.platform.startswith("win") and sys.version_info >= (3, 8):
    import asyncio

    if (
        hasattr(asyncio, "WindowsProactorEventLoopPolicy")
        and hasattr(asyncio, "WindowsSelectorEventLoopPolicy")
        and type(asyncio.get_event_loop_policy()) is asyncio.WindowsProactorEventLoopPolicy  # pylint: disable=C0123
    ):
        # WindowsProactorEventLoopPolicy is not compatible with tornado 6
        # fallback to the pre-3.8 default of Selector
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


class QtConsole(RichJupyterWidget):
    """Qt view for the console, an integrated iPython terminal in napari.
    Parameters
    ----------
    user_variables : dict
        Dictionary of user variables to declare in console name space.
    Attributes
    ----------
    kernel_client : qtconsole.inprocess.QtInProcessKernelClient,
                    qtconsole.client.QtKernelClient, or None
        Client for the kernel if it exists, None otherwise.
    shell : ipykernel.inprocess.ipkernel.InProcessInteractiveShell,
            ipykernel.zmqshell.ZMQInteractiveShell, or None.
        Shell for the kernel if it exists, None otherwise.
    """

    def __init__(self, main_window: "BaseMainWindow"):
        super().__init__()

        self.main_window = main_window
        self.syntax_style = "default"
        self.style_sheet = ""

        # Connect theme update
        user_variables = {"window": self.main_window, "settings": self.main_window.settings}

        # get current running instance or create new instance
        shell = get_ipython()

        if shell is None:
            # If there is no currently running instance create an in-process
            # kernel.
            kernel_manager = QtInProcessKernelManager()
            kernel_manager.start_kernel(show_banner=False)
            kernel_manager.kernel.gui = "qt"

            kernel_client = kernel_manager.client()
            kernel_client.start_channels()

            self.kernel_manager = kernel_manager
            self.kernel_client = kernel_client
            self.shell = kernel_manager.kernel.shell
            self.push = self.shell.push
        elif type(shell) is InProcessInteractiveShell:  # pylint: disable=C0123
            # If there is an existing running InProcessInteractiveShell
            # it is likely because multiple viewers have been launched from
            # the same process. In that case create a new kernel.
            # Connect existing kernel
            kernel_manager = QtInProcessKernelManager(kernel=shell.kernel)
            kernel_client = kernel_manager.client()

            self.kernel_manager = kernel_manager
            self.kernel_client = kernel_client
            self.shell = kernel_manager.kernel.shell
            self.push = self.shell.push
        elif isinstance(shell, TerminalInteractiveShell):
            # if launching from an ipython terminal then adding a console is
            # not supported. Instead users should use the ipython terminal for
            # the same functionality.
            self.kernel_client = None
            self.kernel_manager = None
            self.shell = None
            self.push = lambda var: None

        elif isinstance(shell, ZMQInteractiveShell):
            # if launching from jupyter notebook, connect to the existing
            # kernel
            kernel_client = QtKernelClient(connection_file=get_connection_file())
            kernel_client.load_connection_file()
            kernel_client.start_channels()

            self.kernel_manager = None
            self.kernel_client = kernel_client
            self.shell = shell
            self.push = self.shell.push
        else:
            raise ValueError("ipython shell not recognized; " f"got {type(shell)}")
        # Add any user variables
        user_variables = user_variables or {}
        self.push(user_variables)

        self.enable_calltips = False

        # Set stylings
        # self._update_theme()

        # TODO: Try to get console from jupyter to run without a shift click
        # self.execute_on_complete_input = True

    def _update_theme(self, event=None):
        """Update the napari GUI theme."""
        from napari.qt import get_stylesheet
        from napari.utils.theme import get_theme, template

        # qtconsole unfortunately won't inherit the parent stylesheet
        # so it needs to be directly set
        raw_stylesheet = get_stylesheet()
        # template and apply the primary stylesheet
        # (should probably be done by napari)
        theme = get_theme(self.viewer.theme)
        self.style_sheet = template(raw_stylesheet, **theme)

        # After napari 0.4.6 the following syntax will be allowed
        # self.style_sheet = get_stylesheet(self.viewer.theme)

        # Set syntax styling and highlighting using theme
        self.syntax_style = theme["syntax_style"]
        bracket_color = QColor(*str_to_rgb(theme["highlight"]))
        self._bracket_matcher.format.setBackground(bracket_color)

    def closeEvent(self, event):
        """Clean up the integrated console in napari."""
        # Disconnect theme update
        # self.viewer.events.theme.disconnect(self._update_theme)

        if self.kernel_client is not None:
            self.kernel_client.stop_channels()
        if self.kernel_manager is not None and self.kernel_manager.has_kernel:
            self.kernel_manager.shutdown_kernel()

        # RichJupyterWidget doesn't clean these up
        self._completion_widget.deleteLater()
        self._call_tip_widget.deleteLater()
        self.deleteLater()
        event.accept()
