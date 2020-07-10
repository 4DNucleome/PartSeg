import numpy as np
from napari._qt.qt_main_window import Window
from napari._qt.qt_viewer import QtViewer
from napari._qt.threading import create_worker
from napari.components import ViewerModel


class Viewer(ViewerModel):
    """Napari ndarray viewer.

    Parameters
    ----------
    title : string, optional
        The title of the viewer window. by default 'napari'.
    ndisplay : {2, 3}, optional
        Number of displayed dimensions. by default 2.
    order : tuple of int, optional
        Order in which dimensions are displayed where the last two or last
        three dimensions correspond to row x column or plane x row x column if
        ndisplay is 2 or 3. by default None
    axis_labels : list of str, optional
        Dimension names. by default they are labeled with sequential numbers
    show : bool, optional
        Whether to show the viewer after instantiation. by default True.
    """

    def __init__(
        self, title="napari", ndisplay=2, order=None, axis_labels=None, show=True,
    ):
        super().__init__(
            title=title, ndisplay=ndisplay, order=order, axis_labels=axis_labels,
        )
        qt_viewer = QtViewer(self)
        self.window = Window(qt_viewer, show=show)

    def update_console(self, variables):
        """Update console's namespace with desired variables.

        Parameters
        ----------
        variables : dict, str or list/tuple of str
            The variables to inject into the console's namespace.  If a dict, a
            simple update is done.  If a str, the string is assumed to have
            variable names separated by spaces.  A list/tuple of str can also
            be used to give the variable names.  If just the variable names are
            give (list/tuple/str) then the variable values looked up in the
            callers frame.
        """
        if self.window.qt_viewer.console is None:
            return
        else:
            self.window.qt_viewer.console.push(variables)

    def screenshot(self, path=None, *, canvas_only=True):
        """Take currently displayed screen and convert to an image array.

        Parameters
        ----------
        path : str
            Filename for saving screenshot image.
        canvas_only : bool
            If True, screenshot shows only the image display canvas, and
            if False include the napari viewer frame in the screenshot,
            By default, True.

        Returns
        -------
        image : array
            Numpy array of type ubyte and shape (h, w, 4). Index [0, 0] is the
            upper-left corner of the rendered region.
        """
        if canvas_only:
            image = self.window.qt_viewer.screenshot(path=path)
        else:
            image = self.window.screenshot(path=path)
        return image

    @staticmethod
    def update(func, *args, **kwargs):
        import warnings

        warnings.warn(
            "Viewer.update() is deprecated, use " "create_worker(func, *args, **kwargs) instead", DeprecationWarning,
        )
        return create_worker(func, *args, **kwargs, _start_thread=True)

    def show(self):
        """Resize, show, and raise the viewer window."""
        self.window.show()

    def close(self):
        """Close the viewer window."""
        self.window.close()

    def __str__(self):
        """Simple string representation"""
        return f"napari.Viewer: {self.title}"

    def calc_min_scale(self):
        return np.min([layer.scale for layer in self.layers], axis=0)

    def _new_labels(self):
        if self.dims.ndim == 0:
            dims = (512, 512)
        else:
            dims = self._calc_bbox()[1]
            scale = self.calc_min_scale()
            dims = [np.ceil(d / s).astype("int") if d > 0 else 1 for s, d in zip(scale, dims)]
            if len(dims) < 1:
                dims = (512, 512)
        empty_labels = np.zeros(dims, dtype=int)
        self.add_labels(empty_labels, scale=scale)
