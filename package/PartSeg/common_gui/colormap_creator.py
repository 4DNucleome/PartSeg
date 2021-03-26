import bisect
from functools import partial
from math import ceil
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
from napari.utils import Colormap
from qtpy.QtCore import QPointF, QRect, Qt, Signal
from qtpy.QtGui import (
    QBrush,
    QColor,
    QFont,
    QFontMetrics,
    QHideEvent,
    QMouseEvent,
    QPainter,
    QPaintEvent,
    QResizeEvent,
    QShowEvent,
)
from qtpy.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from PartSeg.common_backend.base_settings import ViewSettings
from PartSeg.common_gui.icon_selector import IconSelector
from PartSeg.common_gui.numpy_qimage import convert_colormap_to_image
from PartSeg.common_gui.universal_gui_part import InfoLabel
from PartSegCore.color_image.base_colors import Color
from PartSegCore.custom_name_generate import custom_name_generate


def color_from_qcolor(color: QColor) -> Color:
    """Convert :py:class:`PyQt5.QtGui.QColor` to :py:class:`.Color`"""
    return Color(color.red() / 255, color.green() / 255, color.blue() / 255, color.alpha() / 255)


def qcolor_from_color(color: Color) -> QColor:
    """Convert :py:class:`.Color` to :py:class:`PyQt5.QtGui.QColor`"""
    return QColor(int(color.red * 255), int(color.green * 255), int(color.blue * 255), int(color.alpha * 255))


class ColormapEdit(QWidget):
    """
    Preview of colormap. Double click used for add/remove colors. Single click on marker allows moving them
    """

    double_clicked = Signal(float)  # On double click emit signal with current position factor.

    def __init__(self):
        super().__init__()
        self.color_list: List[Color] = []
        self.position_list: List[float] = []
        self.move_ind = None
        self.image = convert_colormap_to_image(Colormap([(0, 0, 0)]))
        self.setMinimumHeight(60)

    def paintEvent(self, a0: QPaintEvent) -> None:
        painter = QPainter(self)
        margin = 10
        width = self.width() - 2 * margin
        rect = QRect(margin, margin, width, self.height() - 2 * margin)
        painter.drawImage(rect, self.image)
        painter.save()

        for pos_factor in self.position_list:
            pos = width * pos_factor
            point = QPointF(pos + margin, self.height() / 2)
            painter.setBrush(QBrush(Qt.black))
            painter.drawEllipse(point, 5, 5)
            painter.setBrush(QBrush(Qt.white))
            painter.drawEllipse(point, 3, 3)

        painter.restore()

    def refresh(self):
        """Recreate presented image and force repaint event """
        self.image = convert_colormap_to_image(self.colormap)
        self.repaint()

    def _get_color_ind(self, ratio) -> Optional[int]:
        ind = bisect.bisect_left(self.position_list, ratio)
        if len(self.position_list) > ind and abs(self.position_list[ind] - ratio) < 0.01:
            return ind
        if len(self.position_list) > 0 and ind > 0 and abs(self.position_list[ind - 1] - ratio) < 0.01:
            return ind - 1
        return None

    def _get_ratio(self, e: QMouseEvent, margin=10):
        frame_margin = 10
        width = self.width() - 2 * frame_margin
        if e.x() < margin or e.x() > self.width() - margin:
            return
        if e.y() < margin or e.y() > self.height() - margin:
            return
        return (e.x() - frame_margin) / width

    def mousePressEvent(self, e: QMouseEvent) -> None:
        ratio = self._get_ratio(e, 5)
        if ratio is None:
            return
        ind = self._get_color_ind(ratio)
        if ind is None:
            return
        self.move_ind = ind

    def mouseReleaseEvent(self, e: QMouseEvent) -> None:
        self._move_color(e)
        self.move_ind = None

    def _move_color(self, e: QMouseEvent) -> None:
        ratio = self._get_ratio(e)
        if ratio is None or self.move_ind is None:
            return
        self.position_list.pop(self.move_ind)
        ind = bisect.bisect_left(self.position_list, ratio)
        col = self.color_list.pop(self.move_ind)
        self.color_list.insert(ind, col)
        self.position_list.insert(ind, ratio)
        self.move_ind = ind
        self.refresh()

    def mouseMoveEvent(self, e: QMouseEvent) -> None:
        self._move_color(e)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        """
        If click near marker remove it. Otherwise emmit `double_click` signal with event position factor.
        """
        ratio = self._get_ratio(event)
        if ratio is None:
            return
        ind = self._get_color_ind(ratio)
        if ind is not None:
            self.position_list.pop(ind)
            self.color_list.pop(ind)
            self.refresh()
            return
        self.double_clicked.emit(ratio)

    def add_color(self, position: float, color: Color):
        """
        Add color to current colormap

        :param color: Color with position.

        """
        ind = bisect.bisect_left(self.position_list, position)
        self.color_list.insert(ind, color)
        self.position_list.insert(ind, position)
        self.refresh()

    def clear(self):
        """
        Remove color markers. Reset to initial state.
        """
        self.color_list = []
        self.position_list = []
        self.image = convert_colormap_to_image(Colormap([(0, 0, 0), (1, 1, 1)]))
        self.repaint()

    def distribute_evenly(self):
        """
        Distribute color markers evenly.
        """
        for i, pos in enumerate(np.linspace(0, 1, len(self.position_list))):
            self.position_list[i] = pos
        self.refresh()

    @property
    def colormap(self) -> Colormap:
        """colormap getter"""
        if len(self.color_list) == 0:
            return Colormap("black")
        return Colormap(colors=self.color_list, controls=self.position_list)

    @colormap.setter
    def colormap(self, val: Colormap):
        """colormap setter"""
        self.position_list = val.controls.tolist()
        self.color_list = val.colors.tolist()
        self.refresh()

    def reverse(self):
        self.position_list = [1 - x for x in reversed(self.position_list)]
        self.color_list = list(reversed(self.color_list))
        self.refresh()


class ColormapCreator(QWidget):
    """
    Widget for creating colormap.
    """

    colormap_selected = Signal(Colormap)
    """
    emitted on save button click. Contains current colormap in format accepted by :py:func:`create_color_map`
    """

    def __init__(self):
        super().__init__()
        self.color_picker = QColorDialog()
        self.color_picker.setWindowFlag(Qt.Widget)
        self.color_picker.setOptions(QColorDialog.DontUseNativeDialog | QColorDialog.NoButtons)
        self.show_colormap = ColormapEdit()
        self.clear_btn = QPushButton("Clear")
        self.save_btn = QPushButton("Save")
        self.distribute_btn = QPushButton("Distribute evenly")
        self.reverse_btn = QPushButton("Reverse")
        self.info_label = InfoLabel(
            [
                "<strong>Tip:</strong> Select color and double click on a color bar below. "
                + "Repeat to add another colors.",
                "<strong>Tip:</strong> Double click on a marker to remove it.",
                "<strong>Tip:</strong> Press and hold mouse left button on a marker to drag and drop it on color bar.",
            ],
            delay=10000,
        )
        layout = QVBoxLayout()
        layout.addWidget(self.color_picker)
        layout.addWidget(self.info_label)
        layout.addWidget(self.show_colormap)
        btn_layout = QHBoxLayout()
        btn_layout.addStretch(1)
        btn_layout.addWidget(self.reverse_btn)
        btn_layout.addWidget(self.distribute_btn)
        btn_layout.addWidget(self.clear_btn)
        btn_layout.addWidget(self.save_btn)
        layout.addLayout(btn_layout)
        self.setLayout(layout)
        self.show_colormap.double_clicked.connect(self.add_color)
        self.clear_btn.clicked.connect(self.show_colormap.clear)
        self.save_btn.clicked.connect(self.save)
        self.reverse_btn.clicked.connect(self.show_colormap.reverse)
        self.distribute_btn.clicked.connect(self.show_colormap.distribute_evenly)

    def add_color(self, pos):
        color = self.color_picker.currentColor()
        self.show_colormap.add_color(pos, color_from_qcolor(color))

    def save(self):
        if self.show_colormap.colormap:
            self.colormap_selected.emit(self.show_colormap.colormap)

    def current_colormap(self) -> Colormap:
        """:return: current colormap"""
        return self.show_colormap.colormap

    def set_colormap(self, colormap: Colormap):
        """set current colormap"""
        self.show_colormap.colormap = colormap
        self.show_colormap.refresh()


class PColormapCreator(ColormapCreator):
    """
    :py:class:`~.ColormapCreator` variant which save result in :py:class:`.ViewSettings`
    """

    def __init__(self, settings: ViewSettings):
        super().__init__()
        self.settings = settings
        for i, el in enumerate(settings.get_from_profile("custom_colors", [])):
            self.color_picker.setCustomColor(i, qcolor_from_color(Color(*el)))
        self.prohibited_names = set(self.settings.colormap_dict.keys())  # Prohibited name is added to reduce
        # probability of colormap cache collision

    def _save_custom_colors(self):
        colors = [color_from_qcolor(self.color_picker.customColor(i)) for i in range(self.color_picker.customCount())]
        self.settings.set_in_profile("custom_colors", colors)

    def hideEvent(self, a0: QHideEvent) -> None:
        """Save custom colors on hide"""
        self._save_custom_colors()

    def save(self):
        if self.show_colormap.colormap:
            rand_name = custom_name_generate(self.prohibited_names, self.settings.colormap_dict)
            self.prohibited_names.add(rand_name)
            colors = list(self.show_colormap.colormap.colors)
            positions = list(self.show_colormap.colormap.controls)
            if positions[0] != 0:
                positions.insert(0, 0)
                colors.insert(0, colors[0])
            if positions[-1] != 1:
                positions.append(1)
                colors.append(colors[-1])
            self.settings.colormap_dict[rand_name] = Colormap(colors=np.array(colors), controls=np.array(positions))
            self.settings.chosen_colormap_change(rand_name, True)
            self.colormap_selected.emit(self.settings.colormap_dict[rand_name][0])


_icon_selector = IconSelector()


class ChannelPreview(QWidget):
    """
    class for preview single colormap. Witch checkbox for change selection.

    :param colormap: colormap to show
    :param accepted: if checkbox should be checked
    :param name: name which will be emitted in all signals as firs argument
    """

    selection_changed = Signal(str, bool)
    """checkbox selection changed (name)"""
    edit_request = Signal([str], [Colormap])
    """send after pressing edit signal (name) (ColorMap object)"""
    remove_request = Signal(str)
    """Signal with name of colormap (name)"""

    def __init__(self, colormap: Colormap, accepted: bool, name: str, removable: bool = False, used: bool = False):
        super().__init__()
        self.image = convert_colormap_to_image(colormap)
        self.name = name
        self.removable = removable
        self.checked = QCheckBox()
        self.checked.setChecked(accepted)
        self.checked.setDisabled(used)
        self.label = QLabel(name)
        self.label.setFixedWidth(150)
        self.setMinimumWidth(80)
        metrics = QFontMetrics(QFont())
        layout = QHBoxLayout()
        layout.addWidget(self.checked)
        layout.addStretch(1)
        self.remove_btn = QToolButton()
        self.remove_btn.setIcon(_icon_selector.close_icon)
        if removable:
            self.remove_btn.setToolTip("Remove colormap")
        else:
            self.remove_btn.setToolTip("This colormap is protected")
        self.remove_btn.setEnabled(not accepted and self.removable)

        self.edit_btn = QToolButton()
        self.edit_btn.setIcon(_icon_selector.edit_icon)
        layout.addWidget(self.remove_btn)
        layout.addWidget(self.edit_btn)
        layout.addWidget(self.label)
        self.setLayout(layout)
        self.checked.stateChanged.connect(self._selection_changed)
        self.edit_btn.clicked.connect(partial(self.edit_request.emit, name))
        if len(colormap.controls) < 20:
            self.edit_btn.clicked.connect(partial(self.edit_request[Colormap].emit, colormap))
            self.edit_btn.setToolTip("Create colormap base on this")
        else:
            self.edit_btn.setDisabled(True)
            self.edit_btn.setToolTip("This colormap is not editable")
        self.remove_btn.clicked.connect(partial(self.remove_request.emit, name))
        self.setMinimumHeight(max(metrics.height(), self.edit_btn.minimumHeight(), self.checked.minimumHeight()) + 20)
        self.setToolTip(self.name)

    def _selection_changed(self, _=None):
        chk = self.checked.isChecked()
        self.selection_changed.emit(self.name, chk)
        self.remove_btn.setEnabled(not chk and self.removable)

    def set_blocked(self, block):
        """Set if block possibility of remove or uncheck """
        self.checked.setDisabled(block)
        if self.removable and not block:
            self.remove_btn.setToolTip("Remove colormap")
        else:
            self.remove_btn.setToolTip("This colormap is protected")
            self.remove_btn.setDisabled(True)

    @property
    def state_changed(self):
        """Inner checkbox stateChanged signal"""
        return self.checked.stateChanged

    @property
    def is_checked(self):
        """If colormap is selected"""
        return self.checked.isChecked()

    def set_chosen(self, state: bool):
        """Set selection of check box."""
        self.checked.setChecked(state)

    def paintEvent(self, event: QPaintEvent):
        painter = QPainter(self)
        start = 2 * self.checked.x() + self.checked.width()
        end = self.remove_btn.x() - self.checked.x()
        rect = self.rect()
        rect.setX(start)
        rect.setWidth(end - start)
        painter.drawImage(rect, self.image)
        super().paintEvent(event)


class ColormapList(QWidget):
    """
    Show list of colormaps
    """

    edit_signal = Signal(Colormap)
    """Colormap for edit"""

    remove_signal = Signal(str)
    """Name of colormap to remove"""

    visibility_colormap_change = Signal(str, bool)
    """Hide or show colormap"""

    def __init__(self, colormap_map: Dict[str, Tuple[Colormap, bool]], selected: Optional[Iterable[str]] = None):
        super().__init__()
        self._selected = set() if selected is None else set(selected)
        self._blocked = set()
        self.current_columns = 1
        self.colormap_map = colormap_map
        self._widget_dict: Dict[str, ChannelPreview] = {}
        self.scroll_area = QScrollArea()
        self.central_widget = QWidget()
        layout2 = QVBoxLayout()
        self.grid_layout = QGridLayout()
        layout2.addLayout(self.grid_layout)
        layout2.addStretch(1)
        layout2.setContentsMargins(0, 0, 0, 0)

        self.central_widget.setLayout(layout2)
        self.central_widget.setMinimumWidth(300)
        self.scroll_area.setWidget(self.central_widget)
        self.scroll_area.setWidgetResizable(True)

        layout = QVBoxLayout()
        layout.addWidget(self.scroll_area)
        self.setLayout(layout)

    def showEvent(self, event: QShowEvent):
        self.refresh()

    def get_selected(self) -> Set[str]:
        """Already selected colormaps"""
        return set(self._selected)

    def change_selection(self, name, selected):
        if selected:
            self._selected.add(name)
        else:
            self._selected.remove(name)
        self.visibility_colormap_change.emit(name, selected)

    def blocked(self) -> Set[str]:
        """Channels that cannot be turn of and remove"""
        return self._blocked

    def _get_columns(self):
        return max(1, self.width() // 500)

    def resizeEvent(self, event: QResizeEvent):
        if self._get_columns() != self.current_columns:
            self.refresh()
            self.central_widget.repaint()

    def refresh(self):
        layout: QGridLayout = self.grid_layout
        cache_dict: Dict[str, ChannelPreview] = {}
        self._widget_dict = {}
        for _ in range(layout.count()):
            el: ChannelPreview = layout.takeAt(0).widget()
            if el.name in self.colormap_map:
                cache_dict[el.name] = el
            else:
                el.deleteLater()
                el.edit_request[Colormap].disconnect()
                el.remove_request.disconnect()
                el.selection_changed.disconnect()
        selected = self.get_selected()
        blocked = self.blocked()
        columns = self._get_columns()
        for i, (name, (colormap, removable)) in enumerate(self.colormap_map.items()):
            if name in cache_dict:
                widget = cache_dict[name]
                widget.set_blocked(name in blocked)
                widget.set_chosen(name in selected)
            else:
                widget = ChannelPreview(colormap, name in selected, name, removable=removable, used=name in blocked)
                widget.edit_request[Colormap].connect(self.edit_signal)
                widget.remove_request.connect(self._remove_request)
                widget.selection_changed.connect(self.change_selection)
            layout.addWidget(widget, i // columns, i % columns)
            self._widget_dict[name] = widget
        widget: QWidget = layout.itemAt(0).widget()
        height = widget.minimumHeight()
        self.current_columns = columns
        self.central_widget.setMinimumHeight((height + 10) * ceil(len(self.colormap_map) / columns))

    def check_state(self, name: str) -> bool:
        """
        Check state of widget representing given colormap

        :param name: name of colormap which representing widget should be checked
        """
        return self._widget_dict[name].is_checked

    def set_state(self, name: str, state: bool) -> None:
        """
        Set if given colormap is selected

        :param name: name of colormap
        :param state: state to be set
        """
        self._widget_dict[name].set_chosen(state)

    def get_colormap_widget(self, name) -> ChannelPreview:
        """Access to widget showing colormap. Created for testing purpose."""
        return self._widget_dict[name]

    def _remove_request(self, name):
        _, removable = self.colormap_map[name]
        if not removable:
            raise ValueError(f"ColorMap {name} is protected from remove")
        if name not in self.colormap_map:
            raise ValueError(f"color with name {name} not found in any dict")
        del self.colormap_map[name]
        self.refresh()


class PColormapList(ColormapList):
    """
    Show list of colormaps. Integrated with :py:class:`.ViewSettings`

    :param settings: used for store state
    :param control_names: list of names of :py:class:`PartSeg.common_gui.stack_image_view.ImageView`
        for protect used channels from uncheck or remove
    """

    def __init__(self, settings: ViewSettings, control_names: List[str]):
        super().__init__(settings.colormap_dict)
        settings.colormap_dict.colormap_removed.connect(self.refresh)
        settings.colormap_dict.colormap_added.connect(self.refresh)
        settings.colormap_changes.connect(self.refresh)
        self.settings = settings
        self.color_names = control_names

    def get_selected(self) -> Set[str]:
        return set(self.settings.chosen_colormap)

    def change_selection(self, name, selected):
        self.visibility_colormap_change.emit(name, selected)
        self.settings.chosen_colormap_change(name, selected)

    def blocked(self) -> Set[str]:
        # TODO check only currently presented channels
        blocked = set()
        for el in self.color_names:
            num = self.settings.get_from_profile(f"{el}.channels_count", 0)
            for i in range(num):
                blocked.add(self.settings.get_channel_info(el, i))
        return blocked

    def _change_colormap_visibility(self, name, visible):
        colormaps = set(self.settings.chosen_colormap)
        if visible:
            colormaps.add(name)
        else:
            try:
                colormaps.remove(name)
            except KeyError:
                pass
        self.settings.chosen_colormap = list(sorted(colormaps))
