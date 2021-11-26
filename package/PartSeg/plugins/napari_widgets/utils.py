import inspect
import itertools
import typing

from magicgui.widgets import Widget, create_widget
from napari import Viewer
from napari.layers import Image as NapariImage
from napari.layers import Labels
from qtpy.QtWidgets import QWidget

from PartSeg.common_gui.algorithms_description import FormWidget, QtAlgorithmProperty
from PartSeg.common_gui.custom_save_dialog import FormDialog
from PartSegCore import UNIT_SCALE, Units
from PartSegCore.algorithm_describe_base import AlgorithmProperty
from PartSegCore.channel_class import Channel
from PartSegImage import Image


class QtNapariAlgorithmProperty(QtAlgorithmProperty):
    @classmethod
    def _get_field_from_value_type(cls, ap: AlgorithmProperty) -> typing.Union[QWidget, Widget]:
        if inspect.isclass(ap.value_type) and issubclass(ap.value_type, Channel):
            return create_widget(annotation=NapariImage, label="Image", options={})
        return super()._get_field_from_value_type(ap)


class NapariFormWidget(FormWidget):
    @staticmethod
    def _element_list(fields) -> typing.Iterable[QtAlgorithmProperty]:
        return map(QtNapariAlgorithmProperty.from_algorithm_property, fields)

    def reset_choices(self, event=None):
        for widget in self.widgets_dict.values():
            if hasattr(widget.get_field(), "reset_choices"):
                widget.get_field().reset_choices(event)


class NapariFormWidgetWithMask(NapariFormWidget):
    def _element_list(self, fields) -> typing.Iterable[QtAlgorithmProperty]:
        mask = AlgorithmProperty("mask", "Mask", None, value_type=typing.Optional[Labels])
        return super()._element_list(itertools.chain([mask], fields))


class NapariFormDialog(FormDialog):
    @staticmethod
    def widget_class() -> typing.Type[FormWidget]:
        return NapariFormWidget

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.widget.reset_choices()


def generate_image(viewer: Viewer, *layer_names):
    axis_order = Image.axis_order.replace("C", "")
    image_list = []
    for name in dict.fromkeys(layer_names):
        image_layer = viewer.layers[name]
        data_scale = image_layer.scale[-3:] / UNIT_SCALE[Units.nm.value]
        image_list.append(
            Image(
                image_layer.data,
                data_scale,
                axes_order=axis_order[-image_layer.data.ndim :],
                channel_names=[image_layer.name],
            )
        )
    res_image = image_list[0]
    for image in image_list[1:]:
        res_image = res_image.merge(image, "C")

    return res_image
