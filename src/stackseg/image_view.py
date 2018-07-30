from PyQt5.QtCore import QEvent
from PyQt5.QtGui import QHelpEvent
from PyQt5.QtWidgets import QToolTip

from common_gui.stack_image_view import ImageView


class StackImageView(ImageView):
    def component_click(self, point, size):
        if self.labels_layer is None:
            return
        x = int(point.x() / size.width() * self.image_shape.width())
        y = int(point.y() / size.height() * self.image_shape.height())
        num = self.labels_layer[self.stack_slider.value(), y, x]
        if num > 0:
            self.component_clicked.emit(num)

    def event(self, event: QEvent):

        if event.type() == QEvent.ToolTip and self.component is not None:
            text = str(self.component)
            assert(isinstance(event, QHelpEvent))
            if self._settings.component_is_chosen(self.component):
                text = "☑{}".format(self.component)
            else:
                text = "☐{}".format(self.component)
            QToolTip.showText(event.globalPos(), text)
        return super(ImageView, self).event(event)
