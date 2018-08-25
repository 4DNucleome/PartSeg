from PyQt5.QtGui import QHideEvent, QShowEvent

from common_gui.stack_image_view import ImageView

class RawImageView(ImageView):
    pass

    def hideEvent(self, a0: QHideEvent):
        self.parent().layout().setColumnStretch(0, 0)

    def showEvent(self, event: QShowEvent):
        self.parent().layout().setColumnStretch(0, 1)

class ResultImageView(ImageView):
    pass
