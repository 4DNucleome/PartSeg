from qtpy import QT5
from qtpy.QtGui import QMouseEvent

if QT5:

    def get_mouse_x(event: QMouseEvent) -> float:
        return event.x()

    def get_mouse_y(event: QMouseEvent) -> float:
        return event.y()

else:

    def get_mouse_x(event: QMouseEvent) -> float:
        return event.position().x()

    def get_mouse_y(event: QMouseEvent) -> float:
        return event.position().x()
