import os
import tifffile as tif
import appdirs
from PyQt5.QtWidgets import QMainWindow, QLabel

from common_gui.channel_control import ChannelControl
from project_utils.global_settings import static_file_folder
from .partseg_settings import PartSettings
from .image_view import RawImageView, ResultImageView


app_name = "PartSeg2"
app_lab = "LFSG"
config_folder = appdirs.user_data_dir(app_name, app_lab)


class MainWindow(QMainWindow):
    def __init__(self, title):
        super(MainWindow, self).__init__()
        self.setWindowTitle(title)
        self.settings = PartSettings()
        if os.path.exists(os.path.join(config_folder, "settings.json")):
            self.settings.load(os.path.join(config_folder, "settings.json"))
        self.main_menu = MainMenu(self.settings)
        self.channel_control = ChannelControl(self.settings)
        self.raw_image = RawImageView(self.settings, self.channel_control)
        self.result_image = ResultImageView(self.settings, self.channel_control)
        self.info_text = QLabel()
        self.image_view.text_info_change.connect(self.info_text.setText)
        image_view_control = self.image_view.get_control_view()
        self.options_panel = Options(self.settings, image_view_control, self.image_view)
        self.main_menu.image_loaded.connect(self.image_read)
        self.settings.image_changed.connect(self.image_read)

        im = tif.imread(os.path.join(static_file_folder, 'initial_images', "stack.tif"))