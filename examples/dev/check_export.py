from qtpy.QtWidgets import QApplication

from PartSeg._roi_analysis.batch_window import ExportProjectDialog
from PartSeg._roi_analysis.main_window import CONFIG_FOLDER
from PartSeg._roi_analysis.partseg_settings import PartSettings

app = QApplication([])

settings = PartSettings(CONFIG_FOLDER)
settings.load()
print(CONFIG_FOLDER)
dlg = ExportProjectDialog("", "", settings)
dlg.show()

app.exec_()

# settings.dump()
