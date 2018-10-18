from project_utils.progress_thread import ProgressTread
from .stack_settings import StackSettings
from tiff_image import ImageWriter
import os.path

class SaveResultThread(ProgressTread):
    def __init__(self, settings: StackSettings, dir_path: str,  parent=None):
        super().__init__(parent)
        self.image = settings.image
        self.components = settings.chosen_components()
        self.segmentation = settings.segmentation
        self.dir_path = dir_path

    def run(self):
        try:
            print(f"[run] {self.dir_path}")
            file_name = os.path.splitext(os.path.basename(self.image.file_path))[0]
            self.range_changed.emit(0, 2 * len(self.components))
            for i in self.components:
                im = self.image.cut_image(self.segmentation == i, replace_mask=True)
                print(f"[run] {im}")
                ImageWriter.save(im, os.path.join(self.dir_path, f"{file_name}_component{i}.tif"))
                self.step_changed.emit(2*i +1)
                ImageWriter.save_mask(im, os.path.join(self.dir_path, f"{file_name}_component{i}_mask.tif"))
                self.step_changed.emit(2 * i + 2)
        except Exception as e:
            self.error_signal.emit(e)