from PyQt5.QtWidgets import QMainWindow, QPushButton, QHBoxLayout, QWidget


class MainWindow(QMainWindow):
    def __init__(self, title):
        super().__init__()
        self.setWindowTitle(title)
        self.partseg_button = QPushButton("Analysis GUI")
        self.stackseg_button = QPushButton("Segmentation GUI")
        self.partseg_button.clicked.connect(self.launch_partseg)
        self.stackseg_button.clicked.connect(self.launch_stackseg)
        layout = QHBoxLayout()
        layout.addWidget(self.partseg_button)
        layout.addWidget(self.stackseg_button)
        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

    def launch_partseg(self):
        from partseg2.main_window import MainWindow
        self.launch(MainWindow, "PartSeg")

    def launch_stackseg(self):
        from stackseg.stack_gui_main import MainWindow
        self.launch(MainWindow, "StackSeg")

    def window_shown(self):
        self.close()

    def launch(self, cls, title):
        print(self.parent())
        wind = cls(title, self.window_shown)
        wind.show()
        self.wind = wind
