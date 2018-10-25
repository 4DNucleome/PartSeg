import sys
__author__ = "Grzegorz Bokota"
from project_utils.except_hook import my_excepthook

def sig_handler(signum, frame):
    print ("segfault")

# signal.signal(signal.SIGSEGV, sig_handler)


sys.excepthook = my_excepthook


def main():
    import logging
    logging.basicConfig(level=logging.INFO)
    from PyQt5.QtWidgets import QApplication
    from stackseg.stack_gui_main import MainWindow
    import sys
    myApp = QApplication(sys.argv)
    wind = MainWindow("StackSeg")
    wind.show()
    myApp.exec_()
    del wind
    sys.exit()

if __name__ == '__main__':
    main()