import sys
__author__ = "Grzegorz Bokota"

import os
import signal

def sig_handler(signum, frame):
    print ("segfault")

# signal.signal(signal.SIGSEGV, sig_handler)

def my_excepthook(type, value, tback):
    # log the exception here

    # then call the default handler
    sys.__excepthook__(type, value, tback)

sys.excepthook = my_excepthook


def main():
    import logging
    logging.basicConfig(level=logging.INFO)
    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtCore import QThread
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