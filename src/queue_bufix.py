# bugfix in Anaconda python2 pyinstaller
import sys

if sys.version_info.major == 2:
    from Queue import *
else:
    from queue import *