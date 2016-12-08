import os
from PyInstaller import log as logging
from PyInstaller import compat
from os import listdir
 
libdir = compat.base_prefix + "/Library/bin"
mkllib = filter(lambda x: x.startswith('mkl_'), listdir(libdir))
other_libs = ['libiomp5md.dll']
 
if mkllib != []:
    logger = logging.getLogger(__name__)
    logger.info("MKL installed as part of numpy, importing that!")
    binaries = [(os.path.join(libdir, lib), '') for lib in ["mkl_avx.dll", "mkl_def.dll"]]
    # binaries = [(os.path.join(libdir, lib), '') for lib in mkllib]
    # binaries += [(os.path.join(libdir, lib), '') for lib in other_libs]
else:
    logger = logging.getLogger(__name__)
    logger.info("MKL not installed libdir: {}".format(libdir))