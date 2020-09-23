import os
from configparser import ConfigParser

config = ConfigParser()
config.read(os.path.join(os.path.dirname(__file__), "..", "setup.cfg"))

minimum_requirements = config["options"]["install_requires"].replace(">=", "==")
minimum_qt = config["options.extras_require"]["pyqt"].replace(">=", "==")

with open(os.path.join(os.path.dirname(__file__), "minimal-req.txt"), "w") as f_p:
    f_p.write(minimum_requirements)
    f_p.write(minimum_qt)
    f_p.write("\n")
