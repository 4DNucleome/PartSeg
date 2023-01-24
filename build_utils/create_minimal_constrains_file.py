import os
from configparser import ConfigParser


def main():
    config = ConfigParser()
    config.read(os.path.join(os.path.dirname(__file__), "..", "setup.cfg"))

    minimum_requirements = config["options"]["install_requires"].replace(">=", "==")
    minimum_qt = config["options.extras_require"]["pyqt"].replace(">=", "==")

    with open(os.path.join(os.path.dirname(__file__), "minimum-constrains.txt"), "w") as f_p:
        f_p.write(minimum_requirements)
        f_p.write(minimum_qt)
        f_p.write("\nnpe2==0.1.1")
        f_p.write("\n")


if __name__ == "__main__":
    main()
