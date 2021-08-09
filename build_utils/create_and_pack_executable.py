import argparse
import os
import platform
import shutil
import tarfile
import zipfile

from PyInstaller.__main__ import run as pyinstaller_run

import PartSeg

# if len(sys.argv) == 2:
#     base_path = os.path.abspath(sys.argv[1])
# else:
#     base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

parser = argparse.ArgumentParser("PartSeg build")
parser.add_argument(
    "--spec", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "launcher.spec")
)
parser.add_argument("--working-dir", default=os.path.abspath(os.curdir))

args = parser.parse_args()


pyinstaller_run(
    [
        "-y",
        "--debug=all",
        args.spec,
        "--distpath",
        os.path.join(args.working_dir, "dist"),
        "--workpath",
        os.path.join(args.working_dir, "dist"),
    ]
)


name_dict = {"Linux": "linux", "Windows": "windows", "Darwin": "macos"}

system_name = name_dict[platform.system()]

os.makedirs(os.path.join(args.working_dir, "dist2"), exist_ok=True)

if platform.system() == "Darwin":
    arch_file = tarfile.open(
        os.path.join(args.working_dir, "dist2", f"PartSeg-{PartSeg.__version__}-{system_name}.tgz"), "w:gz"
    )
    arch_file.write = arch_file.add
else:
    arch_file = zipfile.ZipFile(
        os.path.join(args.working_dir, "dist2", f"PartSeg-{PartSeg.__version__}-{system_name}.zip"),
        "w",
        zipfile.ZIP_DEFLATED,
    )
base_zip_path = os.path.join(args.working_dir, "dist")

dir_name = "PartSeg.app" if platform.system() == "Darwin2" else "PartSeg"

if platform.system() == "Darwin" and os.path.exists(os.path.join(args.working_dir, "dist", dir_name, "PyQt5", "Qt")):
    shutil.move(
        os.path.join(args.working_dir, "dist", dir_name, "PyQt5", "Qt"),
        os.path.join(args.working_dir, "dist", dir_name, "PyQt5", "Qt5"),
    )

for root, dirs, files in os.walk(os.path.join(args.working_dir, "dist", dir_name), topdown=False, followlinks=True):
    for file_name in files:
        arch_file.write(os.path.join(root, file_name), os.path.relpath(os.path.join(root, file_name), base_zip_path))

arch_file.close()
