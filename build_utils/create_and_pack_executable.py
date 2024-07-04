import argparse
import logging
import os
import platform
import shutil
import tarfile
import zipfile

import tqdm
from PyInstaller.__main__ import run as pyinstaller_run

import PartSeg

logger = logging.getLogger("bundle build")
logger.setLevel(logging.INFO)

SYSTEM_NAME_DICT = {"Linux": "linux", "Windows": "windows", "Darwin": "macos"}


def get_file_path(working_dir, with_version=True):
    if with_version:
        file_name = f"PartSeg-{PartSeg.__version__}-{SYSTEM_NAME_DICT[platform.system()]}"
    else:
        file_name = f"PartSeg-{SYSTEM_NAME_DICT[platform.system()]}"
    if platform.system() == "Darwin" and os.uname().machine == "arm64":
        file_name += "-arm64"
    if platform.system() != "Darwin":
        file_name += ".zip"
    else:
        file_name += ".tgz"

    return os.path.join(working_dir, "dist2", file_name)


def create_archive(working_dir):
    os.makedirs(os.path.join(working_dir, "dist2"), exist_ok=True)
    file_path = get_file_path(working_dir)
    if file_path.endswith(".zip"):
        return zipfile.ZipFile(file_path, "w", zipfile.ZIP_DEFLATED)
    arch_file = tarfile.open(file_path, "w:gz")
    arch_file.write = arch_file.add
    return arch_file


def fix_qt_location(working_dir, dir_name):
    if platform.system() == "Darwin" and os.path.exists(os.path.join(working_dir, "dist", dir_name, "PyQt5", "Qt")):
        shutil.move(
            os.path.join(working_dir, "dist", dir_name, "PyQt5", "Qt"),
            os.path.join(working_dir, "dist", dir_name, "PyQt5", "Qt5"),
        )


def create_bundle(spec_path, working_dir):
    pyinstaller_args = [
        "-y",
        spec_path,
        "--distpath",
        os.path.join(working_dir, "dist"),
        "--workpath",
        os.path.join(working_dir, "build"),
    ]
    logger.info("run PyInstaller %s", " ".join(pyinstaller_args))

    pyinstaller_run(pyinstaller_args)


def archive_build(working_dir, dir_name):
    base_zip_path = os.path.join(working_dir, "dist")

    with create_archive(working_dir) as arch_file:
        for root, _dirs, files in tqdm.tqdm(
            os.walk(os.path.join(working_dir, "dist", dir_name), topdown=False, followlinks=True)
        ):
            for file_name in files:
                arch_file.write(
                    os.path.join(root, file_name), os.path.relpath(os.path.join(root, file_name), base_zip_path)
                )
    shutil.copy(get_file_path(working_dir), get_file_path(working_dir, with_version=False))


def main():
    parser = argparse.ArgumentParser("PartSeg build")
    parser.add_argument(
        "--spec", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "launcher.spec")
    )
    parser.add_argument("--working-dir", default=os.path.abspath(os.curdir))

    args = parser.parse_args()

    create_bundle(args.spec, args.working_dir)

    dir_name = "PartSeg.app" if platform.system() == "Darwin2" else "PartSeg"

    fix_qt_location(args.working_dir, dir_name)
    archive_build(args.working_dir, dir_name)


if __name__ == "__main__":
    main()
