import codecs
import subprocess
import os
import zipfile
import platform
import re
import sys

if len(sys.argv) == 2:
    base_path = os.path.abspath(sys.argv[1])
else:
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

os.chdir(base_path)

subprocess.call(["python", "-m", "PyInstaller", "-y", "--debug=all", "launcher.spec"])


def read(*parts):
    with codecs.open(os.path.join(base_path, *parts), 'r') as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


version = find_version(base_path, "package", "PartSeg", "__init__.py")

name_dict = {"Linux": "linux", "Windows": "windows", "Darwin": "macos"}

system_name = name_dict[platform.system()]

os.makedirs(os.path.join(base_path, "dist2"), exist_ok=True)

zip_file = zipfile.ZipFile(os.path.join(base_path, "dist2", f"PartSeg-{version}-{system_name}.zip"), 'w',
                           zipfile.ZIP_DEFLATED)
base_zip_path = os.path.join(base_path, "dist")

for root, dirs, files in os.walk(os.path.join(base_path, "dist", "PartSeg"), topdown=False):
    for file_name in files:
        zip_file.write(os.path.join(root, file_name), os.path.relpath(os.path.join(root, file_name), base_zip_path))

zip_file.close()
