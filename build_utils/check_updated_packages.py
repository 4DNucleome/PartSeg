import argparse
import re
import subprocess
import sys
from configparser import ConfigParser
from pathlib import Path

name_re = re.compile(r"\w+")
changed_name_re = re.compile(r"\+(\w+)")


src_dir = Path(__file__).parent.parent

parser = argparse.ArgumentParser()
parser.add_argument("--main-packages", action="store_true")
args = parser.parse_args()


out = subprocess.run(
    ["git", "diff", str(src_dir / "requirements" / "requirements_pyinstaller.txt")], capture_output=True
)

changed_packages = [changed_name_re.match(x)[1] for x in out.stdout.decode().split("\n") if changed_name_re.match(x)]

if not args.main_packages:
    print(", ".join(sorted(changed_packages)))
    sys.exit(0)


config = ConfigParser()
config.read(src_dir / "setup.cfg")
packages = config["options"]["install_requires"].split("\n")
packages = [name_re.match(package).group() for package in packages if name_re.match(package)]

print(", ".join(sorted(set(packages) & set(changed_packages))))
