import codecs
import contextlib
import os
import re

from setuptools import setup

current_dir = os.path.dirname(os.path.abspath(__file__))
package_dir = os.path.join(current_dir, "package")


def read(*parts):
    with codecs.open(os.path.join(current_dir, *parts)) as fp:
        return fp.read()


def readme():
    this_directory = os.path.abspath(os.path.dirname(__file__))
    reg = re.compile(r"(!\[[^]]*\])\((images/[^)]*)\)")
    reg2 = re.compile(r"releases/latest/download/PartSeg")
    with open(os.path.join(this_directory, "Readme.md"), encoding="utf8") as f:
        text = f.read()
        text = reg.sub(r"\1(https://raw.githubusercontent.com/4DNucleome/PartSeg/master/\2)", text)
        with contextlib.suppress(ImportError):
            from setuptools_scm import get_version

            text = reg2.sub(f"releases/download/v{get_version()}/PartSeg-{get_version()}", text)

    with open(os.path.join(this_directory, "changelog.md"), encoding="utf8") as f:
        chg = f.read()
        text += "\n\n" + chg.replace("# ", "## ")
    return text


changelog_path = os.path.join(os.path.dirname(__file__), "changelog.md")
changelog_result_path = os.path.join(os.path.dirname(__file__), "package", "PartSeg", "changelog.py")
if os.path.exists(changelog_path):
    with open(changelog_path, encoding="utf8") as ff:
        changelog_str = ff.read()
    with open(changelog_result_path, "w", encoding="utf8") as ff:
        ff.write(f'changelog = """\n{changelog_str}"""\n')


setup(
    include_package_data=True,
    long_description=readme(),
    long_description_content_type="text/markdown",
)
