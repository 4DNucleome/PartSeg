import codecs
import os
import re

from setuptools import setup

current_dir = os.path.dirname(os.path.abspath(__file__))
package_dir = os.path.join(current_dir, "package")


def read(*parts):
    with codecs.open(os.path.join(current_dir, *parts), "r") as fp:
        return fp.read()


def readme():
    this_directory = os.path.abspath(os.path.dirname(__file__))
    reg = re.compile(r"(!\[[^]]*\])\((images/[^)]*)\)")
    reg2 = re.compile(r"PartSeg-lastest")
    with open(os.path.join(this_directory, "Readme.md")) as f:
        text = f.read()
        text = reg.sub(r"\1(https://raw.githubusercontent.com/4DNucleome/PartSeg/master/\2)", text)
        try:
            from setuptools_scm import get_version

            text = reg2.sub(f"PartSeg-{get_version()}", text)
        except ImportError:
            pass
    with open(os.path.join(this_directory, "changelog.md")) as f:
        chg = f.read()
        text += "\n\n" + chg.replace("# ", "## ")
    return text


changelog_path = os.path.join(os.path.dirname(__file__), "changelog.md")
changelog_result_path = os.path.join(os.path.dirname(__file__), "package", "PartSeg", "changelog.py")
if os.path.exists(changelog_path):
    with open(changelog_path) as ff:
        changelog_str = ff.read()
    with open(changelog_result_path, "w") as ff:
        ff.write(f'changelog = """\n{changelog_str}"""\n')


setup(
    include_package_data=True,
    long_description=readme(),
    long_description_content_type="text/markdown",
    use_scm_version=True,
)
