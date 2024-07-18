from pathlib import Path

import tomllib

base_dir = Path(__file__).parent.parent
pyproject_toml = base_dir / "pyproject.toml"


def drop_line(line):
    if "python_version < '3.10'" in line:
        return False

    if "python_version < '3.11'" in line:  # noqa: SIM103
        return False

    return True


def remove_specifier(line: str):
    if ";" in line:
        return line.split(";", maxsplit=1)[0]
    return line


with pyproject_toml.open("rb") as f:
    data = tomllib.load(f)
dependencies = data["project"]["dependencies"]
dependencies = map(remove_specifier, filter(drop_line, dependencies))
dependencies_str = "\n".join([f"  - {v}" for v in dependencies])
dependencies_str = dependencies_str.replace("qtconsole", "qtconsole-base")
print(dependencies_str)
