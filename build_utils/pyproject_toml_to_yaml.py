from pathlib import Path

import tomllib

base_dir = Path(__file__).parent.parent
pyproject_toml = base_dir / "pyproject.toml"

with pyproject_toml.open("rb") as f:
    data = tomllib.load(f)
dependencies = data["project"]["dependencies"]
dependencies_str = "\n".join([f"  - {v}" for v in dependencies])
dependencies_str = dependencies_str.replace("napari", "napari=*=*pyside2")
dependencies_str = dependencies_str.replace("qtconsole", "qtconsole-base")
print(dependencies_str)
