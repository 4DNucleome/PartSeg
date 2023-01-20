import os
from typing import Any

from tox.config.cli.parser import DEFAULT_VERBOSITY
from tox.execute.request import StdinSource
from tox.plugin import impl
from tox.tox_env.api import ToxEnv

HERE = os.path.dirname(__file__)


@impl
def tox_on_install(tox_env: ToxEnv, arguments: Any, section: str, of_type: str):  # noqa: ARG001
    if of_type == "deps" and section == "PythonRun":
        cmd = ["python", os.path.join(HERE, "build_utils", "create_minimal_constrains_file.py")]
        tox_env.execute(
            cmd=cmd,
            stdin=StdinSource.user_only(),
            run_id="create_minimal_requirements",
            show=tox_env.options.verbosity > DEFAULT_VERBOSITY,
        )
