from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from tox.plugin import impl

if TYPE_CHECKING:
    from tox.config.sets import EnvConfigSet
    from tox.session.state import State


@impl
def tox_add_env_config(env_conf: EnvConfigSet, state: State) -> None:
    env_conf.add_constant(["sys_platform"], "string representing current OS", sys.platform)
    print("tox_add_env_config", env_conf, state)
