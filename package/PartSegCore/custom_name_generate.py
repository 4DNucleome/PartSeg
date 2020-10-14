import random
import string
from typing import Any, Dict, Set


def custom_name_generate(prohibited_names: Set[str], dict_names: Dict[str, Any]) -> str:
    for _ in range(1000):
        rand_name = "custom_" + "".join(random.choice(string.ascii_letters + string.digits) for _ in range(10))  # nosec
        if rand_name not in prohibited_names and rand_name not in dict_names:
            return rand_name
        raise RuntimeError("Cannot generate proper names")  # pragma: no cover
